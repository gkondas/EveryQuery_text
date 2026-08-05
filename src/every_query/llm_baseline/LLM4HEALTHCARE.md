# The `llm4healthcare` prompt style

How `EQ_llm_predict` reproduces the prompt from
[yhzhu99/llm4healthcare](https://github.com/yhzhu99/llm4healthcare) on top of EveryQuery's
MEDS event stream, why the design is shaped the way it is, and what it costs.

Written for review — the decisions worth arguing about are called out in
[Things to sanity-check](#things-to-sanity-check) at the end.

---

## 1. What this is

`EQ_llm_predict` now has two serializations, selected by `prompt_style`:

| | `event_stream` (default, unchanged) | `llm4healthcare` (new) |
|---|---|---|
| Layout | chronological list of code strings | feature-major rows over a fixed panel |
| Vocabulary | open — every code in the window | closed — 17 panel variables |
| Values | baked into the code string | real numbers, with units and ref ranges |
| Answer | Yes/No × N-sample vote | float probability |
| Comparable to | EveryQuery's own prior runs | that paper's published numbers |

Everything downstream is shared: sharding, `resume`, output schema, reproducibility metadata,
and `EQ_evaluate` consumption. Switching styles changes only how a history becomes text and
how an answer becomes a number.

```bash
EQ_llm_predict \
    tasks_dir=~/data/MIMIC_MEDS_DEMO/eq_tasks/eval/held_out \
    tensorized_cohort_dir=~/data/MIMIC_MEDS_DEMO/eq_processed/final \
    output_dir=/tmp/l4h \
    prompt_style=llm4healthcare \
    n_samples=1 \
    dry_run=true
```

---

## 2. The problem

Their `{DETAIL}` block is **feature-major**: one line per clinical variable, values
comma-joined across visits.

```
- Heart Rate: "94.0, 105.0, 97.0, 100.0"
- Mean blood pressure: "83.4, 79.5, 73.9, 73.0"
```

A MEDS stream is **event-major** over an open vocabulary: at each timestamp, some set of
codes fired. Turning one into the other is a pivot — and the naive pivot does not fit in a
context window.

Measured on one held_out patient in the MIMIC-IV demo cohort:

| | cells / lines | notes |
|---|---|---|
| event-major (today) | 1,915 lines | one line per measurement |
| naive pivot (all codes × all events) | **102,336 cells** | 492 distinct codes × 208 events, ~98% `nan` |

A 53× blowup of mostly-empty cells. That approach is dead.

**But that is not what they do either.** Their mimic-iv feature axis is the 17-variable
Harutyunyan ICU benchmark panel — small, fixed, and dense. Their entire format presumes a
narrow panel. So fidelity *requires* pinning the feature axis, and the question becomes
whether their specific panel can be expressed over this cohort.

It can.

---

## 3. The three things that made it work

### 3.1 Their panel exists in your vocabulary

All 17 of their variables resolve to codes in the processed cohort, with units already
embedded in the code string:

```
Heart Rate          LAB//220045//bpm            68/100 subjects, 10 bins
Respiratory rate    LAB//220210//insp/min       79
Mean blood pressure LAB//220181//mmHg           74
Oxygen saturation   LAB//220277//%              77
Temperature         LAB//223761//°F             63   (+ 223762 °C, 11)
GCS eye/verbal/motor  220739 / 223900 / 223901  77 / 67 / 77
… and the rest
```

So their feature **names**, their **ordering**, and their `unit.json` / `range.json` strings
transfer verbatim. `panels/mimic_icu17.yaml` is that mapping.

Two panel rows are legitimately empty here: `Glascow coma scale total` (itemid 226755 is
absent from the cohort) and `Capillary refill rate` (present but with no value bins). Their
own published example prompts show these as all-`nan` rows too, so this matches rather than
deviates.

### 3.2 Real numeric values are recoverable

This is the one I expected to block the whole thing. Preprocessing bins numerics into the
code string and keeps only a z-score in `numeric_value` — which sounds lossy. But the z-score
is **within-bin**, and `metadata/codes.parquet` stores per-binned-code `values/sum` and
`values/sum_sqd`:

```
LAB//220045//bpm//value_[90.0,96.0)   mean = 92.40   std = 1.72
LAB//220045//bpm//value_[85.0,90.0)   mean = 86.94   std = 1.40
```

So the original reading comes back exactly:

```
raw = numeric_value × std_bin + mean_bin
```

Per-bin standard deviations are 1–2 bpm for heart rate, so recovery is essentially exact.
Cells render as `94.0`, not as `[90.0,96.0)`. The decoder covers **5,347** binned codes in
this cohort.

Fallbacks, in order: per-bin stats → bin midpoint parsed from the code string (finite edge
for an outer half-open bin) → presence-only. Nothing fabricates precision silently; the
fallback path is exercised by tests.

### 3.3 Visits are just timestamps

Their tjh prompts use irregular real dates (`2020-02-09, 2020-02-10, 2020-02-13`), so "one
unique MEDS timestamp = one visit" is faithful. No resampling needed. Their mimic-iv prompts
render visit indices (`0, 1, 2, 3`), which is the default here (`record_time_mode=index`);
`days_before` is available if you want the irregular spacing surfaced.

---

## 4. How the pivot works

Per `(subject_id, prediction_time)` group, once — the panel is query-independent, so the
whole block is computed once and shared across that group's ~12 queries.

1. **Load** the untruncated pre-prediction window via `load_subject_data` (dense `code`,
   `numeric_value`, `time_delta_days`, `dim1/mask`, plus static codes).
2. **Match** each measurement against the panel. A source is keyed by `itemid` (matching
   `PREFIX//<itemid>//…`) or by `code` (exact match on the code with any `//value_` suffix
   stripped). Off-panel codes are dropped.
3. **Decode** the value: `z × std_bin + mean_bin`, then apply the source's `transform`
   (`f_to_c`, `fio2_fraction`).
4. **Resolve collisions**: within a timestamp, the earliest-listed source in the panel spec
   wins (so non-invasive BP beats arterial BP, Celsius beats Fahrenheit).
5. **Emit a visit** for each timestamp holding ≥1 panel measurement. Timestamps with only
   off-panel codes are dropped — they would be all-`nan` columns costing tokens for nothing.
6. **Truncate** to the most recent `max_visits`.
7. **Impute** per `impute` (see below).
8. **Render** demographics + `{DETAIL}` in the chosen `form`.

Demographics: sex from the `GENDER//*` static code (needs `static_inclusion_mode=include`,
which `__main__` now sets); age from the elapsed time between the `MEDS_BIRTH` event and the
end of the window. Both degrade to placeholders rather than fabricating a value.

---

## 5. The template, slot by slot

Their `USERPROMPT` has eight slots. A test asserts our rendering template is a **pure
refactor** of theirs — the only structural change is collapsing the four patient-header lines
and `{DETAIL}` into one `{PATIENT_BLOCK}` slot, because that part is query-independent and is
rendered once per group.

| Slot | Filled from | Fidelity |
|---|---|---|
| system turn | their ICU persona | verbatim |
| `INPUT_FORMAT_DESCRIPTION` | `l4h.form` + `l4h.impute` note | verbatim |
| `TASK_DESCRIPTION_AND_RESPONSE_FORMAT` | **ours** | see §6 |
| "I do not know" clause | — | verbatim |
| `UNIT_RANGE_CONTEXT` | panel `unit` / `range` strings | verbatim |
| `EXAMPLE` | `l4h.examples` | their numbering/preamble |
| `SEX` / `AGE` | `GENDER//*` static, `MEDS_BIRTH` | exact |
| `LENGTH` / `RECORD_TIME_LIST` | panel visits | exact |
| `DETAIL` | the pivot | exact |
| `RESPONSE_FORMAT` | **ours** | see §6 |

Three `form`s, all theirs:

```
string    - Heart Rate: "94.0, 88.0"        (their default)
list      - Heart Rate: [94.0, 88.0]
batches   Visit 1: / - Heart Rate: 94.0     (one block per visit)
```

Three `impute` modes, mapping onto their `0 | 1 | 2`:

```
none         gaps stay `nan`, appends their missing-value note
locf         forward-fill silently                              (their default)
locf_marked  forward-fill, suffix filled cells `**`, appends their LOCF caveat
```

LOCF is forward-only — a variable never measured before a given visit stays missing.

---

## 6. Deliberate deviations

Four, all forced.

**Task and response strings.** Their tasks are mortality / length-of-stay / readmission.
EveryQuery asks whether an arbitrary code occurs within a horizon. There is no faithful
transfer, so these two slots are written fresh in their register — same sentence shape, same
closing "respond with only a floating-point number…" clause. This was your call.

**Categorical cells render as numbers.** Their Glasgow-coma and capillary-refill cells hold
MIMIC's *text* labels ("Obeys Commands", "Spontaneously"). This cohort's preprocessing kept
only the numeric subscore, so those rows show `4.0`, `6.0`, `5.0`. The text is not present in
the tensorized cohort — not recoverable without re-running preprocessing.

**Capillary refill renders as `present`.** It has no value bins at all in this cohort, so
only presence/absence survives.

**FiO2 units — a real bug I found.** MIMIC charts itemid 223835 as a *percentage* (30–100),
but their reference range says "more than 0.21", a *fraction*. Rendering the raw value would
contradict the panel's own stated range in the same prompt. The panel applies the MIMIC
benchmark's normalization (`fio2_fraction`: divide by 100 when > 1). Flagging because it is
the one place I changed a number rather than passing it through.

---

## 7. What it costs

Three consequences you should weigh, all measured on the MIMIC-IV demo cohort.

**~12% of windows are empty.** 9 of 76 prediction groups contain no measurement of *any*
panel variable and render as a demographics-only prompt. That is the panel restriction
biting. The run logs this as a warning with the rate.

**The panel never shows the query code.** By your instruction the 17 rows are identical
regardless of what is being asked, so the prompt cannot reveal whether the query code
occurred before — usually the strongest single predictor for this task. Their template never
had this problem because their targets are extrinsic to the feature set.

**Prefix caching is lost.** Their template puts the query in the task-description paragraph
near the *top*. So two queries over the same patient no longer share a token prefix, and
vLLM's automatic prefix caching cannot reuse the patient block across a group's ~12 queries.
The `event_stream` style appends the query last and does keep that prefix shared.

Prompt sizes are comfortable either way (approximate, chars/4):

| form | `max_visits` | median | p90 | max |
|---|---|---|---|---|
| `string` | 48 | 1,479 | 1,890 | 1,918 |
| `string` | 16 | 1,046 | 1,058 | 1,062 |
| `batches` | 48 | 4,323 | 6,206 | 6,234 |

Visit counts per window: median 37 (non-empty), max 374 — which is why `max_visits` exists at
all. Their cohorts arrive pre-truncated to a handful of visits.

---

## 8. Answer extraction

`l4h.response_format` picks how a completion becomes a probability.

**`float` (default for this style).** Parses a probability from each sample and averages.
With `n_samples=1` this is exactly their single-shot `float(result)`. With more samples it is
the same estimator with lower variance. Parsing is slightly more permissive than theirs — the
first float-looking token anywhere in the answer, clamped to [0,1] — which survives a stray
newline or trailing period without changing the prompt. Their `I do not know` escape hatch is
not a number, so it is a parse failure and falls back (their constant is `0.501`).

**`yes_no`.** Keeps their prompt but swaps the two task slots for a Yes/No instruction and
extracts with the repeated-sampling vote the `event_stream` style uses. Not the reference
prompt, but better calibrated — see `llm_predict.py`'s module docstring on why free-text
floats mode-collapse to 0.1/0.5/0.9.

`guided_choice` is forced off in float mode; leaving it on would pin every answer to the
literal string "Yes" or "No" and destroy the number. This is asserted by a test.

Diagnostics land in `details.parquet`: `parse_failed`, `n_parsed` (float mode), and
`n_yes` / `n_no` / `n_unparsed` (vote mode).

---

## 9. Code map

```
serialize_l4h.py       template constants (verbatim + ours), Panel/PanelFeature/PanelSource,
                       ValueDecoder, the pivot, LOCF, the three renderers, demographics,
                       L4HConfig, prompt_template_hash
panels/mimic_icu17.yaml  their 17 features -> this cohort's itemids, with their unit/range strings
prompt_style.py        PromptStyle protocol; EventStreamStyle + LLM4HealthcareStyle; build_style
llm_predict.py         parse_probability + the float branch; LLMPredictor takes a style
__main__.py            wiring, static inclusion, style-aware metadata and fingerprinting
configs/llm_predict.yaml  prompt_style + the l4h.* block
```

Tests: `tests/test_llm_baseline_serialize_l4h.py` (34, golden strings + the fidelity guard),
plus two end-to-end CLI tests in `tests/test_llm_predict_cli.py`. Full suite: 388 passing.

---

## 10. Reproducibility

`resume` refuses to extend shards written under a different prompt. The fingerprint now
covers `prompt_style`, `prompt_template_version`, `prompt_template_hash`, `response_format`,
and `style_fields` (which carries the panel digest, form, imputation, `max_visits`, precision,
record-time mode, and shot count). Editing a unit string or an itemid mapping changes the
panel digest, which changes the hash, which blocks the resume. A style mismatch is covered by
a test.

---

## 11. Extending it

Copy `panels/mimic_icu17.yaml`, edit, point `l4h.panel` at the file (or drop it in `panels/`
and use its bare name). Sources key on `itemid` for MIMIC-style codes or `code` for an exact
match — the latter is what makes a panel expressible over a cohort that does not embed
itemids, and is what the CI tests use.

Feature order in the YAML is the rendered order and is part of the prompt.

---

## 12. Things to sanity-check

The parts where I made a judgment call, or where the data surprised me:

1. **FiO2 normalization** (§6) — the one place a number is transformed rather than passed
   through. Reasonable, but it is a silent correction of their range string vs MIMIC's
   encoding.
2. **`float` as the default response format** — most faithful to the paper, and the paper's
   own `0.501` fallback suggests the parse fails often. If the goal is the best baseline
   rather than the closest replication, `yes_no` is the better default.
3. **`max_visits=48`** — no counterpart in their config; I picked it from the token
   measurements. Median window is 37 visits so most are untouched, but the 374-visit tail is
   cut hard.
4. **`value_precision=1`** — their prompts carry full float repr (`37.05555555555556`).
   One decimal is far cheaper and clinically equivalent, but it is a deviation. It also makes
   FiO2 coarse (0.2/0.3/0.4), though those are the bin edges anyway.
5. **The 12% empty-window rate** (§7) — is a demographics-only prompt an acceptable row to
   score, or should those be excluded from the comparison?
6. **Few-shot is wired but empty.** `l4h.examples` works; I did not populate it. Their
   published examples are hand-written for mortality prediction so they need regenerating for
   this task — **from the training split only**, since an example drawn from held_out leaks
   its label into every prompt.
7. **`Glascow` (sic)** is their misspelling, preserved deliberately. Fixing it would silently
   change the prompt being replicated.
