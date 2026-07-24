# LLM baseline (`EQ_llm_predict`)

Serializes patient event histories as text, asks a locally-served LLM for per-row
event-occurrence probabilities, and writes `PredictionSchema`-conformant parquets — a
drop-in alternative to `EQ_predict` whose output feeds `EQ_evaluate` unchanged.

## Serving the model (vLLM)

Launch the server yourself before running `EQ_llm_predict`; the CLI only talks to an
OpenAI-compatible endpoint (`base_url`, default `http://localhost:8000/v1`).

**Dev / pipeline debugging** — Llama 3.1 8B Instruct, single GPU:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --port 8000 \
    --max-model-len 16384
```

**Real baseline run** — Llama 3.1 70B Instruct across 4 GPUs:

```bash
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --port 8000 \
    --tensor-parallel-size 4 \
    --max-model-len 16384
```

Pin `--revision` when serving and record it via `model_revision=...` so the run metadata
captures exactly which weights answered.

> **Size `--max-model-len` to your prompts.** `max_events` counts unique timestamps, and a
> single timestamp can expand to many code lines — so long histories (especially with
> `code_descriptions` on) can exceed 16k tokens. If a prompt overflows the model length,
> vLLM returns a 400 and the run aborts. Dry-run first (below) to eyeball prompt size, and
> raise `--max-model-len` (e.g. `32768`, up to the model's native limit) if needed.

## Running

Eyeball prompts first (no GPU / server contact):

```bash
EQ_llm_predict \
    tasks_dir=/path/to/tasks \
    tensorized_cohort_dir=/path/to/tensorized_cohort \
    output_dir=/path/to/out \
    dry_run=true
```

Smoke run against the 8B server:

```bash
EQ_llm_predict \
    tasks_dir=/path/to/tasks \
    tensorized_cohort_dir=/path/to/tensorized_cohort \
    output_dir=/path/to/out \
    model=meta-llama/Llama-3.1-8B-Instruct \
    limit=100
```

Full 70B run (restartable — rerun the same command with `resume=true` after a crash;
completed rows in `{output_dir}/shards/` are skipped):

```bash
EQ_llm_predict \
    tasks_dir=/path/to/tasks \
    tensorized_cohort_dir=/path/to/tensorized_cohort \
    output_dir=/path/to/out \
    model=meta-llama/Llama-3.1-70B-Instruct \
    max_concurrency=32 \
    resume=true
```

Tuning the sampling knobs (finer resolution + code descriptions + a touch of smoothing):

```bash
EQ_llm_predict \
    tasks_dir=/path/to/tasks \
    tensorized_cohort_dir=/path/to/tensorized_cohort \
    output_dir=/path/to/out \
    model=meta-llama/Llama-3.1-8B-Instruct \
    n_samples=40 \
    temperature=0.8 \
    smoothing=1.0 \
    code_descriptions=/path/to/tensorized_cohort/metadata/codes.parquet
```

`n_samples=40` gives 41 score levels (vs 21 at the default 20); `temperature=0.8` spreads
the vote; `smoothing=1.0` pulls scores off the exact 0/1 endpoints; `code_descriptions=`
annotates raw codes. Leave `guided_choice=true` (the default) unless your server lacks vLLM
guided decoding, in which case add `guided_choice=false`.

Then evaluate exactly as for `EQ_predict` output:

```bash
EQ_evaluate predictions_parquet=/path/to/out/predictions.parquet metrics_parquet=/path/to/metrics.parquet
```

## Hosted APIs (OpenAI) and SLURM

Nothing here is vLLM-specific except `guided_choice`, so the same CLI drives a hosted
OpenAI-compatible endpoint. Point `base_url` at it, supply a real `api_key`, and turn
guided decoding off:

```bash
EQ_llm_predict … \
    base_url=https://api.openai.com/v1 \
    api_key="$OPENAI_API_KEY" \
    model=<chat-model-with-n-support> \
    guided_choice=false \
    max_concurrency=4
```

Two constraints on model choice. The method needs `n > 1` on chat completions (all samples
come from one `n`-way request), and it needs `temperature > 0` — which rules out
reasoning/o-series endpoints, and conveniently means any model that satisfies the first
constraint also accepts `max_tokens`/`temperature`/`seed` as sent. With `guided_choice=false`
the leading Yes/No word is parsed leniently, so watch `n_unparsed` in `details.parquet`: it is
structurally zero under guided decoding and will not be here.

Size `max_concurrency` to your rate-limit tier — exhausting `max_retries` on 429s aborts the
run by design (transport failures are never written as fallback probabilities).

A ready-to-edit CPU-only batch job lives at
[`scripts/eq_llm_predict_openai.sbatch`](../../../scripts/eq_llm_predict_openai.sbatch) —
no GPU, no model serving. All settings are inline in one `EDIT ME` block; the API key is
read from a mode-600 file rather than the tracked script. It preflights egress to the API
(compute nodes are often firewalled), and defaults `batch_size` low because the in-flight
buffer is flushed on Ctrl-C but **not** on SLURM's time-limit `SIGTERM` — so `batch_size`
bounds the rows whose API spend is lost to a timeout. `resume=true` makes it idempotent
under `--requeue`.

Do not run a job array against a single `output_dir`: there is no shard/offset knob, so every
task would compute the same pending set and duplicate rows into `shards/`. Parallelize by
splitting `tasks_dir` into separate `output_dir`s and merging afterward.

## Configuration reference

All settings are Hydra overrides passed as `key=value` on the command line. Defaults and
inline comments live in [`configs/llm_predict.yaml`](configs/llm_predict.yaml); this is the
at-a-glance version.

**Required** (no default — every run must set these three, plus `split`):

| Key | What it is |
| --- | --- |
| `tasks_dir` | Directory of `TaskQuerySchema` parquets (same contract as `EQ_predict`). |
| `tensorized_cohort_dir` | The tokenized/tensorized MEDS cohort (`EQ_process_data` output). |
| `output_dir` | Where shards stream and the merged `predictions.parquet` / `details.parquet` land. |
| `split` | Which cohort split the task subjects live in. Default `held_out`. |

**LLM server:**

| Key | Default | What it is |
| --- | --- | --- |
| `base_url` | `http://localhost:8000/v1` | OpenAI-compatible endpoint — your vLLM server, or a hosted API. |
| `api_key` | `EMPTY` | Ignored by vLLM (the SDK just requires a non-empty value); a real key for hosted APIs. |
| `model` | `meta-llama/Llama-3.1-8B-Instruct` | Served model name — must match what vLLM was launched with. |
| `model_revision` | `null` | HF revision, recorded in metadata (not used to make requests). |

**Sampling / probability** (see the next section for the method):

| Key | Default | What it is |
| --- | --- | --- |
| `n_samples` | `20` | Yes/No samples drawn per query; score has `n_samples + 1` levels. |
| `temperature` | `0.7` | Sampling temperature. **Must be > 0** (see next section). |
| `guided_choice` | `true` | Constrain each answer to exactly `Yes`/`No` via vLLM guided decoding. |
| `smoothing` | `0.0` | Laplace prior on the vote; `0.0` = raw fraction. |
| `fallback_prob` | `null` (→0.5) | Probability written on total parse failure. |
| `extra_body` | `null` | Extra JSON merged into every request (e.g. disable thinking). |

**Serialization** (the two ablation axes):

| Key | Default | What it is |
| --- | --- | --- |
| `max_events` | `256` | Keep the most recent N unique timestamps; older dropped with a marker. |
| `code_descriptions` | `null` | `.json` map or `.parquet` (`code`/`description`) to annotate raw codes. |

**Execution / restarts:**

| Key | Default | What it is |
| --- | --- | --- |
| `max_concurrency` | `16` | In-flight requests to the server (semaphore-bounded). |
| `batch_size` | `500` | Completed rows per output shard flush. |
| `request_timeout` | `120` | Per-request timeout (seconds). |
| `max_retries` | `5` | Retry budget for transient transport errors (exp. backoff), then abort. |
| `seed` | `0` | Sampling seed, recorded in metadata. |
| `limit` | `null` | Cap rows processed (after resume filtering) — for smoke runs. |
| `resume` | `false` | Skip rows already in `{output_dir}/shards/` instead of refusing to start. |
| `dry_run` | `false` | Print ~3 serialized prompts and exit without contacting the server. |

## Probability extraction (repeated-sampling vote)

The model is instructed to answer with exactly one word — `Yes` or `No` — and is sampled
`n_samples` times (default 20) at a non-zero `temperature` (default 0.7). The **empirical
fraction of `Yes` answers** is the occurrence probability (see `llm_predict.py`). All samples
come back from a single `n`-way completion, so the (long) patient-history prefix is prefilled
once and shared across the samples.

Notes worth knowing about:

- **Temperature must be > 0.** At `temperature=0` every sample is identical and the
  probability collapses to 0 or 1; construction raises when `n_samples > 1` and
  `temperature == 0`.
- **Resolution is discrete.** With `n` samples the score takes one of `n + 1` values, so more
  samples buy finer resolution and lower Monte-Carlo variance at linear decode cost. For
  AUROC the empirical fraction and its logit are interchangeable (the transform is monotonic).
- **`guided_choice` (default on).** Each answer is constrained to exactly `Yes`/`No` via vLLM
  guided decoding, so every sample counts and there is nothing to parse-fail on. Set
  `guided_choice=false` for servers without that extension — the leading Yes/No word is then
  parsed leniently, and any other output is counted as unparsed (excluded from the vote).
- **`smoothing`** applies a symmetric Laplace prior `(n_yes + s) / (n_yes + n_no + 2s)`,
  keeping the score off the exact 0/1 endpoints. `0.0` (default) is the raw fraction —
  irrelevant for AUROC, but a small positive value keeps log-loss / Brier finite downstream.

On total parse failure (no sample produced a usable Yes/No) the configured `fallback_prob` is
written (default 0.5; set it explicitly to inject an externally-estimated prevalence — it is
never computed from the evaluated split's labels, which would leak ground truth), and the row
is flagged `parse_failed=True` in `details.parquet`, alongside the per-row `n_yes` / `n_no` /
`n_unparsed` vote counts. Transport failures (after bounded retries with exponential backoff)
abort the run instead of writing fallbacks — restart with `resume=true`.

`censor_prob` is a constant `0.0`: censoring is a data artifact, not a clinical
prediction. In `EQ_evaluate` this only affects the `censor_auroc` sanity metric, which
becomes a well-defined-but-uninformative 0.5 for constant scores; the headline
`occurs_auroc` drops censored rows before scoring and is unaffected.

## Ablations

- **Code descriptions**: raw codes like `ICD//E11.9` are near-opaque to a general instruct
  model. Point `code_descriptions=` at a `.json` mapping or a `.parquet` with
  `code`/`description` columns (a MEDS `metadata/codes.parquet` works directly) to render
  `ICD//E11.9 (Type 2 diabetes mellitus without complications)`. Leave `null` (default) for
  the raw-code arm.
- **History length**: `max_events` controls left-truncation (most recent events kept, with
  an omission marker). Per-run truncation counts/fractions are logged.

## Reproducibility

Every shard and the merged `predictions.parquet` / `details.parquet` carry a JSON blob
under the `every_query_llm_baseline` parquet metadata key: model name/revision, sampling
config (`n_samples`, `temperature`, `smoothing`, `guided_choice`), prompt-template version +
hash, `max_events`, whether code descriptions were used, seed, and a hash of the resolved
config. Read it back with:

```python
import json, pyarrow.parquet as pq
json.loads(pq.read_schema("predictions.parquet").metadata[b"every_query_llm_baseline"])
```

The prompt template lives only in `serialize.py`, is versioned
(`PROMPT_TEMPLATE_VERSION`), and any wording change changes the recorded hash.
