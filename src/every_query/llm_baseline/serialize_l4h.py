"""``llm4healthcare``-style serialization of patient event streams into LLM prompts.

Reproduces the prompt used by https://github.com/yhzhu99/llm4healthcare ("Prompting Large
Language Models for Zero-Shot Clinical Prediction with Structured Longitudinal Electronic
Health Record Data") on top of EveryQuery's MEDS event stream.  Every template constant in
this module is copied verbatim from that repo's ``prompts/prompt.py`` — including the
misspelling of "Glascow" carried by the panel spec — so a rendered prompt is byte-comparable
with theirs apart from the two slots that provably cannot transfer (below).

The structural gap this module bridges: their ``{DETAIL}`` block is **feature-major** over a
small fixed panel of clinical variables, one line per feature with values comma-joined across
visits.  A MEDS stream is **event-major** over an open code vocabulary.  Three properties of
the tensorized cohort make the pivot exact rather than approximate:

1. **The panel exists.**  All 17 of their MIMIC variables are present in the cohort
   vocabulary as ``LAB//<itemid>//<unit>`` codes — see ``panels/mimic_icu17.yaml``.
2. **Values are recoverable.**  Preprocessing bins numerics into the code string
   (``LAB//220045//bpm//value_[90.0,96.0)``) and stores a *within-bin* z-score in
   ``numeric_value``.  Since ``metadata/codes.parquet`` carries per-binned-code
   ``values/sum`` and ``values/sum_sqd``, the original measurement is recovered as
   ``numeric_value * std_bin + mean_bin`` (see :class:`ValueDecoder`).  Per-bin standard
   deviations are small (~1-2 bpm for heart rate), so recovery is essentially exact.
3. **Visits are timestamps.**  Their tjh prompts use irregular real dates, so one MEDS
   unique timestamp = one "visit" is faithful; no resampling is needed.

Two slots deliberately deviate, because no faithful transfer exists:

- ``{TASK_DESCRIPTION_AND_RESPONSE_FORMAT}`` / ``{RESPONSE_FORMAT}``: their tasks are
  mortality / length-of-stay / readmission.  EveryQuery asks whether an arbitrary query code
  occurs within a horizon, so these two strings are written fresh in their register (same
  sentence shape, same closing "respond with only a floating-point number..." clause).
- Categorical cells: their Glasgow-coma and capillary-refill cells hold MIMIC's *text*
  labels ("Obeys Commands").  This cohort's preprocessing discarded those labels and kept
  only the numeric subscore, so those rows render as numbers.  Capillary refill retains no
  value bins at all and renders as presence.

Note the panel is **query-independent by construction**: the same 17 rows are rendered
regardless of which code is being asked about, so a group's ``{DETAIL}`` block is computed
once and shared across all of that group's queries.

The template is versioned via :data:`PROMPT_TEMPLATE_VERSION` and fingerprinted by
:func:`prompt_template_hash`, which covers the panel spec as well as the template strings —
editing a unit string or an itemid mapping changes the hash and so blocks a ``resume`` onto
shards built with the old one.
"""

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from every_query.llm_baseline.serialize import SerializedHistory

# ── Template constants (verbatim from llm4healthcare/prompts/prompt.py) ─────────────────

PROMPT_TEMPLATE_VERSION = "l4h-1"

SYSTEM_PROMPT = "You are an experienced doctor in Intensive Care Unit (ICU) treatment."

# Their ``USERPROMPT`` verbatim, kept as the provenance record and asserted against the
# rendering template below by ``tests/test_llm_baseline_serialize_l4h.py``.  Not used to
# render: the four patient-header lines and ``{DETAIL}`` are query-independent, so they are
# rendered once per (subject, prediction_time) group and injected as one ``{PATIENT_BLOCK}``.
REFERENCE_USER_PROMPT = """I will provide you with medical information from multiple Intensive Care Unit (ICU) visits of a patient, each characterized by a fixed number of features.

{INPUT_FORMAT_DESCRIPTION}

{TASK_DESCRIPTION_AND_RESPONSE_FORMAT}

In situations where the data does not allow for a reasonable conclusion, respond with the phrase `I do not know` without any additional explanation.

{UNIT_RANGE_CONTEXT}

{EXAMPLE}

Now please predict the patient below:
The patient is a {SEX}, aged {AGE} years.
The patient had {LENGTH} visits that occurred at {RECORD_TIME_LIST}.
Details of the features for each visit are as follows:

{DETAIL}

{RESPONSE_FORMAT}
RESPONSE:
"""

# The four header lines plus {DETAIL} of REFERENCE_USER_PROMPT, collapsed into one slot.
PATIENT_BLOCK = """The patient is a {SEX}, aged {AGE} years.
The patient had {LENGTH} visits that occurred at {RECORD_TIME_LIST}.
Details of the features for each visit are as follows:

{DETAIL}"""

USER_PROMPT = """I will provide you with medical information from multiple Intensive Care Unit (ICU) visits of a patient, each characterized by a fixed number of features.

{INPUT_FORMAT_DESCRIPTION}

{TASK_DESCRIPTION_AND_RESPONSE_FORMAT}

In situations where the data does not allow for a reasonable conclusion, respond with the phrase `I do not know` without any additional explanation.

{UNIT_RANGE_CONTEXT}

{EXAMPLE}

Now please predict the patient below:
{PATIENT_BLOCK}

{RESPONSE_FORMAT}
RESPONSE:
"""

INPUT_FORMAT_DESCRIPTION = {
    "string": (
        "Present multiple visit data of a patient in one batch. Represent each feature "
        "within this data as a string of values, separated by commas."
    ),
    "list": (
        "Display multiple visit data of a patient in one batch, expressing each feature as "
        "a list of values, separated by commas."
    ),
    "batches": (
        "Organize visit data of a patient into separate batches, each batch corresponding to one visit."
    ),
}

MISSING_VALUE_DESCRIPTION = " Missing values are represented as `nan`."

INSTRUCTING_MISSING_VALUE = (
    "Values followed by `**` are initially not a number and imputed through Last Observation "
    "Carried Forward(LOCF) method. If a large number of values of a certain feature are "
    "filled, you need to consider that the credibility of the analysis results for that "
    "feature is relatively low."
)

# ── EveryQuery task strings (written in their register; see module docstring) ────────────

TASK_DESCRIPTION = (
    "Your task is to assess the provided medical data and analyze the health records from "
    "ICU visits to determine the likelihood that the clinical event `{code}` will be "
    "recorded for this patient within {days} days of their final visit. Please respond with "
    "only a floating-point number between 0 and 1, where a higher number suggests a greater "
    "likelihood of the event occurring."
)

RESPONSE_FORMAT = (
    "Please respond with only a floating-point number between 0 and 1, where a higher number "
    "suggests a greater likelihood of the event occurring. Do not include any additional "
    "explanation."
)

# Yes/No variant — not part of the reference prompt.  Selected by ``response_format=yes_no``
# for compatibility with the repeated-sampling vote in ``llm_predict.LLMPredictor``, whose
# module docstring explains why free-text floats mode-collapse.
TASK_DESCRIPTION_YES_NO = (
    "Your task is to assess the provided medical data and analyze the health records from "
    "ICU visits to determine whether the clinical event `{code}` will be recorded for this "
    "patient within {days} days of their final visit. Please answer with exactly one word: "
    "Yes or No."
)

RESPONSE_FORMAT_YES_NO = "Answer with exactly one word: Yes or No. Do not include any additional explanation."

MISSING_MARKER = "nan"
PRESENCE_MARKER = "present"
LOCF_MARKER = "**"
UNKNOWN_SEX = "patient of unrecorded sex"
UNKNOWN_AGE = "unknown"
NO_VISITS_LINE = "(no measurements of these features were recorded for this patient)"

# ``LAB//<itemid>//<unit>[//value_[lo,hi)]`` — the only code shape the panel maps.
_CODE_RE = re.compile(r"^[^/]+//(?P<itemid>\d+)//")
_VALUE_BIN_RE = re.compile(r"//value_\[(?P<lo>[^,]+),(?P<hi>[^)]+)\)$")

_SEX_BY_CODE = {"GENDER//M": "male", "GENDER//F": "female"}
_BIRTH_CODE = "MEDS_BIRTH"
_DAYS_PER_YEAR = 365.25


# ── Panel spec ───────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PanelSource:
    """One MEDS code feeding a panel feature, with an optional unit conversion.

    A source identifies its code either by ``itemid`` — matching any code of the shape
    ``PREFIX//<itemid>//…``, the convention MIMIC-derived cohorts use — or by ``code``, an
    exact match against the code string with any ``//value_[a,b)`` bin suffix stripped.  The
    itemid form is sugar for the common case; the ``code`` form is what makes a panel
    expressible over a cohort that does not embed itemids.  Exactly one must be given.
    """

    itemid: int | None = None
    transform: str | None = None
    code: str | None = None

    def __post_init__(self) -> None:
        if (self.itemid is None) == (self.code is None):
            raise ValueError(
                f"Panel source must set exactly one of 'itemid' or 'code', got "
                f"itemid={self.itemid!r}, code={self.code!r}."
            )

    @property
    def key(self) -> tuple[str, int | str]:
        """Lookup key: ``("itemid", 220045)`` or ``("code", "HR")``."""
        return ("itemid", self.itemid) if self.itemid is not None else ("code", self.code)

    def convert(self, value: float) -> float:
        """Apply the source's unit transform.

        ``fio2_fraction`` reproduces the MIMIC benchmark's FiO2 cleaning: the itemid is charted
        inconsistently as either a percentage (40) or a fraction (0.4), so anything above 1 is
        divided by 100.  Without it the rendered value contradicts the panel's own reference
        range ("more than 0.21").

        Examples:
            >>> PanelSource(223761, "f_to_c").convert(98.6)
            37.0
            >>> PanelSource(223762).convert(37.0)
            37.0
            >>> PanelSource(223835, "fio2_fraction").convert(40.0)
            0.4
            >>> PanelSource(223835, "fio2_fraction").convert(0.4)
            0.4

            A source keyed by code string rather than itemid:

            >>> PanelSource(code="HR").key
            ('code', 'HR')
            >>> PanelSource(itemid=1, code="HR")
            Traceback (most recent call last):
                ...
            ValueError: Panel source must set exactly one of 'itemid' or 'code'...
        """
        match self.transform:
            case None:
                return value
            case "f_to_c":
                return (value - 32.0) * 5.0 / 9.0
            case "fio2_fraction":
                return value / 100.0 if value > 1.0 else value
            case _:
                raise ValueError(f"Unknown panel transform {self.transform!r}")


@dataclass(frozen=True)
class PanelFeature:
    """One row of the ``{DETAIL}`` block: a named clinical variable and its MEDS sources."""

    name: str
    unit: str
    range: str
    sources: tuple[PanelSource, ...]
    kind: str = "numeric"  # "numeric" | "presence"


@dataclass(frozen=True)
class Panel:
    """An ordered fixed feature panel — the feature axis of the pivot.

    The order is the rendered order, and is part of the prompt: the reference implementation
    lists categorical variables before numeric ones, which ``mimic_icu17.yaml`` preserves.
    """

    name: str
    features: tuple[PanelFeature, ...]

    @property
    def source_index(self) -> dict[tuple[str, int | str], tuple[PanelFeature, PanelSource]]:
        """Reverse index from a source key to its ``(feature, source)`` pair.

        Examples:
            >>> panel = Panel("t", (PanelFeature("Heart Rate", "u", "r", (PanelSource(220045),)),))
            >>> feature, source = panel.source_index[("itemid", 220045)]
            >>> feature.name, source.itemid
            ('Heart Rate', 220045)
        """
        return {s.key: (f, s) for f in self.features for s in f.sources}

    def spec_digest(self) -> str:
        """Stable digest of the panel, folded into :func:`prompt_template_hash`."""
        payload = [
            [f.name, f.unit, f.range, f.kind, [[s.itemid, s.code, s.transform] for s in f.sources]]
            for f in self.features
        ]
        return hashlib.sha256(json.dumps([self.name, payload], sort_keys=True).encode()).hexdigest()[:16]


def load_panel(path: Path) -> Panel:
    """Load a panel spec from YAML.

    Raises on an empty panel or a duplicated source — one code feeding two features would make
    the pivot order-dependent, which is a spec bug rather than something to resolve silently.
    """
    raw = yaml.safe_load(Path(path).read_text())
    features = []
    for entry in raw.get("features") or []:
        sources = tuple(
            PanelSource(
                itemid=None if s.get("itemid") is None else int(s["itemid"]),
                code=s.get("code"),
                transform=s.get("transform"),
            )
            for s in entry.get("sources") or []
        )
        features.append(
            PanelFeature(
                name=entry["name"],
                unit=entry.get("unit", ""),
                range=entry.get("range", ""),
                sources=sources,
                kind=entry.get("kind", "numeric"),
            )
        )
    if not features:
        raise ValueError(f"Panel spec {path} defines no features.")

    seen: dict[tuple[str, int | str], str] = {}
    for feature in features:
        for source in feature.sources:
            if source.key in seen:
                kind, value = source.key
                raise ValueError(
                    f"Panel spec {path}: {kind} {value} is claimed by both "
                    f"{seen[source.key]!r} and {feature.name!r} — each source must map to "
                    f"exactly one feature."
                )
            seen[source.key] = feature.name
    return Panel(name=raw.get("name", Path(path).stem), features=tuple(features))


# ── Value recovery ───────────────────────────────────────────────────────────────────────


@dataclass
class ValueDecoder:
    """Recovers original measurements from binned codes plus within-bin z-scores.

    Preprocessing bins a numeric measurement into the code string and normalizes the residual
    against *that bin's* mean and standard deviation, both of which survive in
    ``metadata/codes.parquet`` as ``values/sum`` / ``values/sum_sqd`` / ``values/n_occurrences``.
    So ``raw = z * std + mean`` inverts the transform exactly.

    When a code has no usable statistics (``values/n_occurrences == 0``, or a variance that
    floating-point cancellation drove negative), the decoder falls back to the bin midpoint
    parsed out of the code string, and to the finite edge for a half-open outer bin.  A bin
    with two infinite edges, or a code with no bin at all, yields ``None`` (presence only).
    """

    stats: dict[str, tuple[float, float]] = field(default_factory=dict)

    @classmethod
    def from_codes_parquet(cls, path: Path) -> "ValueDecoder":
        """Build the per-code ``(mean, std)`` table from a MEDS ``metadata/codes.parquet``."""
        import polars as pl

        df = pl.read_parquet(
            path, columns=["code", "values/n_occurrences", "values/sum", "values/sum_sqd"]
        ).filter(pl.col("values/n_occurrences") > 0)
        mean = pl.col("values/sum") / pl.col("values/n_occurrences")
        variance = pl.col("values/sum_sqd") / pl.col("values/n_occurrences") - mean**2
        df = df.with_columns(mean.alias("_mean"), variance.alias("_var"))
        stats = {
            code: (m, math.sqrt(v) if v is not None and v > 0.0 else 0.0)
            for code, m, v in zip(df["code"], df["_mean"], df["_var"], strict=True)
            if m is not None and not math.isnan(m)
        }
        return cls(stats=stats)

    @staticmethod
    def bin_midpoint(code: str) -> float | None:
        """Midpoint of a code's ``//value_[lo,hi)`` suffix, or the finite edge of an outer bin.

        Examples:
            >>> ValueDecoder.bin_midpoint("LAB//220045//bpm//value_[90.0,96.0)")
            93.0
            >>> ValueDecoder.bin_midpoint("LAB//220045//bpm//value_[116.0,inf)")
            116.0
            >>> ValueDecoder.bin_midpoint("LAB//220045//bpm//value_[-inf,68.0)")
            68.0
            >>> ValueDecoder.bin_midpoint("LAB//224308//UNK") is None
            True
        """
        m = _VALUE_BIN_RE.search(code)
        if not m:
            return None
        try:
            lo, hi = float(m.group("lo")), float(m.group("hi"))
        except ValueError:
            return None
        match (math.isinf(lo), math.isinf(hi)):
            case (False, False):
                return (lo + hi) / 2.0
            case (True, False):
                return hi
            case (False, True):
                return lo
            case _:
                return None

    def decode(self, code: str, z: float | None) -> float | None:
        """Recover the original measurement for ``code`` given its within-bin z-score.

        Examples:
            >>> dec = ValueDecoder({"LAB//220045//bpm//value_[90.0,96.0)": (92.4, 1.72)})
            >>> round(dec.decode("LAB//220045//bpm//value_[90.0,96.0)", 0.93), 2)
            94.0

            Without statistics it falls back to the bin midpoint:

            >>> ValueDecoder({}).decode("LAB//220045//bpm//value_[90.0,96.0)", 0.93)
            93.0

            A non-finite z-score is treated as "no residual recorded", so the bin's own mean
            is used rather than propagating a NaN into the prompt:

            >>> ValueDecoder({"C//value_[1.0,2.0)": (1.5, 0.2)}).decode("C//value_[1.0,2.0)", float("nan"))
            1.5
        """
        stats = self.stats.get(code)
        if stats is not None:
            mean, std = stats
            if z is None or not math.isfinite(z):
                return mean
            return z * std + mean
        return self.bin_midpoint(code)


# ── Pivot ────────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PanelVisit:
    """One visit column: the panel values measured at a single timestamp."""

    values: dict[str, float | None]  # feature name -> value (None = presence-only hit)
    days_before_end: float


def _parse_itemid(code: str) -> int | None:
    """Extract the MIMIC itemid from a MEDS code string, or ``None`` if it has none.

    Examples:
        >>> _parse_itemid("LAB//220045//bpm//value_[90.0,96.0)")
        220045
        >>> _parse_itemid("DIAGNOSIS//ICD//10//A419") is None
        True
    """
    m = _CODE_RE.match(code)
    return int(m.group("itemid")) if m else None


def _base_code(code: str) -> str:
    """Strip a code's ``//value_[a,b)`` bin suffix, leaving the feature-identifying part.

    Examples:
        >>> _base_code("LAB//220045//bpm//value_[90.0,96.0)")
        'LAB//220045//bpm'
        >>> _base_code("HR")
        'HR'
    """
    return _VALUE_BIN_RE.sub("", code)


def _source_keys(code: str) -> list[tuple[str, int | str]]:
    """Candidate panel-source lookup keys for a MEDS code, most specific first."""
    keys: list[tuple[str, int | str]] = [("code", _base_code(code))]
    itemid = _parse_itemid(code)
    if itemid is not None:
        keys.append(("itemid", itemid))
    return keys


def panel_visits_from_dense(
    dense: dict[str, np.ndarray],
    index_to_code: dict[int, str],
    panel: Panel,
    decoder: ValueDecoder,
) -> list[PanelVisit]:
    """Pivot a subject's dense event stream onto the panel's feature axis.

    One :class:`PanelVisit` per unique timestamp at which *at least one* panel feature was
    measured; timestamps carrying only off-panel codes are dropped entirely (they would be
    all-missing columns and cost tokens for nothing).  Where a timestamp holds several
    measurements of the same feature, the earliest-listed source in the panel spec wins, and
    within one source the last measurement at that timestamp wins.

    Examples:
        >>> panel = Panel("t", (
        ...     PanelFeature("Heart Rate", "u", "r", (PanelSource(220045),)),
        ...     PanelFeature("Temperature", "u", "r",
        ...                  (PanelSource(223762), PanelSource(223761, "f_to_c"))),
        ... ))
        >>> idx2code = {
        ...     3: "LAB//220045//bpm//value_[90.0,96.0)",
        ...     4: "LAB//223761//°F//value_[98.0,99.0)",
        ...     5: "DIAGNOSIS//ICD//10//A419",
        ... }
        >>> dense = {
        ...     "code": np.array([[3, 4], [5, 0]]),
        ...     "dim1/mask": np.array([[True, True], [True, False]]),
        ...     "numeric_value": np.array([[0.0, 0.0], [0.0, 0.0]]),
        ...     "time_delta_days": np.array([np.nan, 2.0]),
        ... }
        >>> visits = panel_visits_from_dense(dense, idx2code, panel, ValueDecoder({}))
        >>> len(visits)                                  # the diagnosis-only event is dropped
        1
        >>> round(visits[0].values["Heart Rate"], 1)
        93.0
        >>> round(visits[0].values["Temperature"], 1)    # 98.5 F -> C
        36.9
    """
    codes = np.atleast_2d(np.asarray(dense["code"]))
    n_events = codes.shape[0]
    if n_events == 0:
        return []

    mask = dense.get("dim1/mask")
    mask = np.ones_like(codes, dtype=bool) if mask is None else np.atleast_2d(np.asarray(mask, dtype=bool))
    numeric = dense.get("numeric_value")
    numeric = None if numeric is None else np.atleast_2d(np.asarray(numeric, dtype=float))

    deltas = dense.get("time_delta_days")
    deltas = np.zeros(n_events) if deltas is None else np.nan_to_num(np.asarray(deltas, dtype=float))
    elapsed = np.cumsum(deltas)
    total_elapsed = float(elapsed[-1]) if n_events else 0.0

    lookup = panel.source_index
    visits: list[PanelVisit] = []
    for i in range(n_events):
        values: dict[str, float | None] = {}
        chosen_priority: dict[str, int] = {}
        for j in range(codes.shape[1]):
            code_idx = int(codes[i, j])
            if not mask[i, j] or code_idx == 0:
                continue
            code = index_to_code.get(code_idx)
            if code is None:
                continue
            hit = next((lookup[k] for k in _source_keys(code) if k in lookup), None)
            if hit is None:
                continue
            feature, source = hit
            priority = feature.sources.index(source)
            if feature.name in chosen_priority and chosen_priority[feature.name] < priority:
                continue
            if feature.kind == "presence":
                values[feature.name] = None
            else:
                z = float(numeric[i, j]) if numeric is not None else None
                raw = decoder.decode(code, z)
                values[feature.name] = None if raw is None else source.convert(raw)
            chosen_priority[feature.name] = priority
        if values:
            visits.append(PanelVisit(values=values, days_before_end=total_elapsed - float(elapsed[i])))
    return visits


def apply_locf(visits: list[PanelVisit], panel: Panel) -> tuple[list[PanelVisit], set[tuple[int, str]]]:
    """Forward-fill each feature across visits; returns the filled visits and which cells were filled.

    Reproduces the reference implementation's default imputation (``impute=1``/``2``).  Only
    forward fill is applied — a feature never measured before a given visit stays missing,
    exactly as Last Observation Carried Forward implies.

    Examples:
        >>> panel = Panel("t", (PanelFeature("HR", "u", "r", (PanelSource(1),)),))
        >>> visits = [PanelVisit({"HR": 80.0}, 2.0), PanelVisit({"Other": 1.0}, 1.0)]
        >>> filled, marks = apply_locf(visits, panel)
        >>> filled[1].values["HR"], sorted(marks)
        (80.0, [(1, 'HR')])
    """
    last: dict[str, float | None] = {}
    have: set[str] = set()
    filled: list[PanelVisit] = []
    marks: set[tuple[int, str]] = set()
    for i, visit in enumerate(visits):
        values = dict(visit.values)
        for feature in panel.features:
            if feature.name in values:
                last[feature.name] = values[feature.name]
                have.add(feature.name)
            elif feature.name in have:
                values[feature.name] = last[feature.name]
                marks.add((i, feature.name))
        filled.append(PanelVisit(values=values, days_before_end=visit.days_before_end))
    return filled, marks


# ── Rendering ────────────────────────────────────────────────────────────────────────────


def format_value(value: float | None, precision: int) -> str:
    """Render one cell value.

    ``None`` marks a presence-only feature (no recoverable magnitude); a feature absent from
    the visit is not passed here at all — the caller emits :data:`MISSING_MARKER`.

    Examples:
        >>> format_value(93.0, 1)
        '93.0'
        >>> format_value(36.94444, 1)
        '36.9'
        >>> format_value(36.94444, 3)
        '36.944'
        >>> format_value(None, 1)
        'present'
    """
    if value is None:
        return PRESENCE_MARKER
    if not math.isfinite(value):
        return MISSING_MARKER
    return f"{value:.{precision}f}"


def render_detail(
    visits: list[PanelVisit],
    panel: Panel,
    *,
    form: str,
    precision: int,
    locf_marks: set[tuple[int, str]] | None = None,
) -> str:
    """Render the ``{DETAIL}`` block in one of the reference implementation's three forms.

    Every panel feature gets a row (or an entry per visit under ``batches``) whether or not it
    was ever measured — the fixed feature axis is the point of the format, and their published
    prompts carry all-``nan`` rows for unmeasured variables.

    Examples:
        >>> panel = Panel("t", (
        ...     PanelFeature("Heart Rate", "u", "r", (PanelSource(220045),)),
        ...     PanelFeature("pH", "u", "r", (PanelSource(223830),)),
        ... ))
        >>> visits = [PanelVisit({"Heart Rate": 94.0}, 1.0), PanelVisit({"Heart Rate": 88.0}, 0.0)]
        >>> print(render_detail(visits, panel, form="string", precision=1))
        - Heart Rate: "94.0, 88.0"
        - pH: "nan, nan"

        >>> print(render_detail(visits, panel, form="list", precision=1))
        - Heart Rate: [94.0, 88.0]
        - pH: [nan, nan]

        >>> print(render_detail(visits[:1], panel, form="batches", precision=1))
        Visit 1:
        - Heart Rate: 94.0
        - pH: nan
    """
    if not visits:
        return NO_VISITS_LINE
    marks = locf_marks or set()

    def cell(visit_idx: int, feature: PanelFeature) -> str:
        visit = visits[visit_idx]
        if feature.name not in visit.values:
            return MISSING_MARKER
        text = format_value(visit.values[feature.name], precision)
        return f"{text}{LOCF_MARKER}" if (visit_idx, feature.name) in marks else text

    lines: list[str] = []
    match form:
        case "string" | "list":
            open_, close = ('"', '"') if form == "string" else ("[", "]")
            for feature in panel.features:
                joined = ", ".join(cell(i, feature) for i in range(len(visits)))
                lines.append(f"- {feature.name}: {open_}{joined}{close}")
        case "batches":
            for i in range(len(visits)):
                lines.append(f"Visit {i + 1}:")
                lines.extend(f"- {f.name}: {cell(i, f)}" for f in panel.features)
                lines.append("")
            lines.pop()
        case _:
            raise ValueError(f"Unknown form {form!r} — expected one of 'string', 'list', 'batches'.")
    return "\n".join(lines)


def render_unit_range(panel: Panel, *, unit: bool, reference_range: bool) -> str:
    """Render the ``{UNIT_RANGE_CONTEXT}`` block, or an empty string when both flags are off.

    Examples:
        >>> panel = Panel("t", (PanelFeature("Heart Rate", "Unit: bpm.",
        ...                                  "Reference range: 60 - 100.", (PanelSource(1),)),))
        >>> print(render_unit_range(panel, unit=True, reference_range=True))
        - Heart Rate: Unit: bpm. Reference range: 60 - 100.
        >>> print(render_unit_range(panel, unit=True, reference_range=False))
        - Heart Rate: Unit: bpm.
        >>> render_unit_range(panel, unit=False, reference_range=False)
        ''
    """
    if not (unit or reference_range):
        return ""
    lines = []
    for feature in panel.features:
        parts = ([feature.unit] if unit else []) + ([feature.range] if reference_range else [])
        lines.append(f"- {feature.name}: {' '.join(p for p in parts if p)}")
    return "\n".join(lines)


def render_record_times(visits: list[PanelVisit], mode: str) -> str:
    """Render ``{RECORD_TIME_LIST}``.

    ``index`` reproduces their MIMIC prompts (``0, 1, 2, 3``); ``days_before`` renders each
    visit's distance from the prediction time, which carries the irregular spacing a MEDS
    stream has and their resampled ICU panels do not.

    Examples:
        >>> visits = [PanelVisit({}, 3.5), PanelVisit({}, 1.0), PanelVisit({}, 0.0)]
        >>> render_record_times(visits, "index")
        '0, 1, 2'
        >>> render_record_times(visits, "days_before")
        '3.5 days before prediction, 1 day before prediction, prediction time'
    """
    match mode:
        case "index":
            return ", ".join(str(i) for i in range(len(visits)))
        case "days_before":
            out = []
            for visit in visits:
                days = visit.days_before_end
                if days <= 0:
                    out.append("prediction time")
                else:
                    out.append(f"{days:g} day{'' if days == 1 else 's'} before prediction")
            return ", ".join(out)
        case _:
            raise ValueError(f"Unknown record_time_mode {mode!r} — expected 'index' or 'days_before'.")


# ── Demographics ─────────────────────────────────────────────────────────────────────────


def subject_demographics(
    dense: dict[str, np.ndarray],
    static_codes: list[int] | None,
    index_to_code: dict[int, str],
) -> tuple[str, str]:
    """Recover ``({SEX}, {AGE})`` for the prompt header.

    Sex comes from the ``GENDER//*`` static code; age from the elapsed time between the
    ``MEDS_BIRTH`` event and the end of the window.  Either may be unrecoverable — a subject
    without a birth event, or a run with ``static_inclusion_mode=omit`` — in which case the
    corresponding placeholder is rendered rather than a fabricated value.

    Examples:
        >>> idx2code = {7: "GENDER//M", 9: "MEDS_BIRTH", 3: "LAB//220045//bpm"}
        >>> dense = {
        ...     "code": np.array([[9], [3]]),
        ...     "dim1/mask": np.array([[True], [True]]),
        ...     "time_delta_days": np.array([np.nan, 365.25 * 52]),
        ... }
        >>> subject_demographics(dense, [7], idx2code)
        ('male', '52')

        Missing statics degrade to placeholders instead of raising:

        >>> subject_demographics({"code": np.zeros((0, 1))}, None, idx2code)
        ('patient of unrecorded sex', 'unknown')
    """
    sex = UNKNOWN_SEX
    for code_idx in static_codes or []:
        mapped = _SEX_BY_CODE.get(index_to_code.get(int(code_idx), ""))
        if mapped:
            sex = mapped
            break

    codes = np.atleast_2d(np.asarray(dense.get("code", np.zeros((0, 1)))))
    if codes.shape[0] == 0:
        return sex, UNKNOWN_AGE
    mask = dense.get("dim1/mask")
    mask = np.ones_like(codes, dtype=bool) if mask is None else np.atleast_2d(np.asarray(mask, dtype=bool))
    deltas = dense.get("time_delta_days")
    if deltas is None:
        return sex, UNKNOWN_AGE
    elapsed = np.cumsum(np.nan_to_num(np.asarray(deltas, dtype=float)))

    for i in range(codes.shape[0]):
        for j in range(codes.shape[1]):
            if not mask[i, j] or int(codes[i, j]) == 0:
                continue
            if index_to_code.get(int(codes[i, j])) == _BIRTH_CODE:
                years = (float(elapsed[-1]) - float(elapsed[i])) / _DAYS_PER_YEAR
                return sex, (f"{years:.0f}" if years >= 0 else UNKNOWN_AGE)
    return sex, UNKNOWN_AGE


# ── Assembly ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class L4HConfig:
    """Knobs mirroring the reference implementation's ``config`` dict.

    ``form``, ``unit``, ``reference_range`` and ``n_shot`` are theirs directly.  ``impute``
    maps onto their ``0 | 1 | 2``: ``none`` leaves gaps as ``nan`` and appends their
    missing-value note, ``locf`` forward-fills silently, ``locf_marked`` forward-fills and
    appends their ``**``-marker instruction.  ``max_visits`` has no counterpart — their cohorts
    are pre-truncated to a handful of visits, whereas a MEDS window here reaches ~370.
    """

    form: str = "string"
    impute: str = "locf"
    unit: bool = True
    reference_range: bool = True
    max_visits: int = 48
    value_precision: int = 1
    record_time_mode: str = "index"
    response_format: str = "float"
    examples: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.form not in INPUT_FORMAT_DESCRIPTION:
            raise ValueError(f"form must be one of {sorted(INPUT_FORMAT_DESCRIPTION)}, got {self.form!r}")
        if self.impute not in {"none", "locf", "locf_marked"}:
            raise ValueError(f"impute must be 'none', 'locf' or 'locf_marked', got {self.impute!r}")
        if self.response_format not in {"float", "yes_no"}:
            raise ValueError(f"response_format must be 'float' or 'yes_no', got {self.response_format!r}")
        if self.max_visits < 1:
            raise ValueError(f"max_visits must be >= 1, got {self.max_visits}")


def render_examples(examples: tuple[str, ...]) -> str:
    """Render the ``{EXAMPLE}`` block using their numbering and preamble.

    Examples:
        >>> print(render_examples(("A", "B")))
        Here are 2 examples of input information:
        Example #1:A
        Example #2:B
        >>> render_examples(())
        ''
    """
    if not examples:
        return ""
    if len(examples) == 1:
        return f"Here is an example of input information:\nExample #1:{examples[0]}\n"
    header = f"Here are {len(examples)} examples of input information:\n"
    return header + "\n".join(f"Example #{i + 1}:{e}" for i, e in enumerate(examples)) + "\n"


def serialize_history(
    dense: dict[str, np.ndarray],
    static_codes: list[int] | None,
    index_to_code: dict[int, str],
    panel: Panel,
    decoder: ValueDecoder,
    cfg: L4HConfig,
) -> SerializedHistory:
    """Render the patient-specific portion of the prompt (header + ``{DETAIL}``).

    Everything here is query-independent, so ``__main__`` computes it once per
    ``(subject_id, prediction_time)`` group and shares it across that group's queries.  The
    returned :class:`SerializedHistory` counts *visits* rather than raw events, so the
    truncation statistics logged per run refer to the panel's visit axis.
    """
    visits = panel_visits_from_dense(dense, index_to_code, panel, decoder)
    n_total = len(visits)
    n_dropped = max(0, n_total - cfg.max_visits)
    visits = visits[n_dropped:]

    marks: set[tuple[int, str]] = set()
    if cfg.impute in {"locf", "locf_marked"}:
        visits, marks = apply_locf(visits, panel)
    if cfg.impute != "locf_marked":
        marks = set()

    sex, age = subject_demographics(dense, static_codes, index_to_code)
    text = PATIENT_BLOCK.format(
        SEX=sex,
        AGE=age,
        LENGTH=len(visits),
        RECORD_TIME_LIST=render_record_times(visits, cfg.record_time_mode),
        DETAIL=render_detail(visits, panel, form=cfg.form, precision=cfg.value_precision, locf_marks=marks),
    )
    return SerializedHistory(text=text, n_events_total=n_total, n_events_dropped=n_dropped)


def build_user_prompt(
    history_text: str,
    query_code: str,
    duration_days: float,
    panel: Panel,
    cfg: L4HConfig,
) -> str:
    """Assemble the full user turn from the shared patient block and this row's query.

    Note the query appears in the *task description*, near the top of the prompt, because that
    is where the reference template puts it.  That placement means two queries over the same
    patient no longer share a token prefix — see the module docstring in ``__main__`` for the
    throughput consequence.
    """
    input_format = INPUT_FORMAT_DESCRIPTION[cfg.form]
    if cfg.impute == "none":
        input_format += MISSING_VALUE_DESCRIPTION
    elif cfg.impute == "locf_marked":
        input_format += " " + INSTRUCTING_MISSING_VALUE

    task_template = TASK_DESCRIPTION if cfg.response_format == "float" else TASK_DESCRIPTION_YES_NO
    response_format = RESPONSE_FORMAT if cfg.response_format == "float" else RESPONSE_FORMAT_YES_NO

    return USER_PROMPT.format(
        INPUT_FORMAT_DESCRIPTION=input_format,
        TASK_DESCRIPTION_AND_RESPONSE_FORMAT=task_template.format(code=query_code, days=f"{duration_days:g}"),
        UNIT_RANGE_CONTEXT=render_unit_range(panel, unit=cfg.unit, reference_range=cfg.reference_range),
        EXAMPLE=render_examples(cfg.examples),
        PATIENT_BLOCK=history_text,
        RESPONSE_FORMAT=response_format,
    )


def prompt_template_hash(panel: Panel, cfg: L4HConfig) -> str:
    """Short stable fingerprint over the template constants, the panel spec and the knobs.

    Any edit to a template string, a panel unit/itemid, or a rendering knob changes the hash,
    so ``resume`` cannot extend shards written under a different prompt.
    """
    parts = [
        PROMPT_TEMPLATE_VERSION,
        SYSTEM_PROMPT,
        USER_PROMPT,
        json.dumps(INPUT_FORMAT_DESCRIPTION, sort_keys=True),
        MISSING_VALUE_DESCRIPTION,
        INSTRUCTING_MISSING_VALUE,
        TASK_DESCRIPTION,
        RESPONSE_FORMAT,
        TASK_DESCRIPTION_YES_NO,
        RESPONSE_FORMAT_YES_NO,
        panel.spec_digest(),
        json.dumps(
            {
                "form": cfg.form,
                "impute": cfg.impute,
                "unit": cfg.unit,
                "reference_range": cfg.reference_range,
                "max_visits": cfg.max_visits,
                "value_precision": cfg.value_precision,
                "record_time_mode": cfg.record_time_mode,
                "response_format": cfg.response_format,
                "n_shot": len(cfg.examples),
            },
            sort_keys=True,
        ),
    ]
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]
