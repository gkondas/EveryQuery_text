"""Serialization of patient event sequences into LLM prompts.

Converts the tensorized event stream a :class:`~every_query.data.dataset.EveryQueryPytorchDataset`
sample is built from (vocab-index codes, inter-event ``time_delta_days``, optional
``numeric_value``) into readable text, plus the per-row question the LLM answers.

Conventions:

- ``time_delta_days`` is a **delta between consecutive events**, not an absolute timestamp.
  :func:`events_from_jnrt_dense` cumulative-sums it and renders each event as days *before
  the prediction time* (negative, e.g. ``[-45d]``), so the "within the next d days" question
  is naturally anchored at 0.  The gap between the last observed event and the actual
  prediction time is passed in as ``gap_days`` (from MTD's ``window_last_observed`` schema
  column) so the anchor is the prediction time itself, not the last event.
- ``numeric_value`` is included where present and omitted entirely where absent — no
  ``None`` / ``nan`` ever appears in the text.
- Histories longer than ``max_events`` are truncated from the left (most recent events
  kept), with a marker line noting how many earlier events were dropped.
- Codes are rendered raw (``ICD//E11.9``) or, when a ``code_descriptions`` mapping is
  provided, annotated (``ICD//E11.9 (Type 2 diabetes mellitus without complications)``).

The prompt template lives in this module only, is versioned via
:data:`PROMPT_TEMPLATE_VERSION`, and is fingerprinted by :func:`prompt_template_hash` —
recorded in the output parquet metadata so a run is reconstructable from its output alone.
"""

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ── Prompt template (versioned; every constant below feeds prompt_template_hash) ────────

PROMPT_TEMPLATE_VERSION = "1"

SYSTEM_PROMPT = (
    "You are a clinical prediction assistant. You are given a patient's medical event history "
    "as a list of timestamped codes, where times are in days relative to the prediction time "
    "(day 0). Answer the question about a future clinical event based only on the provided "
    "history."
)

HISTORY_HEADER = "Patient history (most recent last):"
TRUNCATION_LINE = "[{n} earlier {noun} omitted]"
EMPTY_HISTORY_LINE = "- (no prior events recorded)"
EVENT_LINE = "- [{days}d] {code}"
EVENT_LINE_WITH_VALUE = "- [{days}d] {code}: {value}"
QUESTION_LINE = "Question: Will code {code} occur within {days} days?"
LOGPROB_INSTRUCTION = "Answer with exactly one word: Yes or No."
GUIDED_INSTRUCTION = "Answer with a single number: the probability (between 0 and 1) that this event occurs."


def prompt_template_hash() -> str:
    """Return a short stable fingerprint of the prompt template.

    Covers every template constant plus :data:`PROMPT_TEMPLATE_VERSION`, so any edit to the
    prompt's wording or structure changes the hash.  Recorded in output parquet metadata.

    Examples:
        The hash is deterministic and 16 hex chars:

        >>> h = prompt_template_hash()
        >>> h == prompt_template_hash() and len(h) == 16
        True
        >>> int(h, 16) >= 0
        True
    """
    parts = [
        PROMPT_TEMPLATE_VERSION,
        SYSTEM_PROMPT,
        HISTORY_HEADER,
        TRUNCATION_LINE,
        EMPTY_HISTORY_LINE,
        EVENT_LINE,
        EVENT_LINE_WITH_VALUE,
        QUESTION_LINE,
        LOGPROB_INSTRUCTION,
        GUIDED_INSTRUCTION,
    ]
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


# ── Event extraction ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Measurement:
    """A single measurement within an event: a code string plus an optional numeric value."""

    code: str
    numeric_value: float | None = None


@dataclass(frozen=True)
class Event:
    """One event (unique timestamp): its time relative to prediction plus its measurements.

    ``days_before_prediction`` is ≤ 0 by construction — 0 means the event coincides with the prediction time.
    """

    days_before_prediction: float
    measurements: tuple[Measurement, ...]


def events_from_jnrt_dense(
    dense: dict[str, np.ndarray],
    index_to_code: dict[int, str],
    gap_days: float = 0.0,
) -> list[Event]:
    """Convert a single subject's densified nested-ragged event data into :class:`Event` s.

    ``dense`` is the ``to_dense()`` output of the ``JointNestedRaggedTensorDict`` returned by
    ``MEDSPytorchDataset.load_subject_data`` — ``time_delta_days`` is ``[n_events]``, ``code``
    / ``numeric_value`` are ``[n_events, max_measurements]`` with a ``dim1/mask`` validity
    mask.  ``time_delta_days`` deltas are cumulative-summed and re-anchored so the *last*
    event sits at ``-gap_days`` (the gap between the last observed event and the prediction
    time); all times come out ≤ 0, i.e. days before prediction time.

    Pad entries (code index 0) and masked-out slots are dropped.  Codes missing from
    ``index_to_code`` render as ``UNKNOWN_CODE_{idx}`` rather than raising — a vocab-mapping
    hole shouldn't kill a multi-hour run, and the marker is greppable in serialized prompts.

    Examples:
        >>> dense = {
        ...     "time_delta_days": np.array([np.nan, 12.0, 33.0]),
        ...     "code": np.array([[3, 4], [5, 0], [6, 0]]),
        ...     "numeric_value": np.array([[np.nan, 7.2], [np.nan, np.nan], [1.5, np.nan]]),
        ...     "dim1/mask": np.array([[True, True], [True, False], [True, False]]),
        ... }
        >>> idx2code = {3: "ICD//E11.9", 4: "LAB//HbA1c", 5: "ICD//I10", 6: "LAB//CR"}
        >>> for e in events_from_jnrt_dense(dense, idx2code):
        ...     print(round(e.days_before_prediction, 1), [(m.code, m.numeric_value) for m in e.measurements])
        -45.0 [('ICD//E11.9', None), ('LAB//HbA1c', 7.2)]
        -33.0 [('ICD//I10', None)]
        0.0 [('LAB//CR', 1.5)]

        ``gap_days`` shifts every event back by the last-event → prediction-time gap:

        >>> events_from_jnrt_dense(dense, idx2code, gap_days=5.0)[-1].days_before_prediction
        -5.0

        Empty input yields no events:

        >>> events_from_jnrt_dense(
        ...     {"time_delta_days": np.array([]), "code": np.zeros((0, 1)),
        ...      "numeric_value": np.zeros((0, 1)), "dim1/mask": np.zeros((0, 1), dtype=bool)},
        ...     idx2code,
        ... )
        []
    """
    time_deltas = np.nan_to_num(np.asarray(dense["time_delta_days"], dtype=float).reshape(-1), nan=0.0)
    n_events = len(time_deltas)
    if n_events == 0:
        return []

    codes = np.atleast_2d(np.asarray(dense["code"]))
    numeric_values = np.atleast_2d(np.asarray(dense["numeric_value"], dtype=float))
    mask = dense.get("dim1/mask")
    mask = np.ones_like(codes, dtype=bool) if mask is None else np.atleast_2d(np.asarray(mask, dtype=bool))

    cum_days = np.cumsum(time_deltas)
    total_days = cum_days[-1]

    events: list[Event] = []
    for i in range(n_events):
        measurements = []
        for j in range(codes.shape[1]):
            code_idx = int(codes[i, j])
            if not mask[i, j] or code_idx == 0:  # 0 is MTD's PAD index
                continue
            value = float(numeric_values[i, j])
            measurements.append(
                Measurement(
                    code=index_to_code.get(code_idx, f"UNKNOWN_CODE_{code_idx}"),
                    numeric_value=None if math.isnan(value) else value,
                )
            )
        if measurements:
            events.append(Event(float(cum_days[i] - total_days - gap_days), tuple(measurements)))
    return events


# ── Rendering ────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SerializedHistory:
    """A rendered history string plus the truncation accounting for run-level stats."""

    text: str
    n_events_total: int
    n_events_dropped: int

    @property
    def truncated(self) -> bool:
        """Whether any events were dropped from the left.

        Examples:
            >>> SerializedHistory("t", n_events_total=5, n_events_dropped=2).truncated
            True
            >>> SerializedHistory("t", n_events_total=5, n_events_dropped=0).truncated
            False
        """
        return self.n_events_dropped > 0


def _render_code(code: str, code_descriptions: dict[str, str] | None) -> str:
    """Render a code, appending its human-readable description when one is available.

    Examples:
        >>> _render_code("ICD//E11.9", {"ICD//E11.9": "Type 2 diabetes mellitus without complications"})
        'ICD//E11.9 (Type 2 diabetes mellitus without complications)'
        >>> _render_code("ICD//E11.9", None)
        'ICD//E11.9'
        >>> _render_code("LAB//HbA1c", {"ICD//E11.9": "..."})
        'LAB//HbA1c'
    """
    if code_descriptions:
        description = code_descriptions.get(code)
        if description:
            return f"{code} ({description})"
    return code


def serialize_history(
    events: list[Event],
    *,
    max_events: int,
    code_descriptions: dict[str, str] | None = None,
) -> SerializedHistory:
    """Render a patient's events as the history block of the prompt.

    Keeps the most recent ``max_events`` events (truncating from the left) and, when
    truncation occurs, prepends a marker line noting how many earlier events were dropped.
    Each measurement gets its own line, tagged with its event's (integer-rounded) days
    before prediction time; ``numeric_value`` is appended after a colon only where present.

    Examples:
        >>> events = [
        ...     Event(-45.0, (Measurement("ICD//E11.9"), Measurement("LAB//HbA1c", 7.2))),
        ...     Event(-33.0, (Measurement("ICD//I10"),)),
        ...     Event(0.0, (Measurement("LAB//CR", 1.5),)),
        ... ]
        >>> print(serialize_history(events, max_events=10).text)
        Patient history (most recent last):
        - [-45d] ICD//E11.9
        - [-45d] LAB//HbA1c: 7.2
        - [-33d] ICD//I10
        - [0d] LAB//CR: 1.5

        Truncation keeps the most recent events and prepends a marker:

        >>> h = serialize_history(events, max_events=2)
        >>> print(h.text)
        Patient history (most recent last):
        [1 earlier event omitted]
        - [-33d] ICD//I10
        - [0d] LAB//CR: 1.5
        >>> h.n_events_total, h.n_events_dropped, h.truncated
        (3, 1, True)

        Descriptions annotate codes when provided:

        >>> print(serialize_history(events[1:2], max_events=10,
        ...     code_descriptions={"ICD//I10": "Essential hypertension"}).text)
        Patient history (most recent last):
        - [-33d] ICD//I10 (Essential hypertension)

        An empty history renders a placeholder rather than an empty block:

        >>> print(serialize_history([], max_events=10).text)
        Patient history (most recent last):
        - (no prior events recorded)
    """
    n_total = len(events)
    n_dropped = max(0, n_total - max_events)
    kept = events[n_dropped:]

    lines = [HISTORY_HEADER]
    if n_dropped:
        lines.append(TRUNCATION_LINE.format(n=n_dropped, noun="event" if n_dropped == 1 else "events"))
    for event in kept:
        days = f"{round(event.days_before_prediction)}"
        for m in event.measurements:
            code = _render_code(m.code, code_descriptions)
            if m.numeric_value is None:
                lines.append(EVENT_LINE.format(days=days, code=code))
            else:
                lines.append(EVENT_LINE_WITH_VALUE.format(days=days, code=code, value=f"{m.numeric_value:g}"))
    if not kept:
        lines.append(EMPTY_HISTORY_LINE)

    return SerializedHistory(text="\n".join(lines), n_events_total=n_total, n_events_dropped=n_dropped)


def serialize_question(
    query_code: str,
    duration_days: float,
    code_descriptions: dict[str, str] | None = None,
) -> str:
    """Render the per-row question the LLM answers.

    Examples:
        >>> serialize_question("ICD//I50.9", 90.0, {"ICD//I50.9": "Heart failure, unspecified"})
        'Question: Will code ICD//I50.9 (Heart failure, unspecified) occur within 90 days?'
        >>> serialize_question("ICD//I50.9", 30.5)
        'Question: Will code ICD//I50.9 occur within 30.5 days?'
    """
    return QUESTION_LINE.format(code=_render_code(query_code, code_descriptions), days=f"{duration_days:g}")


def build_user_prompt(history_text: str, question: str, method: str) -> str:
    """Assemble the full user-turn prompt for the given probability-extraction ``method``.

    ``logprob`` appends the one-word Yes/No instruction; ``guided`` appends the numeric
    instruction.  The system turn is :data:`SYSTEM_PROMPT`, sent separately by the caller.

    Examples:
        >>> print(build_user_prompt("Patient history (most recent last):\\n- [0d] HR: 88",
        ...                         "Question: Will code TEMP occur within 30 days?", "logprob"))
        Patient history (most recent last):
        - [0d] HR: 88
        <BLANKLINE>
        Question: Will code TEMP occur within 30 days?
        Answer with exactly one word: Yes or No.

        >>> build_user_prompt("h", "q", "guided").endswith(GUIDED_INSTRUCTION)
        True

        >>> build_user_prompt("h", "q", "freeform")
        Traceback (most recent call last):
            ...
        ValueError: Unknown probability-extraction method 'freeform'; expected 'logprob' or 'guided'.
    """
    match method:
        case "logprob":
            instruction = LOGPROB_INSTRUCTION
        case "guided":
            instruction = GUIDED_INSTRUCTION
        case _:
            raise ValueError(
                f"Unknown probability-extraction method {method!r}; expected 'logprob' or 'guided'."
            )
    return f"{history_text}\n\n{question}\n{instruction}"


# ── Code descriptions ────────────────────────────────────────────────────────────────────


def load_code_descriptions(path: Path) -> dict[str, str]:
    """Load a ``code → human-readable description`` mapping from ``path``.

    Supports two formats: a ``.json`` file holding a flat string-to-string object, or a
    ``.parquet`` file with ``code`` and ``description`` columns (e.g. a MEDS
    ``metadata/codes.parquet``).  Entries with a null / empty description are dropped so
    ``_render_code`` never emits ``CODE ()``.

    Examples:
        >>> import tempfile, json
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     p = Path(tmp) / "desc.json"
        ...     _ = p.write_text(json.dumps({"ICD//I10": "Essential hypertension", "X": ""}))
        ...     load_code_descriptions(p)
        {'ICD//I10': 'Essential hypertension'}

        >>> import polars as pl
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     p = Path(tmp) / "codes.parquet"
        ...     pl.DataFrame({"code": ["HR", "TEMP"], "description": ["Heart rate", None]}).write_parquet(p)
        ...     load_code_descriptions(p)
        {'HR': 'Heart rate'}

        >>> load_code_descriptions(Path("descriptions.csv"))
        Traceback (most recent call last):
            ...
        ValueError: Unsupported code_descriptions format '.csv'...
    """
    match path.suffix:
        case ".json":
            raw = json.loads(Path(path).read_text())
            return {str(k): str(v) for k, v in raw.items() if v}
        case ".parquet":
            import polars as pl

            df = pl.read_parquet(path, columns=["code", "description"])
            return {
                code: description
                for code, description in zip(df["code"], df["description"], strict=True)
                if description
            }
        case _:
            raise ValueError(
                f"Unsupported code_descriptions format {path.suffix!r} — provide a .json mapping "
                f"or a .parquet with 'code' and 'description' columns."
            )
