"""Golden-string tests for ``every_query.llm_baseline.serialize``.

Covers the fixture event sequence end-to-end (JNRT-dense arrays → prompt text), the
truncation case, the missing-``numeric_value`` case, and the with/without-descriptions
switch.  No MTD dataset or tensorized cohort needed — the serializer's inputs are plain
numpy arrays by design.
"""

import numpy as np
import pytest

from every_query.llm_baseline.serialize import (
    Event,
    Measurement,
    build_user_prompt,
    events_from_jnrt_dense,
    load_code_descriptions,
    prompt_template_hash,
    serialize_history,
    serialize_question,
)

_INDEX_TO_CODE = {
    3: "ICD//E11.9",
    4: "LAB//HbA1c",
    5: "ICD//I10",
    6: "ICD//I50.9",
}

_DESCRIPTIONS = {
    "ICD//E11.9": "Type 2 diabetes mellitus without complications",
    "ICD//I10": "Essential hypertension",
    "ICD//I50.9": "Heart failure, unspecified",
}


@pytest.fixture
def fixture_dense() -> dict[str, np.ndarray]:
    """Three events: (diabetes dx + HbA1c lab), (hypertension dx), (HbA1c lab), 120/108/75d ago."""
    return {
        # Deltas between consecutive events; the leading NaN mirrors MTD's first-event delta.
        "time_delta_days": np.array([np.nan, 12.0, 33.0]),
        "code": np.array([[3, 4], [5, 0], [4, 0]]),
        "numeric_value": np.array([[np.nan, 7.2], [np.nan, np.nan], [7.9, np.nan]]),
        "dim1/mask": np.array([[True, True], [True, False], [True, False]]),
    }


def test_events_from_jnrt_dense_times_and_values(fixture_dense):
    # gap_days=75 → the last event sits 75 days before prediction time.
    events = events_from_jnrt_dense(fixture_dense, _INDEX_TO_CODE, gap_days=75.0)
    assert [round(e.days_before_prediction) for e in events] == [-120, -108, -75]
    assert [m.code for e in events for m in e.measurements] == [
        "ICD//E11.9",
        "LAB//HbA1c",
        "ICD//I10",
        "LAB//HbA1c",
    ]
    # numeric_value is None exactly where the input had NaN.
    assert [m.numeric_value for e in events for m in e.measurements] == [None, 7.2, None, 7.9]


def test_golden_history_without_descriptions(fixture_dense):
    events = events_from_jnrt_dense(fixture_dense, _INDEX_TO_CODE, gap_days=75.0)
    history = serialize_history(events, max_events=10)
    assert history.text == (
        "Patient history (most recent last):\n"
        "- [-120d] ICD//E11.9\n"
        "- [-120d] LAB//HbA1c: 7.2\n"
        "- [-108d] ICD//I10\n"
        "- [-75d] LAB//HbA1c: 7.9"
    )
    assert not history.truncated
    assert history.n_events_total == 3
    # No None / nan ever appears in the text.
    assert "None" not in history.text and "nan" not in history.text


def test_golden_history_with_descriptions(fixture_dense):
    events = events_from_jnrt_dense(fixture_dense, _INDEX_TO_CODE, gap_days=75.0)
    history = serialize_history(events, max_events=10, code_descriptions=_DESCRIPTIONS)
    assert history.text == (
        "Patient history (most recent last):\n"
        "- [-120d] ICD//E11.9 (Type 2 diabetes mellitus without complications)\n"
        "- [-120d] LAB//HbA1c: 7.2\n"
        "- [-108d] ICD//I10 (Essential hypertension)\n"
        "- [-75d] LAB//HbA1c: 7.9"
    )


def test_golden_history_truncation(fixture_dense):
    events = events_from_jnrt_dense(fixture_dense, _INDEX_TO_CODE, gap_days=75.0)
    history = serialize_history(events, max_events=1)
    assert history.text == (
        "Patient history (most recent last):\n[2 earlier events omitted]\n- [-75d] LAB//HbA1c: 7.9"
    )
    assert history.truncated
    assert (history.n_events_total, history.n_events_dropped) == (3, 2)

    singular = serialize_history(events, max_events=2)
    assert "[1 earlier event omitted]" in singular.text


def test_empty_history_placeholder():
    history = serialize_history([], max_events=10)
    assert history.text == "Patient history (most recent last):\n- (no prior events recorded)"
    assert not history.truncated


def test_full_prompt_golden(fixture_dense):
    events = events_from_jnrt_dense(fixture_dense, _INDEX_TO_CODE, gap_days=75.0)
    history = serialize_history(events, max_events=10, code_descriptions=_DESCRIPTIONS)
    question = serialize_question("ICD//I50.9", 90.0, _DESCRIPTIONS)
    prompt = build_user_prompt(history.text, question, "logprob")
    assert prompt == (
        "Patient history (most recent last):\n"
        "- [-120d] ICD//E11.9 (Type 2 diabetes mellitus without complications)\n"
        "- [-120d] LAB//HbA1c: 7.2\n"
        "- [-108d] ICD//I10 (Essential hypertension)\n"
        "- [-75d] LAB//HbA1c: 7.9\n"
        "\n"
        "Question: Will code ICD//I50.9 (Heart failure, unspecified) occur within 90 days?\n"
        "Answer with exactly one word: Yes or No."
    )


def test_unknown_code_index_renders_marker(fixture_dense):
    events = events_from_jnrt_dense(fixture_dense, {}, gap_days=0.0)
    assert events[0].measurements[0].code == "UNKNOWN_CODE_3"


def test_prompt_template_hash_is_stable_and_short():
    assert prompt_template_hash() == prompt_template_hash()
    assert len(prompt_template_hash()) == 16


def test_events_are_immutable():
    event = Event(-1.0, (Measurement("HR", 88.0),))
    with pytest.raises(AttributeError):
        event.days_before_prediction = 0.0


def test_load_code_descriptions_rejects_unknown_suffix(tmp_path):
    fp = tmp_path / "descriptions.txt"
    fp.write_text("HR\tHeart rate")
    with pytest.raises(ValueError, match="Unsupported code_descriptions format"):
        load_code_descriptions(fp)
