"""Golden-string tests for ``every_query.llm_baseline.serialize_l4h``.

Covers the pivot from JNRT-dense arrays onto the fixed feature panel, the within-bin z-score
inversion that recovers original measurements, the three ``{DETAIL}`` forms, LOCF imputation,
visit truncation, demographics recovery, and the assembled prompt.  Inputs are plain numpy
arrays, so no MTD dataset or tensorized cohort is needed.

One test guards fidelity directly: the rendering template must be a pure refactor of the
reference implementation's ``USERPROMPT``, differing only in that the four patient-header
lines and ``{DETAIL}`` are collapsed into a single ``{PATIENT_BLOCK}`` slot.
"""

import numpy as np
import pytest

from every_query.llm_baseline.prompt_style import _resolve_panel_path
from every_query.llm_baseline.serialize_l4h import (
    PATIENT_BLOCK,
    REFERENCE_USER_PROMPT,
    USER_PROMPT,
    L4HConfig,
    Panel,
    PanelFeature,
    PanelSource,
    PanelVisit,
    ValueDecoder,
    apply_locf,
    build_user_prompt,
    load_panel,
    panel_visits_from_dense,
    prompt_template_hash,
    render_detail,
    render_examples,
    render_record_times,
    render_unit_range,
    serialize_history,
    subject_demographics,
)

# Heart Rate + Temperature (Fahrenheit source, converted) + a presence-only feature.
_PANEL = Panel(
    "test",
    (
        PanelFeature("Heart Rate", "Unit: bpm.", "Reference range: 60 - 100.", (PanelSource(220045),)),
        PanelFeature(
            "Temperature",
            "Unit: degrees Celsius.",
            "Reference range: 36.1 - 37.2.",
            (PanelSource(223762), PanelSource(223761, "f_to_c")),
        ),
        PanelFeature(
            "Capillary refill rate", "Unit: /.", "Reference range: /.", (PanelSource(224308),), "presence"
        ),
    ),
)

_HR_BIN = "LAB//220045//bpm//value_[90.0,96.0)"
_TEMP_F_BIN = "LAB//223761//°F//value_[98.0,99.0)"

_INDEX_TO_CODE = {
    3: _HR_BIN,
    4: _TEMP_F_BIN,
    5: "DIAGNOSIS//ICD//10//A419",  # off-panel: must be ignored entirely
    6: "LAB//224308//UNK",  # presence-only
    7: "GENDER//F",
    8: "MEDS_BIRTH",
}

# Per-bin mean/std as they appear in metadata/codes.parquet for this cohort.
_DECODER = ValueDecoder({_HR_BIN: (92.4, 1.72), _TEMP_F_BIN: (98.5, 0.3)})


@pytest.fixture
def fixture_dense() -> dict[str, np.ndarray]:
    """Three timestamps: (HR + temp), (off-panel diagnosis only), (HR)."""
    return {
        "code": np.array([[3, 4], [5, 0], [3, 0]]),
        "dim1/mask": np.array([[True, True], [True, False], [True, False]]),
        # Within-bin z-scores: +0.93 sd, 0.0 sd, -1.0 sd.
        "numeric_value": np.array([[0.93, 0.0], [0.0, 0.0], [-1.0, 0.0]]),
        "time_delta_days": np.array([np.nan, 1.0, 2.0]),
    }


# ── Pivot and value recovery ─────────────────────────────────────────────────────────────


def test_pivot_drops_off_panel_events(fixture_dense):
    visits = panel_visits_from_dense(fixture_dense, _INDEX_TO_CODE, _PANEL, _DECODER)
    # The diagnosis-only timestamp contributes no panel measurement, so it is not a visit.
    assert len(visits) == 2


def test_value_recovery_inverts_the_within_bin_zscore(fixture_dense):
    visits = panel_visits_from_dense(fixture_dense, _INDEX_TO_CODE, _PANEL, _DECODER)
    # 0.93 sd above a bin mean of 92.4 with sd 1.72 -> the original 94.0 bpm reading.
    assert visits[0].values["Heart Rate"] == pytest.approx(94.0, abs=0.01)
    # -1.0 sd below the same bin mean.
    assert visits[1].values["Heart Rate"] == pytest.approx(90.68, abs=0.01)


def test_source_transform_applied_to_recovered_value(fixture_dense):
    visits = panel_visits_from_dense(fixture_dense, _INDEX_TO_CODE, _PANEL, _DECODER)
    # 98.5 F at z=0 -> 36.94 C, and the panel's unit is Celsius.
    assert visits[0].values["Temperature"] == pytest.approx(36.94, abs=0.01)


def test_decoder_falls_back_to_bin_midpoint_without_statistics(fixture_dense):
    visits = panel_visits_from_dense(fixture_dense, _INDEX_TO_CODE, _PANEL, ValueDecoder({}))
    assert visits[0].values["Heart Rate"] == pytest.approx(93.0)  # midpoint of [90.0, 96.0)


def test_presence_only_feature_has_no_magnitude():
    dense = {
        "code": np.array([[6]]),
        "dim1/mask": np.array([[True]]),
        "numeric_value": np.array([[0.0]]),
        "time_delta_days": np.array([0.0]),
    }
    visits = panel_visits_from_dense(dense, _INDEX_TO_CODE, _PANEL, _DECODER)
    assert visits[0].values["Capillary refill rate"] is None


def test_higher_priority_source_wins_within_a_visit():
    # Both Temperature sources at one timestamp: the Celsius source is listed first.
    idx2code = {1: "LAB//223762//°C//value_[36.0,37.0)", 2: _TEMP_F_BIN}
    dense = {
        "code": np.array([[2, 1]]),  # Fahrenheit listed first in the stream
        "dim1/mask": np.array([[True, True]]),
        "numeric_value": np.array([[0.0, 0.0]]),
        "time_delta_days": np.array([0.0]),
    }
    decoder = ValueDecoder({"LAB//223762//°C//value_[36.0,37.0)": (36.5, 0.1), _TEMP_F_BIN: (98.5, 0.3)})
    visits = panel_visits_from_dense(dense, idx2code, _PANEL, decoder)
    assert visits[0].values["Temperature"] == pytest.approx(36.5)


def test_empty_stream_yields_no_visits():
    dense = {"code": np.zeros((0, 1)), "dim1/mask": np.zeros((0, 1), dtype=bool)}
    assert panel_visits_from_dense(dense, _INDEX_TO_CODE, _PANEL, _DECODER) == []


# ── Rendering ────────────────────────────────────────────────────────────────────────────


def test_golden_detail_string_form(fixture_dense):
    visits = panel_visits_from_dense(fixture_dense, _INDEX_TO_CODE, _PANEL, _DECODER)
    assert render_detail(visits, _PANEL, form="string", precision=1) == (
        '- Heart Rate: "94.0, 90.7"\n- Temperature: "36.9, nan"\n- Capillary refill rate: "nan, nan"'
    )


def test_golden_detail_list_form(fixture_dense):
    visits = panel_visits_from_dense(fixture_dense, _INDEX_TO_CODE, _PANEL, _DECODER)
    assert render_detail(visits, _PANEL, form="list", precision=1) == (
        "- Heart Rate: [94.0, 90.7]\n- Temperature: [36.9, nan]\n- Capillary refill rate: [nan, nan]"
    )


def test_golden_detail_batches_form(fixture_dense):
    visits = panel_visits_from_dense(fixture_dense, _INDEX_TO_CODE, _PANEL, _DECODER)
    assert render_detail(visits, _PANEL, form="batches", precision=1) == (
        "Visit 1:\n"
        "- Heart Rate: 94.0\n"
        "- Temperature: 36.9\n"
        "- Capillary refill rate: nan\n"
        "\n"
        "Visit 2:\n"
        "- Heart Rate: 90.7\n"
        "- Temperature: nan\n"
        "- Capillary refill rate: nan"
    )


def test_every_panel_feature_gets_a_row_even_when_never_measured(fixture_dense):
    visits = panel_visits_from_dense(fixture_dense, _INDEX_TO_CODE, _PANEL, _DECODER)
    detail = render_detail(visits, _PANEL, form="string", precision=1)
    # The fixed feature axis is the point of the format — an unmeasured variable is an
    # all-nan row, not an omitted one.
    assert len(detail.splitlines()) == len(_PANEL.features)


def test_unknown_form_rejected(fixture_dense):
    visits = panel_visits_from_dense(fixture_dense, _INDEX_TO_CODE, _PANEL, _DECODER)
    with pytest.raises(ValueError, match="Unknown form"):
        render_detail(visits, _PANEL, form="csv", precision=1)


def test_locf_forward_fills_and_marks(fixture_dense):
    visits = panel_visits_from_dense(fixture_dense, _INDEX_TO_CODE, _PANEL, _DECODER)
    filled, marks = apply_locf(visits, _PANEL)
    # Temperature was measured only at visit 1; visit 2 inherits it and is marked.
    assert filled[1].values["Temperature"] == pytest.approx(36.94, abs=0.01)
    assert ("Temperature") in {feature for _, feature in marks}
    marked = render_detail(filled, _PANEL, form="string", precision=1, locf_marks=marks)
    assert '- Temperature: "36.9, 36.9**"' in marked


def test_locf_does_not_backfill():
    # Heart Rate first appears at visit 2, so visit 1 must stay missing.
    visits = [PanelVisit({"Temperature": 36.5}, 1.0), PanelVisit({"Heart Rate": 80.0}, 0.0)]
    filled, _ = apply_locf(visits, _PANEL)
    assert "Heart Rate" not in filled[0].values


def test_render_unit_range_toggles():
    assert render_unit_range(_PANEL, unit=True, reference_range=True).splitlines()[0] == (
        "- Heart Rate: Unit: bpm. Reference range: 60 - 100."
    )
    assert render_unit_range(_PANEL, unit=False, reference_range=True).splitlines()[0] == (
        "- Heart Rate: Reference range: 60 - 100."
    )
    assert render_unit_range(_PANEL, unit=False, reference_range=False) == ""


def test_render_record_times_modes():
    visits = [PanelVisit({}, 3.5), PanelVisit({}, 1.0), PanelVisit({}, 0.0)]
    assert render_record_times(visits, "index") == "0, 1, 2"
    assert render_record_times(visits, "days_before") == (
        "3.5 days before prediction, 1 day before prediction, prediction time"
    )
    with pytest.raises(ValueError, match="Unknown record_time_mode"):
        render_record_times(visits, "wallclock")


def test_render_examples_numbering():
    assert render_examples(()) == ""
    assert render_examples(("A",)) == "Here is an example of input information:\nExample #1:A\n"
    assert render_examples(("A", "B")) == (
        "Here are 2 examples of input information:\nExample #1:A\nExample #2:B\n"
    )


# ── Demographics ─────────────────────────────────────────────────────────────────────────


def test_demographics_from_static_and_birth_event():
    dense = {
        "code": np.array([[8], [3]]),
        "dim1/mask": np.array([[True], [True]]),
        "time_delta_days": np.array([np.nan, 365.25 * 52]),
    }
    assert subject_demographics(dense, [7], _INDEX_TO_CODE) == ("female", "52")


def test_demographics_degrade_to_placeholders(fixture_dense):
    # No GENDER static and no MEDS_BIRTH event in the window.
    sex, age = subject_demographics(fixture_dense, None, _INDEX_TO_CODE)
    assert sex == "patient of unrecorded sex" and age == "unknown"


# ── Truncation and assembly ──────────────────────────────────────────────────────────────


def test_visit_truncation_keeps_most_recent(fixture_dense):
    cfg = L4HConfig(max_visits=1, impute="none")
    history = serialize_history(fixture_dense, [7], _INDEX_TO_CODE, _PANEL, _DECODER, cfg)
    assert (history.n_events_total, history.n_events_dropped) == (2, 1)
    assert history.truncated
    # The surviving visit is the later one (90.7 bpm), not the earlier 94.0.
    assert '- Heart Rate: "90.7"' in history.text


def test_empty_panel_history_renders_placeholder():
    dense = {
        "code": np.array([[5]]),  # off-panel only
        "dim1/mask": np.array([[True]]),
        "numeric_value": np.array([[0.0]]),
        "time_delta_days": np.array([0.0]),
    }
    history = serialize_history(dense, [7], _INDEX_TO_CODE, _PANEL, _DECODER, L4HConfig())
    assert history.n_events_total == 0
    assert "The patient had 0 visits that occurred at ." in history.text
    assert "(no measurements of these features were recorded for this patient)" in history.text


def test_full_prompt_golden(fixture_dense):
    cfg = L4HConfig(form="string", impute="none", unit=True, reference_range=True)
    history = serialize_history(fixture_dense, [7], _INDEX_TO_CODE, _PANEL, _DECODER, cfg)
    prompt = build_user_prompt(history.text, "ICD//I50.9", 90.0, _PANEL, cfg)

    assert prompt == (
        "I will provide you with medical information from multiple Intensive Care Unit (ICU) "
        "visits of a patient, each characterized by a fixed number of features.\n"
        "\n"
        "Present multiple visit data of a patient in one batch. Represent each feature within "
        "this data as a string of values, separated by commas. Missing values are represented "
        "as `nan`.\n"
        "\n"
        "Your task is to assess the provided medical data and analyze the health records from "
        "ICU visits to determine the likelihood that the clinical event `ICD//I50.9` will be "
        "recorded for this patient within 90 days of their final visit. Please respond with "
        "only a floating-point number between 0 and 1, where a higher number suggests a "
        "greater likelihood of the event occurring.\n"
        "\n"
        "In situations where the data does not allow for a reasonable conclusion, respond with "
        "the phrase `I do not know` without any additional explanation.\n"
        "\n"
        "- Heart Rate: Unit: bpm. Reference range: 60 - 100.\n"
        "- Temperature: Unit: degrees Celsius. Reference range: 36.1 - 37.2.\n"
        "- Capillary refill rate: Unit: /. Reference range: /.\n"
        "\n"
        "\n"
        "\n"
        "Now please predict the patient below:\n"
        "The patient is a female, aged unknown years.\n"
        "The patient had 2 visits that occurred at 0, 1.\n"
        "Details of the features for each visit are as follows:\n"
        "\n"
        '- Heart Rate: "94.0, 90.7"\n'
        '- Temperature: "36.9, nan"\n'
        '- Capillary refill rate: "nan, nan"\n'
        "\n"
        "Please respond with only a floating-point number between 0 and 1, where a higher "
        "number suggests a greater likelihood of the event occurring. Do not include any "
        "additional explanation.\n"
        "RESPONSE:\n"
    )


def test_yes_no_response_format_swaps_both_slots(fixture_dense):
    cfg = L4HConfig(response_format="yes_no")
    history = serialize_history(fixture_dense, [7], _INDEX_TO_CODE, _PANEL, _DECODER, cfg)
    prompt = build_user_prompt(history.text, "ICD//I50.9", 90.0, _PANEL, cfg)
    assert "Please answer with exactly one word: Yes or No." in prompt
    assert "Answer with exactly one word: Yes or No." in prompt
    assert "floating-point number" not in prompt


def test_impute_mode_changes_the_input_format_description(fixture_dense):
    def prompt_for(impute: str) -> str:
        cfg = L4HConfig(impute=impute)
        history = serialize_history(fixture_dense, [7], _INDEX_TO_CODE, _PANEL, _DECODER, cfg)
        return build_user_prompt(history.text, "Q", 30.0, _PANEL, cfg)

    assert "Missing values are represented as `nan`." in prompt_for("none")
    assert "Missing values are represented as `nan`." not in prompt_for("locf")
    assert "Last Observation Carried Forward(LOCF)" in prompt_for("locf_marked")


# ── Fidelity guard ───────────────────────────────────────────────────────────────────────


def test_rendering_template_is_a_pure_refactor_of_the_reference_prompt():
    """The only difference from their USERPROMPT is the collapsed patient block."""
    assert USER_PROMPT.replace("{PATIENT_BLOCK}", PATIENT_BLOCK) == REFERENCE_USER_PROMPT


# ── Panel spec ───────────────────────────────────────────────────────────────────────────


def test_bundled_mimic_panel_matches_the_reference_feature_set():
    panel = load_panel(_resolve_panel_path("mimic_icu17"))
    assert [f.name for f in panel.features] == [
        # Their categorical block, in their order (including their "Glascow" spelling).
        "Capillary refill rate",
        "Glascow coma scale eye opening",
        "Glascow coma scale motor response",
        "Glascow coma scale total",
        "Glascow coma scale verbal response",
        # Their numeric block, in their order.
        "Diastolic blood pressure",
        "Fraction inspired oxygen",
        "Glucose",
        "Heart Rate",
        "Height",
        "Mean blood pressure",
        "Oxygen saturation",
        "Respiratory rate",
        "Systolic blood pressure",
        "Temperature",
        "Weight",
        "pH",
    ]
    # Unit and reference-range strings are reproduced verbatim from their unit/range JSON.
    by_name = {f.name: f for f in panel.features}
    assert by_name["Heart Rate"].unit == "Unit: bpm."
    assert by_name["Heart Rate"].range == "Reference range: 60 - 100."
    assert by_name["pH"].range == "Reference range: 7.35 - 7.45."


def test_panel_rejects_an_itemid_claimed_by_two_features(tmp_path):
    fp = tmp_path / "dupe.yaml"
    fp.write_text(
        "name: dupe\nfeatures:\n"
        "  - name: A\n    unit: u\n    range: r\n    sources:\n      - itemid: 1\n"
        "  - name: B\n    unit: u\n    range: r\n    sources:\n      - itemid: 1\n"
    )
    with pytest.raises(ValueError, match="claimed by both"):
        load_panel(fp)


def test_panel_rejects_empty_spec(tmp_path):
    fp = tmp_path / "empty.yaml"
    fp.write_text("name: empty\nfeatures: []\n")
    with pytest.raises(ValueError, match="no features"):
        load_panel(fp)


def test_unknown_panel_name_lists_the_bundled_ones():
    with pytest.raises(FileNotFoundError, match="mimic_icu17"):
        _resolve_panel_path("no_such_panel")


# ── Config validation and fingerprinting ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"form": "csv"}, "form must be"),
        ({"impute": "mean"}, "impute must be"),
        ({"response_format": "logprob"}, "response_format must be"),
        ({"max_visits": 0}, "max_visits must be"),
    ],
)
def test_invalid_config_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        L4HConfig(**kwargs)


def test_template_hash_is_stable_and_covers_panel_and_knobs():
    cfg = L4HConfig()
    base = prompt_template_hash(_PANEL, cfg)
    assert base == prompt_template_hash(_PANEL, cfg) and len(base) == 16
    # A rendering knob changes the hash.
    assert prompt_template_hash(_PANEL, L4HConfig(form="batches")) != base
    # So does an edit to the panel — here a unit string.
    edited = Panel(
        "test",
        (
            PanelFeature("Heart Rate", "Unit: BPM.", "Reference range: 60 - 100.", (PanelSource(220045),)),
            *_PANEL.features[1:],
        ),
    )
    assert prompt_template_hash(edited, cfg) != base
