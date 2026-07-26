"""Integration tests for ``EQ_llm_predict``.

Uses the session-scoped ``tensorized_cohort_dir`` fixture (the same MTD-preprocessed
simple-static cohort ``demo_dataset`` builds on) plus a ``TaskQuerySchema``-conformant
tasks directory.  Covers:

- the ``dry_run`` smoke path via the real console script (subprocess — no server contact);
- the full in-process pipeline with a mocked OpenAI client (serialization → repeated Yes/No
  sampling → empirical-fraction probability → sharded ``PredictionSchema`` output → merge),
  verified to be consumable by ``EQ_evaluate``'s ``compute_metrics`` unchanged;
- ``resume=true`` skipping already-completed rows.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import polars as pl
import pyarrow.parquet as pq
import pytest

from every_query.data.schema import TaskQuerySchema
from every_query.evaluate.metrics import compute_metrics
from every_query.llm_baseline.__main__ import METADATA_KEY
from every_query.predict.schema import PredictionSchema

_VENV_BIN = str(Path(sys.executable).parent)

# Train-split subjects of the simple-static testing cohort, with prediction times inside
# their event sequences (same values the root conftest's ``demo_dataset`` uses).
_SUBJECT_PRED_TIMES = {
    239684: datetime(2010, 5, 11, 18, 0),
    1195293: datetime(2010, 6, 20, 20, 30),
}
_QUERY_CODES = ["HR", "TEMP"]
_DURATION_DAYS = 30.0

# Deterministic vote: the fake server answers Yes on _N_YES of every _N_SAMPLES samples.
_N_SAMPLES = 4
_N_YES = 3
_EXPECTED_PROB = _N_YES / _N_SAMPLES


@pytest.fixture(scope="module")
def llm_tasks_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A flat tasks dir: 2 subjects x 2 query codes, so each subject is one dedup group."""
    tasks_dir = tmp_path_factory.mktemp("llm_tasks")
    rows = [
        {
            "subject_id": subject,
            "prediction_time": pred_time,
            # One censored row (null) among the four, so the boolean_value round-trip
            # through the dataset's (censor, occurs) collapse is exercised.
            "boolean_value": None if i == 0 else i % 2 == 0,
            "query": query,
            "duration_days": _DURATION_DAYS,
        }
        for i, ((subject, pred_time), query) in enumerate(
            (sp, q) for sp in _SUBJECT_PRED_TIMES.items() for q in _QUERY_CODES
        )
    ]
    aligned = TaskQuerySchema.align(
        pl.DataFrame(rows).cast({"prediction_time": pl.Datetime("us")}).to_arrow()
    )
    pq.write_table(aligned, tasks_dir / "tasks.parquet")
    return tasks_dir


def _overrides(tasks_dir: Path, cohort_dir: Path, output_dir: Path, **extra) -> list[str]:
    base = {
        "tasks_dir": str(tasks_dir),
        "tensorized_cohort_dir": str(cohort_dir),
        "output_dir": str(output_dir),
        "split": "train",
        "n_samples": _N_SAMPLES,
        **extra,
    }
    return [f"{key}={value}" for key, value in base.items()]


class _FakeCompletions:
    """Returns an n-way completion of _N_YES 'Yes' + rest 'No', shaped like the openai SDK."""

    def __init__(self) -> None:
        self.n_calls = 0

    async def create(self, **kwargs):
        self.n_calls += 1
        n = kwargs.get("n", 1)
        answers = ["Yes"] * min(_N_YES, n) + ["No"] * max(0, n - _N_YES)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=a)) for a in answers])


def _run_main_with_fake_client(argv_overrides: list[str]) -> _FakeCompletions:
    """Invoke the hydra-decorated main in-process with the OpenAI client mocked out."""
    from every_query.llm_baseline.__main__ import main

    completions = _FakeCompletions()
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    with (
        patch("every_query.llm_baseline.llm_predict.AsyncOpenAI", return_value=fake_client),
        patch.object(sys, "argv", ["EQ_llm_predict", *argv_overrides]),
    ):
        main()
    return completions


def test_llm_predict_dry_run_subprocess(tensorized_cohort_dir, llm_tasks_dir, tmp_path):
    """The dry-run smoke path prints serialized prompts and never writes output."""
    output_dir = tmp_path / "out"
    result = subprocess.run(
        ["EQ_llm_predict", *_overrides(llm_tasks_dir, tensorized_cohort_dir, output_dir, dry_run="true")],
        capture_output=True,
        text=True,
        env={"PATH": f"{_VENV_BIN}:/usr/bin:/bin"},
        timeout=300,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "Patient history (most recent last):" in result.stdout
    assert "Question: Will code" in result.stdout
    assert "Answer with exactly one word: Yes or No." in result.stdout
    assert "no requests were sent" in result.stdout
    assert not (output_dir / "shards").exists() or not list((output_dir / "shards").glob("*.parquet"))


def test_llm_predict_full_run_and_resume(tensorized_cohort_dir, llm_tasks_dir, tmp_path):
    """Full mocked-server run: PredictionSchema output, EQ_evaluate consumption, resume."""
    output_dir = tmp_path / "out"
    _run_main_with_fake_client(_overrides(llm_tasks_dir, tensorized_cohort_dir, output_dir))

    predictions_fp = output_dir / "predictions.parquet"
    table = PredictionSchema.align(pq.read_table(predictions_fp))
    predictions = pl.from_arrow(table)

    n_tasks = len(_SUBJECT_PRED_TIMES) * len(_QUERY_CODES)
    assert predictions.height == n_tasks
    # Every row votes the same way, so the empirical fraction is constant; censor_prob is 0.
    assert predictions["occurs_prob"].to_list() == pytest.approx([_EXPECTED_PROB] * n_tasks)
    assert predictions["censor_prob"].to_list() == [0.0] * n_tasks
    # boolean_value round-trips (one null / censored row among the four).
    assert predictions["boolean_value"].null_count() == 1

    # EQ_evaluate's metrics pipeline consumes the output unchanged.
    metrics = compute_metrics(predictions)
    assert metrics.height == predictions.select("query", "duration_days").unique().height

    # Sidecar diagnostics: vote counts recorded, no parse failures with the fake server.
    details = pl.read_parquet(output_dir / "details.parquet")
    assert details.height == n_tasks
    assert details["parse_failed"].sum() == 0
    assert set(details["n_yes"].to_list()) == {_N_YES}
    assert set(details["n_no"].to_list()) == {_N_SAMPLES - _N_YES}
    assert set(details["n_unparsed"].to_list()) == {0}

    # Reproducibility metadata is on the merged parquet.
    meta = json.loads(pq.read_schema(predictions_fp).metadata[b"every_query_llm_baseline"])
    assert meta["n_samples"] == _N_SAMPLES
    assert meta["prompt_style"] == "event_stream"
    assert meta["response_format"] == "yes_no"
    assert meta["style_fields"]["max_events"] == 256
    assert meta["style_fields"]["code_descriptions_used"] is False
    assert len(meta["prompt_template_hash"]) == 16

    # A second run without resume refuses to mix into the same output dir (the underlying
    # FileExistsError is converted to SystemExit(1) by hydra's error handling).
    with pytest.raises(SystemExit):
        _run_main_with_fake_client(_overrides(llm_tasks_dir, tensorized_cohort_dir, output_dir))

    # …and with resume=true it finds nothing pending and leaves the output unchanged.
    resumed = _run_main_with_fake_client(
        _overrides(llm_tasks_dir, tensorized_cohort_dir, output_dir, resume="true")
    )
    assert resumed.n_calls == 0
    assert pl.read_parquet(predictions_fp).height == n_tasks


def test_llm_predict_partial_resume(tensorized_cohort_dir, llm_tasks_dir, tmp_path):
    """resume=true after an interrupted (limit-capped) run issues only the remaining requests."""
    output_dir = tmp_path / "out"
    n_tasks = len(_SUBJECT_PRED_TIMES) * len(_QUERY_CODES)

    first = _run_main_with_fake_client(_overrides(llm_tasks_dir, tensorized_cohort_dir, output_dir, limit=2))
    assert first.n_calls == 2

    # A leftover staging file from a hard-killed run must never be swept into the merge
    # (regression: pyarrow directory-level dataset discovery reads any extension).
    (output_dir / "shards" / "shard_leftover.parquet.tmp").write_bytes(b"not a parquet")

    second = _run_main_with_fake_client(
        _overrides(llm_tasks_dir, tensorized_cohort_dir, output_dir, resume="true")
    )
    assert second.n_calls == n_tasks - 2

    predictions = pl.read_parquet(output_dir / "predictions.parquet")
    assert predictions.height == n_tasks
    id_cols = ["subject_id", "prediction_time", "query", "duration_days"]
    assert predictions.unique(subset=id_cols).height == n_tasks
    details = pl.read_parquet(output_dir / "details.parquet")
    assert details.height == n_tasks


def test_llm_predict_resume_refuses_config_mismatch(tensorized_cohort_dir, llm_tasks_dir, tmp_path):
    """resume=true onto shards produced under a different model raises instead of mixing.

    The underlying ValueError is converted to SystemExit(1) by hydra's error handling.
    """
    output_dir = tmp_path / "out"
    _run_main_with_fake_client(_overrides(llm_tasks_dir, tensorized_cohort_dir, output_dir, limit=1))

    with pytest.raises(SystemExit):
        _run_main_with_fake_client(
            _overrides(
                llm_tasks_dir, tensorized_cohort_dir, output_dir, resume="true", model="some/other-model"
            )
        )


# ── llm4healthcare prompt style ──────────────────────────────────────────────────────────

# The testing cohort's codes are bare strings (HR, TEMP) with no MIMIC itemids, so the panel
# is keyed by `code:` rather than `itemid:` — the same mechanism the bundled mimic_icu17 panel
# uses, exercised over a non-MIMIC vocabulary.
_TEST_PANEL = """\
name: test_vitals
features:
  - name: Heart Rate
    unit: "Unit: bpm."
    range: "Reference range: 60 - 100."
    sources:
      - code: HR
  - name: Temperature
    unit: "Unit: degrees Celsius."
    range: "Reference range: 36.1 - 37.2."
    sources:
      - code: TEMP
  - name: pH
    unit: "Unit: /."
    range: "Reference range: 7.35 - 7.45."
    sources:
      - code: NOT_IN_THIS_COHORT
"""


class _FakeFloatCompletions:
    """Returns an n-way completion of float answers, shaped like the openai SDK's."""

    def __init__(self, answer: str = "0.7") -> None:
        self.n_calls = 0
        self.answer = answer
        self.last_kwargs: dict = {}

    async def create(self, **kwargs):
        self.n_calls += 1
        self.last_kwargs = kwargs
        n = kwargs.get("n", 1)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.answer)) for _ in range(n)]
        )


@pytest.fixture
def test_panel_fp(tmp_path: Path) -> Path:
    fp = tmp_path / "test_vitals.yaml"
    fp.write_text(_TEST_PANEL)
    return fp


def test_llm4healthcare_style_end_to_end(
    llm_tasks_dir: Path, tensorized_cohort_dir: Path, tmp_path: Path, test_panel_fp: Path
):
    """The panel style runs the same pipeline and produces EQ_evaluate-consumable output."""
    output_dir = tmp_path / "out"
    completions = _FakeFloatCompletions("0.7")
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    argv = _overrides(
        llm_tasks_dir,
        tensorized_cohort_dir,
        output_dir,
        prompt_style="llm4healthcare",
        **{"l4h.panel": str(test_panel_fp)},
    )
    with (
        patch("every_query.llm_baseline.llm_predict.AsyncOpenAI", return_value=fake_client),
        patch.object(sys, "argv", ["EQ_llm_predict", *argv]),
    ):
        from every_query.llm_baseline.__main__ import main

        main()

    predictions = pl.read_parquet(output_dir / "predictions.parquet")
    assert predictions.height == 4
    assert predictions["occurs_prob"].to_list() == pytest.approx([0.7] * 4)
    # Still PredictionSchema-conformant and consumable by the metrics pipeline unchanged.
    PredictionSchema.align(predictions.to_arrow())
    assert compute_metrics(predictions).height > 0

    # Float mode records n_parsed and leaves the Yes/No counters at zero.
    details = pl.read_parquet(output_dir / "details.parquet")
    assert details["parse_failed"].sum() == 0
    assert set(details["n_parsed"].to_list()) == {_N_SAMPLES}
    assert set(details["n_yes"].to_list()) == {0}

    # guided_choice must not be sent in float mode even though the config leaves it true.
    assert "extra_body" not in completions.last_kwargs

    # The rendered prompt is the reference template over the panel's fixed feature axis.
    user_turn = completions.last_kwargs["messages"][1]["content"]
    assert user_turn.startswith("I will provide you with medical information from multiple")
    assert "- Heart Rate: Unit: bpm. Reference range: 60 - 100." in user_turn
    assert "Please respond with only a floating-point number between 0 and 1" in user_turn
    assert user_turn.rstrip().endswith("RESPONSE:")
    # Every panel feature gets a row, including the one absent from this cohort.
    for feature in ("Heart Rate", "Temperature", "pH"):
        assert f'- {feature}: "' in user_turn

    # Metadata records the style and its knobs, so a resume cannot mix styles.
    meta = json.loads(pq.read_schema(output_dir / "predictions.parquet").metadata[METADATA_KEY])
    assert meta["prompt_style"] == "llm4healthcare"
    assert meta["response_format"] == "float"
    assert meta["style_fields"]["panel"] == "test_vitals"
    assert meta["style_fields"]["form"] == "string"


def test_resume_refuses_to_mix_prompt_styles(
    llm_tasks_dir: Path, tensorized_cohort_dir: Path, tmp_path: Path, test_panel_fp: Path
):
    """Shards written under one prompt style may never be extended under another."""
    output_dir = tmp_path / "out"
    _run_main_with_fake_client(_overrides(llm_tasks_dir, tensorized_cohort_dir, output_dir))

    completions = _FakeFloatCompletions("0.7")
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    argv = _overrides(
        llm_tasks_dir,
        tensorized_cohort_dir,
        output_dir,
        resume="true",
        prompt_style="llm4healthcare",
        **{"l4h.panel": str(test_panel_fp)},
    )
    with (
        patch("every_query.llm_baseline.llm_predict.AsyncOpenAI", return_value=fake_client),
        patch.object(sys, "argv", ["EQ_llm_predict", *argv]),
        pytest.raises(SystemExit),  # hydra converts the ValueError to SystemExit(1)
    ):
        from every_query.llm_baseline.__main__ import main

        main()
