"""Probability-extraction tests for ``every_query.llm_baseline.llm_predict``.

Mocked OpenAI client throughout — covers the pure Yes/No parse + empirical-fraction helpers,
the repeated-sampling request flow (n-way completion, guided_choice, smoothing), total parse
failure, and transport-error retry/raise semantics.  No server involved.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from openai import APIConnectionError

from every_query.llm_baseline.llm_predict import (
    YES_NO_CHOICES,
    LLMPredictor,
    empirical_probability,
    parse_yes_no,
)


def _sample_response(answers: list[str | None]):
    """Fake n-way chat completion: one choice per answer string, shaped like the openai SDK's."""
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=a)) for a in answers])


def _predictor(create_mock: AsyncMock, **kwargs) -> LLMPredictor:
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock)))
    # temperature > 0 by default so the n_samples>1 guard doesn't trip in tests that don't care.
    kwargs.setdefault("temperature", 0.7)
    return LLMPredictor(model="test-model", client=client, **kwargs)


# ── Pure helpers ───────────────────────────────────────────────────────────────────────────


def test_parse_yes_no():
    assert [parse_yes_no(t) for t in ["Yes", " no", "Yes.", "No, because ...", "YES\n"]] == [
        "yes",
        "no",
        "yes",
        "no",
        "yes",
    ]
    assert parse_yes_no("Maybe") is None
    assert parse_yes_no("") is None
    assert parse_yes_no(None) is None


def test_empirical_probability():
    assert empirical_probability(3, 1) == pytest.approx(0.75)
    assert empirical_probability(4, 0) == pytest.approx(1.0)
    assert empirical_probability(0, 4) == pytest.approx(0.0)
    # Symmetric Laplace prior pulls the estimate off the endpoints.
    assert empirical_probability(4, 0, smoothing=1.0) == pytest.approx(5.0 / 6.0)
    # No parseable votes → None (caller emits the fallback).
    assert empirical_probability(0, 0) is None


# ── Predictor request flow (mocked client) ───────────────────────────────────────────────


def test_sample_happy_path_empirical_fraction():
    create = AsyncMock(return_value=_sample_response(["Yes", "Yes", "Yes", "No"]))
    predictor = _predictor(create, n_samples=4)

    result = asyncio.run(predictor.predict_prob("history", "question"))

    assert not result.parse_failed
    assert (result.n_yes, result.n_no, result.n_unparsed) == (3, 1, 0)
    assert result.prob == pytest.approx(0.75)
    create.assert_awaited_once()
    kwargs = create.await_args.kwargs
    assert kwargs["n"] == 4 and kwargs["max_tokens"] == 8
    assert kwargs["temperature"] == 0.7 and kwargs["seed"] == 0
    # guided_choice constrains the answer to exactly Yes/No.
    assert kwargs["extra_body"] == {"guided_choice": list(YES_NO_CHOICES)}
    # The Yes/No instruction reaches the user turn.
    assert "Answer with exactly one word: Yes or No." in kwargs["messages"][1]["content"]


def test_unparseable_samples_excluded_from_denominator():
    # One sample mode-collapses to prose; it's counted as unparsed, not as a No.
    create = AsyncMock(return_value=_sample_response(["Yes", "Yes", "No", "Maybe not"]))
    predictor = _predictor(create, n_samples=4)

    result = asyncio.run(predictor.predict_prob("history", "question"))

    assert (result.n_yes, result.n_no, result.n_unparsed) == (2, 1, 1)
    assert result.prob == pytest.approx(2.0 / 3.0)  # denominator is parseable votes only
    assert not result.parse_failed
    assert predictor.n_unparsed_samples == 1


def test_smoothing_applied():
    create = AsyncMock(return_value=_sample_response(["Yes", "Yes", "Yes", "Yes"]))
    predictor = _predictor(create, n_samples=4, smoothing=1.0)

    result = asyncio.run(predictor.predict_prob("history", "question"))

    assert (result.n_yes, result.n_no) == (4, 0)
    assert result.prob == pytest.approx(5.0 / 6.0)  # (4 + 1) / (4 + 2)


def test_guided_choice_disabled_sends_no_extra_body():
    create = AsyncMock(return_value=_sample_response(["Yes", "No"]))
    predictor = _predictor(create, n_samples=2, guided_choice=False)

    asyncio.run(predictor.predict_prob("history", "question"))

    assert "extra_body" not in create.await_args.kwargs


def test_total_parse_failure_uses_per_row_fallback_and_flags():
    create = AsyncMock(return_value=_sample_response(["Maybe", "I cannot say", None]))
    predictor = _predictor(create, n_samples=3)

    result = asyncio.run(predictor.predict_prob("history", "question", fallback_prob=0.123))

    assert result.parse_failed
    assert (result.n_yes, result.n_no, result.n_unparsed) == (0, 0, 3)
    assert result.prob == pytest.approx(0.123)  # per-row (task-prevalence) fallback wins
    assert predictor.n_parse_failures == 1


def test_total_parse_failure_default_fallback():
    create = AsyncMock(return_value=_sample_response([None, "hmm"]))
    predictor = _predictor(create, n_samples=2, fallback_prob=0.25)

    result = asyncio.run(predictor.predict_prob("history", "question"))

    assert result.parse_failed and result.prob == pytest.approx(0.25)


def test_transient_error_retries_then_succeeds():
    err = APIConnectionError(request=httpx.Request("POST", "http://localhost:8000/v1"))
    create = AsyncMock(side_effect=[err, _sample_response(["Yes", "No"])])
    predictor = _predictor(create, n_samples=2, max_retries=2)

    with patch("every_query.llm_baseline.llm_predict.asyncio.sleep", new=AsyncMock()) as sleep:
        result = asyncio.run(predictor.predict_prob("history", "question"))

    assert not result.parse_failed and result.prob == pytest.approx(0.5)
    assert create.await_count == 2
    sleep.assert_awaited_once()


def test_transport_failure_after_retries_raises_not_fallback():
    err = APIConnectionError(request=httpx.Request("POST", "http://localhost:8000/v1"))
    create = AsyncMock(side_effect=err)
    predictor = _predictor(create, n_samples=2, max_retries=1)

    with (
        patch("every_query.llm_baseline.llm_predict.asyncio.sleep", new=AsyncMock()),
        pytest.raises(APIConnectionError),
    ):
        asyncio.run(predictor.predict_prob("history", "question"))

    assert predictor.n_parse_failures == 0  # transport errors are never coerced to fallbacks


def test_extra_body_merged_with_per_call_precedence():
    create = AsyncMock(return_value=_sample_response(["Yes", "No"]))
    predictor = _predictor(
        create,
        n_samples=2,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}, "guided_choice": ["WRONG"]},
    )

    asyncio.run(predictor.predict_prob("history", "question"))

    kwargs = create.await_args.kwargs
    # Instance-level extra survives, and the per-call guided_choice overrides the instance one.
    assert kwargs["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert kwargs["extra_body"]["guided_choice"] == list(YES_NO_CHOICES)


def test_temperature_zero_with_repeated_sampling_rejected():
    with pytest.raises(ValueError, match="temperature"):
        LLMPredictor(model="m", n_samples=8, temperature=0.0, client=SimpleNamespace())


def test_invalid_n_samples_rejected_at_construction():
    with pytest.raises(ValueError, match="n_samples must be"):
        LLMPredictor(model="m", n_samples=0, client=SimpleNamespace())
