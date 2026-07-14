"""Probability-extraction tests for ``every_query.llm_baseline.llm_predict``.

Mocked OpenAI client throughout — covers logprob extraction, tokenizer-variant Yes/No
tokens, the guided fallback path, total parse failure, and transport-error retry/raise
semantics.  No server involved.
"""

import asyncio
import math
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from openai import APIConnectionError

from every_query.llm_baseline.llm_predict import (
    GUIDED_REGEX,
    LLMPredictor,
    parse_guided_probability,
    yes_no_probability,
)


def _logprob_response(top: list[tuple[str, float]]):
    """Fake chat completion carrying first-token top logprobs, shaped like the openai SDK's."""
    entries = [SimpleNamespace(token=token, logprob=logprob) for token, logprob in top]
    first = SimpleNamespace(token=top[0][0], logprob=top[0][1], top_logprobs=entries)
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=top[0][0]),
                logprobs=SimpleNamespace(content=[first]),
            )
        ]
    )


def _text_response(text: str | None):
    """Fake chat completion with plain text content and no logprobs."""
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text), logprobs=None)])


def _predictor(create_mock: AsyncMock, **kwargs) -> LLMPredictor:
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock)))
    return LLMPredictor(model="test-model", client=client, **kwargs)


# ── Pure extraction functions ────────────────────────────────────────────────────────────


def test_yes_no_probability_both_sides():
    p = yes_no_probability([("Yes", -0.2), (" No", -1.8)])
    assert p == pytest.approx(1.0 / (1.0 + math.exp(-1.8 - (-0.2))))


def test_yes_no_probability_tokenizer_variants():
    baseline = yes_no_probability([("Yes", -0.2), ("No", -1.8)])
    # Leading-space, BPE-marker, and casing variants all land on the same sides.
    for yes_tok, no_tok in [(" Yes", " No"), ("ĠYes", "ĠNo"), ("▁yes", "▁NO"), ("YES", "no")]:
        assert yes_no_probability([(yes_tok, -0.2), (no_tok, -1.8)]) == pytest.approx(baseline)


def test_yes_no_probability_max_over_duplicate_variants():
    # Two yes-variants: the higher logprob (-0.2) must win, not the later/lower one.
    p = yes_no_probability([("Yes", -0.2), (" yes", -4.0), ("No", -1.8)])
    assert p == pytest.approx(1.0 / (1.0 + math.exp(-1.8 - (-0.2))))


def test_yes_no_probability_one_sided_uses_topk_floor():
    # Only "Yes" present; "No" is bounded by the smallest returned logprob (-9.0).
    p = yes_no_probability([("Yes", -0.01), ("Maybe", -6.0), ("The", -9.0)])
    assert p == pytest.approx(1.0 / (1.0 + math.exp(-9.0 - (-0.01))))


def test_yes_no_probability_neither_side():
    assert yes_no_probability([("Maybe", -0.5), ("0", -1.0)]) is None
    assert yes_no_probability([]) is None


def test_parse_guided_probability():
    assert parse_guided_probability("0.85") == 0.85
    assert parse_guided_probability(" 1.0\n") == 1.0
    assert parse_guided_probability("0") == 0.0
    assert parse_guided_probability("1") == 1.0
    assert parse_guided_probability("high") is None
    assert parse_guided_probability("") is None
    assert parse_guided_probability(None) is None
    # Out-of-range numerics are rejected, never clamped.
    assert parse_guided_probability("1.5") is None


# ── Predictor request flow (mocked client) ───────────────────────────────────────────────


def test_logprob_method_happy_path():
    create = AsyncMock(return_value=_logprob_response([("Yes", -0.2), (" No", -1.8)]))
    predictor = _predictor(create)

    result = asyncio.run(predictor.predict_prob("history", "question"))

    assert result.method_used == "logprob"
    assert not result.parse_failed
    assert result.prob == pytest.approx(1.0 / (1.0 + math.exp(-1.6)))
    create.assert_awaited_once()
    kwargs = create.await_args.kwargs
    assert kwargs["max_tokens"] == 1 and kwargs["logprobs"] is True and kwargs["top_logprobs"] == 20
    assert kwargs["temperature"] == 0.0 and kwargs["seed"] == 0
    # The Yes/No instruction reaches the user turn.
    assert "Answer with exactly one word: Yes or No." in kwargs["messages"][1]["content"]


def test_logprob_falls_back_to_guided_when_yes_no_absent():
    create = AsyncMock(
        side_effect=[
            _logprob_response([("Maybe", -0.5), ("It", -1.0)]),  # no Yes/No in top-k
            _text_response("0.42"),
        ]
    )
    predictor = _predictor(create)

    result = asyncio.run(predictor.predict_prob("history", "question"))

    assert result.method_used == "guided_fallback"
    assert result.prob == pytest.approx(0.42)
    assert not result.parse_failed
    assert predictor.n_guided_fallbacks == 1
    guided_kwargs = create.await_args_list[1].kwargs
    assert guided_kwargs["extra_body"] == {"guided_regex": GUIDED_REGEX}


def test_guided_primary_method():
    create = AsyncMock(return_value=_text_response("0.07"))
    predictor = _predictor(create, method="guided")

    result = asyncio.run(predictor.predict_prob("history", "question"))

    assert result.method_used == "guided"
    assert result.prob == pytest.approx(0.07)
    create.assert_awaited_once()  # no logprob attempt for the guided primary method


def test_total_parse_failure_uses_per_row_fallback_and_flags():
    create = AsyncMock(
        side_effect=[
            _logprob_response([("Maybe", -0.5)]),
            _text_response("not a number"),
        ]
    )
    predictor = _predictor(create)

    result = asyncio.run(predictor.predict_prob("history", "question", fallback_prob=0.123))

    assert result.parse_failed
    assert result.method_used == "fallback"
    assert result.prob == pytest.approx(0.123)  # per-row (task-prevalence) fallback wins
    assert predictor.n_parse_failures == 1


def test_total_parse_failure_default_fallback():
    create = AsyncMock(side_effect=[_logprob_response([("Maybe", -0.5)]), _text_response(None)])
    predictor = _predictor(create, fallback_prob=0.25)

    result = asyncio.run(predictor.predict_prob("history", "question"))

    assert result.parse_failed and result.prob == pytest.approx(0.25)


def test_transient_error_retries_then_succeeds():
    err = APIConnectionError(request=httpx.Request("POST", "http://localhost:8000/v1"))
    create = AsyncMock(side_effect=[err, _logprob_response([("Yes", -0.1), ("No", -3.0)])])
    predictor = _predictor(create, max_retries=2)

    with patch("every_query.llm_baseline.llm_predict.asyncio.sleep", new=AsyncMock()) as sleep:
        result = asyncio.run(predictor.predict_prob("history", "question"))

    assert result.method_used == "logprob" and not result.parse_failed
    assert create.await_count == 2
    sleep.assert_awaited_once()


def test_transport_failure_after_retries_raises_not_fallback():
    err = APIConnectionError(request=httpx.Request("POST", "http://localhost:8000/v1"))
    create = AsyncMock(side_effect=err)
    predictor = _predictor(create, max_retries=1)

    with (
        patch("every_query.llm_baseline.llm_predict.asyncio.sleep", new=AsyncMock()),
        pytest.raises(APIConnectionError),
    ):
        asyncio.run(predictor.predict_prob("history", "question"))

    assert predictor.n_parse_failures == 0  # transport errors are never coerced to fallbacks


def test_missing_top_logprobs_falls_back_to_guided_not_half():
    # Server ignored top_logprobs (only the sampled token came back): a lone one-sided
    # token must not be scored — the top-k floor bound would collapse every row to a
    # constant 0.5 flagged as a clean logprob success.  It must fall through to guided.
    no_topk = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="Yes"),
                logprobs=SimpleNamespace(
                    content=[SimpleNamespace(token="Yes", logprob=-0.01, top_logprobs=None)]
                ),
            )
        ]
    )
    create = AsyncMock(side_effect=[no_topk, _text_response("0.9")])
    predictor = _predictor(create)

    result = asyncio.run(predictor.predict_prob("history", "question"))

    assert result.method_used == "guided_fallback"
    assert result.prob == pytest.approx(0.9)
    assert not result.parse_failed


def test_extra_body_merged_with_per_call_precedence():
    create = AsyncMock(
        side_effect=[
            _logprob_response([("Maybe", -0.5), ("It", -1.0)]),  # forces the guided fallback
            _text_response("0.3"),
        ]
    )
    predictor = _predictor(
        create,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}, "guided_regex": "WRONG"},
    )

    asyncio.run(predictor.predict_prob("history", "question"))

    logprob_kwargs = create.await_args_list[0].kwargs
    assert logprob_kwargs["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert logprob_kwargs["extra_body"]["guided_regex"] == "WRONG"
    guided_kwargs = create.await_args_list[1].kwargs
    # Per-call extras win: the guided attempt's real regex overrides the instance-level one.
    assert guided_kwargs["extra_body"]["guided_regex"] == GUIDED_REGEX
    assert guided_kwargs["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}


def test_invalid_method_rejected_at_construction():
    with pytest.raises(ValueError, match="method must be"):
        LLMPredictor(model="m", method="freeform", client=SimpleNamespace())
