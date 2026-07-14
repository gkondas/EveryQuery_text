"""Probability extraction from a locally-served LLM over an OpenAI-compatible API (vLLM).

This is the crux of the baseline.  Free-text "output a probability between 0 and 1" mode-
collapses to a handful of round numbers (0.1 / 0.5 / 0.9), which makes the baseline look
artificially bad.  Two methods are implemented behind the ``method`` config field:

1. ``logprob`` (default, primary) — constrain the model to a one-word ``Yes`` / ``No``
   answer, request the top-k token logprobs of the first generated token, and softmax over
   just the Yes/No pair to get a continuous probability.  Tokenizer variation
   (leading-space / BPE-marker / casing variants) is normalized away.  When only one of the
   two targets appears in the top-k, the other side is bounded by the smallest returned
   logprob (its true logprob must be ≤ that floor), giving a tight estimate exactly in the
   confident cases where one side falls out of the top-k.  When *neither* appears, the
   sample falls back to method 2.
2. ``guided`` (fallback / comparison) — free-text numeric answer constrained by vLLM's
   ``guided_regex`` structured output, then parsed.

Failure semantics are explicit, never silent:

- **Parse failure** (both methods exhausted): the configured fallback probability is
  written, the row is flagged ``parse_failed=True`` in the sidecar details parquet, and the
  per-run failure rate is logged.
- **Transport failure** (connection / timeout / 429 / 5xx after retries with exponential
  backoff): the run *raises*.  Writing a fallback for transport errors would silently
  poison the output; the CLI's sharded output + ``resume=true`` handles restarts instead.
"""

import asyncio
import logging
import math
import re
from dataclasses import dataclass

from openai import APIConnectionError, AsyncOpenAI, InternalServerError, RateLimitError

from every_query.llm_baseline.serialize import SYSTEM_PROMPT, build_user_prompt

logger = logging.getLogger(__name__)

# vLLM structured-output constraint for the guided method (see module docstring).
GUIDED_REGEX = r"(0\.\d{1,4}|1\.0|0|1)"

# APITimeoutError subclasses APIConnectionError, so timeouts are covered by the first entry.
_RETRYABLE_ERRORS = (APIConnectionError, RateLimitError, InternalServerError)
_BACKOFF_BASE_SECONDS = 1.0
_BACKOFF_CAP_SECONDS = 30.0


@dataclass(frozen=True)
class ProbabilityResult:
    """One row's extracted probability plus how it was obtained.

    ``method_used`` is one of ``logprob``, ``guided`` (primary guided method), ``guided_fallback`` (logprob
    method fell through to guided), or ``fallback`` (total parse failure — ``prob`` is the configured
    fallback and ``parse_failed`` is True).
    """

    prob: float
    parse_failed: bool
    method_used: str


def _normalize_token(token: str) -> str:
    """Normalize a tokenizer-variant token string for Yes/No matching.

    Handles leading/trailing whitespace, BPE space markers (``Ġ``, ``▁``), and casing.

    Examples:
        >>> [_normalize_token(t) for t in ["Yes", " Yes", "ĠYes", "▁yes", "YES", " No"]]
        ['yes', 'yes', 'yes', 'yes', 'yes', 'no']
    """
    return token.replace("Ġ", " ").replace("▁", " ").strip().lower()


def yes_no_probability(top_logprobs: list[tuple[str, float]]) -> float | None:
    """Compute P(Yes) by softmaxing over the Yes/No token logprobs in a top-k list.

    ``top_logprobs`` is ``(token, logprob)`` pairs for the first generated token position.
    Token strings are normalized via :func:`_normalize_token`, and the max logprob per side
    is used when multiple variants match.  If exactly one side is present, the missing
    side's logprob is bounded by the smallest logprob in the list (it must be ≤ every
    returned entry), which is tight precisely when the model is confident.  Returns ``None``
    when neither side is found — the caller should fall back to the guided method.

    Examples:
        Both sides present — plain two-way softmax:

        >>> p = yes_no_probability([("Yes", -0.2), (" No", -1.8)])
        >>> round(p, 4)
        0.832

        Tokenizer variants collapse onto the same side (max logprob wins):

        >>> p2 = yes_no_probability([("ĠYes", -0.2), ("YES", -5.0), (" No", -1.8)])
        >>> p2 == p
        True

        One side missing — bounded by the top-k floor, near 1 when the model is confident:

        >>> p = yes_no_probability([("Yes", -0.01), ("Maybe", -6.0), ("The", -9.0)])
        >>> p > 0.999
        True

        Neither side present:

        >>> yes_no_probability([("Maybe", -0.5), ("0", -1.0)]) is None
        True
        >>> yes_no_probability([]) is None
        True
    """
    if not top_logprobs:
        return None

    yes_lp: float | None = None
    no_lp: float | None = None
    for token, logprob in top_logprobs:
        match _normalize_token(token):
            case "yes":
                yes_lp = logprob if yes_lp is None else max(yes_lp, logprob)
            case "no":
                no_lp = logprob if no_lp is None else max(no_lp, logprob)

    if yes_lp is None and no_lp is None:
        return None

    floor = min(lp for _, lp in top_logprobs)
    if yes_lp is None:
        yes_lp = floor
    if no_lp is None:
        no_lp = floor
    return 1.0 / (1.0 + math.exp(no_lp - yes_lp))


def parse_guided_probability(text: str | None) -> float | None:
    """Parse the guided method's numeric answer into a probability, or ``None`` on failure.

    Accepts anything matching :data:`GUIDED_REGEX` (possibly embedded in surrounding
    whitespace / text, since even guided decoding can be preceded by stray whitespace
    depending on server version).  Out-of-range values return ``None`` — never silently
    clamped.

    Examples:
        >>> parse_guided_probability("0.85")
        0.85
        >>> parse_guided_probability(" 1.0\\n")
        1.0
        >>> parse_guided_probability("0")
        0.0
        >>> parse_guided_probability("The probability is 0.3") is None  # not a bare number
        False
        >>> parse_guided_probability("high") is None
        True
        >>> parse_guided_probability(None) is None
        True
        >>> parse_guided_probability("...") is None
        True
    """
    if not text:
        return None
    m = re.search(r"(?<![\d.])(?:1(?:\.0+)?|0(?:\.\d+)?)(?![\d.])", text)
    if m is None:
        return None
    value = float(m.group(0))
    return value if 0.0 <= value <= 1.0 else None


def _first_token_top_logprobs(response) -> list[tuple[str, float]]:
    """Extract ``(token, logprob)`` pairs for the first generated token, defensively.

    Returns an empty list when the response carries no logprobs (server misconfiguration, empty completion)
    — the caller treats that as "Yes/No not found" and falls back.
    """
    try:
        content = response.choices[0].logprobs.content
    except (AttributeError, IndexError):
        return []
    if not content:
        return []
    first = content[0]
    pairs = [(entry.token, entry.logprob) for entry in (first.top_logprobs or [])]
    # The sampled token itself is normally repeated inside top_logprobs, but not on every
    # server version — include it defensively (duplicates are harmless for a max/min scan).
    pairs.append((first.token, first.logprob))
    return pairs


class LLMPredictor:
    """Async client wrapper: one prompt in, one :class:`ProbabilityResult` out.

    Requests are bounded by a semaphore (``max_concurrency``) and retried with exponential backoff on
    transient transport errors.  ``temperature=0`` and a fixed ``seed`` by default.  Counters (``n_requests``,
    ``n_guided_fallbacks``, ``n_parse_failures``) are exposed for per-run logging; asyncio's single-threaded
    scheduling makes bare-int increments safe.

    A pre-built ``client`` can be injected for testing; otherwise an ``AsyncOpenAI`` client is constructed
    against ``base_url`` with the SDK's own retries disabled (this class owns retry policy so backoff and
    logging live in one place).
    """

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        model: str,
        method: str = "logprob",
        temperature: float = 0.0,
        seed: int = 0,
        max_concurrency: int = 16,
        max_retries: int = 5,
        request_timeout: float = 120.0,
        top_logprobs: int = 20,
        fallback_prob: float = 0.5,
        extra_body: dict | None = None,
        client: AsyncOpenAI | None = None,
    ) -> None:
        if method not in ("logprob", "guided"):
            raise ValueError(f"method must be 'logprob' or 'guided', got {method!r}")
        self.model = model
        self.method = method
        self.temperature = temperature
        self.seed = seed
        self.max_retries = max_retries
        self.top_logprobs = top_logprobs
        self.fallback_prob = fallback_prob
        self.extra_body = extra_body
        self._client = client or AsyncOpenAI(
            base_url=base_url, api_key=api_key, timeout=request_timeout, max_retries=0
        )
        self._semaphore = asyncio.Semaphore(max_concurrency)

        self.n_requests = 0
        self.n_guided_fallbacks = 0
        self.n_parse_failures = 0

    async def _create(self, **kwargs):
        """One chat-completion call with exponential-backoff retries on transient errors."""
        if self.extra_body:
            # Per-call extras (e.g. guided_regex) win over instance-level ones.
            kwargs["extra_body"] = {**self.extra_body, **kwargs.get("extra_body", {})}
        for attempt in range(self.max_retries + 1):
            try:
                self.n_requests += 1
                return await self._client.chat.completions.create(
                    model=self.model, temperature=self.temperature, seed=self.seed, **kwargs
                )
            except _RETRYABLE_ERRORS as e:
                if attempt == self.max_retries:
                    raise
                delay = min(_BACKOFF_BASE_SECONDS * 2**attempt, _BACKOFF_CAP_SECONDS)
                logger.warning(
                    f"Transient LLM request error ({type(e).__name__}); "
                    f"retry {attempt + 1}/{self.max_retries} in {delay:.1f}s"
                )
                await asyncio.sleep(delay)

    def _messages(self, history_text: str, question: str, method: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(history_text, question, method)},
        ]

    async def _logprob_attempt(self, history_text: str, question: str) -> float | None:
        response = await self._create(
            messages=self._messages(history_text, question, "logprob"),
            max_tokens=1,
            logprobs=True,
            top_logprobs=self.top_logprobs,
        )
        return yes_no_probability(_first_token_top_logprobs(response))

    async def _guided_attempt(self, history_text: str, question: str) -> float | None:
        response = await self._create(
            messages=self._messages(history_text, question, "guided"),
            max_tokens=8,
            extra_body={"guided_regex": GUIDED_REGEX},
        )
        try:
            text = response.choices[0].message.content
        except (AttributeError, IndexError):
            text = None
        return parse_guided_probability(text)

    async def predict_prob(
        self, history_text: str, question: str, fallback_prob: float | None = None
    ) -> ProbabilityResult:
        """Extract one probability for ``(history_text, question)``.

        Runs the configured primary method; the ``logprob`` method falls back to ``guided``
        when neither Yes nor No is found in the top-k.  On total parse failure, returns
        ``fallback_prob`` (per-row override, e.g. the task's marginal prevalence) or the
        instance-level default, flagged ``parse_failed=True``.

        Raises:
            openai.APIError: on transport failure after retries are exhausted — see module
                docstring for why transport errors are never converted into fallbacks.
        """

        async with self._semaphore:
            if self.method == "logprob":
                prob = await self._logprob_attempt(history_text, question)
                if prob is not None:
                    return ProbabilityResult(prob, parse_failed=False, method_used="logprob")
                self.n_guided_fallbacks += 1

            prob = await self._guided_attempt(history_text, question)
            if prob is not None:
                method_used = "guided" if self.method == "guided" else "guided_fallback"
                return ProbabilityResult(prob, parse_failed=False, method_used=method_used)

            self.n_parse_failures += 1
            fb = self.fallback_prob if fallback_prob is None else fallback_prob
            return ProbabilityResult(fb, parse_failed=True, method_used="fallback")
