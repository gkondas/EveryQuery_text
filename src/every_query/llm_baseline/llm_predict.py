"""Empirical Yes/No probability extraction from an LLM over an OpenAI-compatible API (vLLM).

This is the crux of the baseline.  Free-text "output a probability between 0 and 1" mode-
collapses to a handful of round numbers (0.1 / 0.5 / 0.9), and token logprobs are not
exposed by every serving stack (the hosted frontier chat APIs, notably).  Instead we
constrain the model to a one-word ``Yes`` / ``No`` answer, draw ``n_samples`` samples at a
non-zero temperature, and use the **empirical fraction of Yes answers** as the occurrence
probability.  With ``n`` samples the score takes one of ``n + 1`` discrete values, so more
samples buy finer resolution and lower Monte-Carlo variance at linear decode cost.  The
whole ``n``-way completion is a single request, so the (long) patient-history prefix is
prefilled once and shared across all samples.

On vLLM the answer is additionally constrained with ``guided_choice=["Yes", "No"]`` so every
sample is a clean Yes or No and there is nothing to parse-fail on; disable ``guided_choice``
for servers without that extension (the leading Yes/No word is then parsed leniently).

Failure semantics are explicit, never silent:

- **Parse failure** (no sample yielded a usable Yes/No answer): the configured fallback
  probability is written, the row is flagged ``parse_failed=True`` in the sidecar details
  parquet, and the per-run failure rate is logged.
- **Transport failure** (connection / timeout / 429 / 5xx after retries with exponential
  backoff): the run *raises*.  Writing a fallback for transport errors would silently poison
  the output; the CLI's sharded output + ``resume=true`` handles restarts instead.
"""

import asyncio
import logging
import re
from dataclasses import dataclass

from openai import APIConnectionError, AsyncOpenAI, InternalServerError, RateLimitError

logger = logging.getLogger(__name__)

# vLLM structured-output constraint: force each sampled answer to be exactly one of these.
YES_NO_CHOICES = ("Yes", "No")
# Max tokens per sampled answer — "Yes"/"No" is 1-2 tokens; a small cap keeps sampling cheap.
_ANSWER_MAX_TOKENS = 8
# Float answers need room for "0.85" plus any stray leading whitespace/newline.
_FLOAT_MAX_TOKENS = 16

# Leading Yes/No word, case-insensitive, tolerating trailing punctuation / text.
_YES_NO_RE = re.compile(r"\s*(yes|no)\b", re.IGNORECASE)
# First float-looking token anywhere in the answer.
_FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

# APITimeoutError subclasses APIConnectionError, so timeouts are covered by the first entry.
_RETRYABLE_ERRORS = (APIConnectionError, RateLimitError, InternalServerError)
_BACKOFF_BASE_SECONDS = 1.0
_BACKOFF_CAP_SECONDS = 30.0


@dataclass(frozen=True)
class SampleResult:
    """One row's empirical Yes-probability plus the vote breakdown that produced it.

    ``prob`` is the (optionally Laplace-smoothed) fraction of Yes votes among the parseable
    samples, or the configured fallback probability when ``parse_failed`` is True (no sample
    produced a usable Yes/No answer).  ``n_yes`` / ``n_no`` / ``n_unparsed`` sum to the number
    of samples requested and are recorded in the details sidecar for per-row Monte-Carlo error
    bars.
    """

    prob: float
    n_yes: int
    n_no: int
    n_unparsed: int
    parse_failed: bool
    #: Float-extraction mode only: how many samples yielded a usable probability.  Always 0
    #: under the Yes/No vote, where ``n_yes + n_no`` carries the same information.
    n_parsed: int = 0


def parse_yes_no(text: str | None) -> str | None:
    """Parse a sampled answer into ``"yes"`` / ``"no"`` / ``None`` (unparseable).

    Matches the leading word case-insensitively, tolerating trailing punctuation or text
    ("Yes.", "No, because …") that a non-guided server may emit.

    Examples:
        >>> [parse_yes_no(t) for t in ["Yes", " no", "Yes.", "No, because ...", "YES\\n"]]
        ['yes', 'no', 'yes', 'no', 'yes']
        >>> parse_yes_no("Maybe") is None
        True
        >>> parse_yes_no("") is None and parse_yes_no(None) is None
        True
    """
    if not text:
        return None
    m = _YES_NO_RE.match(text)
    return m.group(1).lower() if m else None


def empirical_probability(n_yes: int, n_no: int, smoothing: float = 0.0) -> float | None:
    """Empirical P(Yes) from the vote counts, or ``None`` when there are no parseable votes.

    With ``smoothing == 0`` this is the raw fraction ``n_yes / (n_yes + n_no)``.  A positive
    ``smoothing`` applies a symmetric Laplace prior ``(n_yes + s) / (n_yes + n_no + 2s)``,
    keeping the score off the exact 0/1 endpoints — irrelevant for AUROC (the transform is
    monotonic, so the ranking and AUC are unchanged) but keeping log-loss / Brier finite if
    those are computed downstream.

    Examples:
        >>> empirical_probability(3, 1)
        0.75
        >>> empirical_probability(4, 0)
        1.0
        >>> empirical_probability(4, 0, smoothing=1.0)
        0.8333333333333334
        >>> empirical_probability(0, 0) is None
        True
    """
    total = n_yes + n_no
    if total == 0:
        return None
    return (n_yes + smoothing) / (total + 2.0 * smoothing)


def parse_probability(text: str | None) -> float | None:
    """Parse a free-text probability answer into ``[0, 1]``, or ``None`` if unusable.

    Takes the first float-looking token anywhere in the answer and clamps it to the unit
    interval.  The reference implementation this mode replicates calls bare ``float(result)``
    and falls back on any exception; scanning for the first number is strictly more permissive
    (it survives a stray newline or a trailing period) without changing the prompt, so fewer
    usable answers are thrown away.

    Examples:
        >>> [parse_probability(t) for t in ["0.85", " 0.3\\n", "0.75.", "Probability: 0.2"]]
        [0.85, 0.3, 0.75, 0.2]
        >>> parse_probability("1.4"), parse_probability("-0.2")
        (1.0, 0.0)
        >>> parse_probability("I do not know") is None
        True
        >>> parse_probability("") is None and parse_probability(None) is None
        True
    """
    if not text:
        return None
    m = _FLOAT_RE.search(text)
    if not m:
        return None
    try:
        return min(1.0, max(0.0, float(m.group())))
    except ValueError:
        return None


class LLMPredictor:
    """Async client wrapper: one prompt in, one :class:`SampleResult` out.

    Draws ``n_samples`` Yes/No samples per query (a single ``n``-way completion so the history
    prefix is prefilled once) and returns the empirical Yes-fraction.  Requests are bounded by
    a semaphore (``max_concurrency``) and retried with exponential backoff on transient
    transport errors.  Counters (``n_requests``, ``n_parse_failures``, ``n_unparsed_samples``)
    are exposed for per-run logging; asyncio's single-threaded scheduling makes bare-int
    increments safe.

    A pre-built ``client`` can be injected for testing; otherwise an ``AsyncOpenAI`` client is
    constructed against ``base_url`` with the SDK's own retries disabled (this class owns retry
    policy so backoff and logging live in one place).
    """

    def __init__(
        self,
        *,
        style,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        model: str,
        n_samples: int = 20,
        temperature: float = 0.7,
        smoothing: float = 0.0,
        guided_choice: bool = True,
        seed: int = 0,
        max_concurrency: int = 16,
        max_retries: int = 5,
        request_timeout: float = 120.0,
        fallback_prob: float = 0.5,
        extra_body: dict | None = None,
        client: AsyncOpenAI | None = None,
    ) -> None:
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")
        if n_samples > 1 and temperature <= 0.0:
            raise ValueError(
                f"n_samples={n_samples} with temperature={temperature}: repeated sampling at "
                f"temperature 0 returns identical samples, so every probability collapses to 0 "
                f"or 1.  Set temperature > 0 (e.g. 0.7) for repeated sampling."
            )
        self.style = style
        self.model = model
        self.n_samples = n_samples
        self.temperature = temperature
        self.smoothing = smoothing
        # Guided choice constrains the answer to exactly "Yes"/"No", which is meaningless (and
        # would corrupt the answer) when the style asks for a float.
        self.guided_choice = guided_choice and style.response_format == "yes_no"
        self.seed = seed
        self.max_retries = max_retries
        self.fallback_prob = fallback_prob
        self.extra_body = extra_body
        self._client = client or AsyncOpenAI(
            base_url=base_url, api_key=api_key, timeout=request_timeout, max_retries=0
        )
        self._semaphore = asyncio.Semaphore(max_concurrency)

        self.n_requests = 0
        self.n_parse_failures = 0
        self.n_unparsed_samples = 0

    async def _create(self, **kwargs):
        """One chat-completion call with exponential-backoff retries on transient errors."""
        if self.extra_body:
            # Per-call extras (e.g. guided_choice) win over instance-level ones.
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

    def _messages(self, history_text: str, query_code: str, duration_days: float) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.style.system_prompt()},
            {"role": "user", "content": self.style.user_prompt(history_text, query_code, duration_days)},
        ]

    async def _completions(
        self, history_text: str, query_code: str, duration_days: float
    ) -> list[str | None]:
        """Draw ``n_samples`` completions and return their raw text."""
        is_float = self.style.response_format == "float"
        kwargs: dict = {
            "messages": self._messages(history_text, query_code, duration_days),
            "max_tokens": _FLOAT_MAX_TOKENS if is_float else _ANSWER_MAX_TOKENS,
            "n": self.n_samples,
        }
        if self.guided_choice:
            kwargs["extra_body"] = {"guided_choice": list(YES_NO_CHOICES)}
        response = await self._create(**kwargs)

        contents: list[str | None] = []
        for choice in response.choices:
            try:
                contents.append(choice.message.content)
            except (AttributeError, IndexError):
                contents.append(None)
        return contents

    @staticmethod
    def _tally_yes_no(contents: list[str | None]) -> tuple[int, int, int]:
        """Tally ``(n_yes, n_no, n_unparsed)`` over sampled answers."""
        n_yes = n_no = n_unparsed = 0
        for content in contents:
            match parse_yes_no(content):
                case "yes":
                    n_yes += 1
                case "no":
                    n_no += 1
                case _:
                    n_unparsed += 1
        return n_yes, n_no, n_unparsed

    async def predict_prob(
        self,
        history_text: str,
        query_code: str,
        duration_days: float,
        fallback_prob: float | None = None,
    ) -> SampleResult:
        """Extract one occurrence probability for ``(history, query_code, duration_days)``.

        Under the Yes/No style, samples the model ``n_samples`` times and returns the
        (optionally smoothed) fraction of Yes answers.  Under the float style, parses a
        probability out of each sample and returns their mean — with ``n_samples=1`` that is
        exactly the reference implementation's single-shot ``float(result)``, and with more
        samples it is the same estimator with lower variance.

        On total parse failure — no sample produced a usable answer — returns ``fallback_prob``
        (per-row override, e.g. an externally-estimated prevalence) or the instance-level
        default, flagged ``parse_failed=True``.

        Raises:
            openai.APIError: on transport failure after retries are exhausted — see module
                docstring for why transport errors are never converted into fallbacks.
        """
        async with self._semaphore:
            contents = await self._completions(history_text, query_code, duration_days)
            fb = self.fallback_prob if fallback_prob is None else fallback_prob

            if self.style.response_format == "float":
                parsed = [p for p in (parse_probability(c) for c in contents) if p is not None]
                n_unparsed = len(contents) - len(parsed)
                self.n_unparsed_samples += n_unparsed
                if parsed:
                    return SampleResult(
                        sum(parsed) / len(parsed), 0, 0, n_unparsed, False, n_parsed=len(parsed)
                    )
                self.n_parse_failures += 1
                # A total float failure means no sample contained even a digit, which is almost
                # always the answer being empty or truncated rather than a bad number.  Nothing
                # else records the raw completion, so echo the first few — otherwise a run that
                # fails on every row gives you nothing to debug from.
                if self.n_parse_failures <= 3:
                    logger.warning(f"float parse failure, raw completions: {contents!r}")
                return SampleResult(fb, 0, 0, n_unparsed, parse_failed=True, n_parsed=0)

            n_yes, n_no, n_unparsed = self._tally_yes_no(contents)
            self.n_unparsed_samples += n_unparsed
            prob = empirical_probability(n_yes, n_no, self.smoothing)
            if prob is not None:
                return SampleResult(prob, n_yes, n_no, n_unparsed, parse_failed=False)

            self.n_parse_failures += 1
            return SampleResult(fb, n_yes, n_no, n_unparsed, parse_failed=True)
