"""LLM-baseline inference entry point — ``EQ_llm_predict``.

A drop-in alternative to ``EQ_predict``: takes a directory of task-query parquets
(:class:`TaskQuerySchema`), serializes each patient's event history as text, asks a
locally-served LLM (OpenAI-compatible API, e.g. vLLM) for a per-row occurrence probability,
and writes :class:`PredictionSchema`-conformant parquet shards consumable by ``EQ_evaluate``
unchanged.

Differences from ``EQ_predict`` worth knowing about:

- **No model run dir.**  There is no trained checkpoint; the dataset config is built
  directly from ``tensorized_cohort_dir`` (which ``EQ_predict`` recovers from the training
  run's resolved config).
- **Two prompt styles.**  ``prompt_style=event_stream`` (default) renders EveryQuery's own
  chronological code list with a Yes/No question; ``prompt_style=llm4healthcare`` reproduces
  the feature-major panel prompt of https://github.com/yhzhu99/llm4healthcare and parses a
  float probability.  Both implement
  :class:`~every_query.llm_baseline.prompt_style.PromptStyle`, so everything below is shared.
  Note the panel style places the query in the task-description paragraph near the *top* of
  the prompt (that is where the reference template puts it), so rows in the same group no
  longer share a token prefix and vLLM's automatic prefix caching cannot reuse the patient
  block across a group's ~10-20 queries.  The event-stream style appends the query last and
  does keep that prefix shared.
- **Query-axis deduplication.**  A single ``(subject, prediction_time)`` commonly carries
  many ``(query, duration_days)`` pairs; the history is serialized exactly once per group
  and reused across that group's queries.  Each query is still its own LLM request (one
  ``n``-way sampled completion → one probability).
- **Raw histories, not model inputs.**  ``EveryQueryPytorchDataset._seeded_getitem``
  prepends the query's vocab token to the sequence (a model-input convention); the baseline
  bypasses it via the base-class ``load_subject_data``, which also yields the full
  untruncated pre-prediction-time history so serialization-side truncation counts are exact.
- **Sharded, resumable output.**  A full 70B run will die partway through; completed rows
  are flushed to ``output_dir/shards/`` as they finish (atomic tmp-rename per shard) and
  ``resume=true`` anti-joins already-written rows out of the pending set — after first
  verifying every existing shard's config fingerprint matches the current run (a resume may
  never silently mix models/prompts/configs).  On completion, shards are merged into
  ``output_dir/predictions.parquet``.
- **``censor_prob`` is a constant 0.0.**  Censoring is a data artifact, not a clinical
  prediction, and the LLM is not asked about it.  Downstream this only touches
  ``EQ_evaluate``'s ``censor_auroc`` sanity metric, which degrades to a meaningless-but-
  well-defined 0.5 for constant scores; ``occurs_auroc`` (the headline metric) drops
  censored rows before scoring and is unaffected.
- **Sidecar diagnostics.**  ``PredictionSchema.align`` rejects extra columns, so per-row
  ``parse_failed`` and the sample counts (``n_yes`` / ``n_no`` / ``n_unparsed`` for the
  Yes/No vote, ``n_parsed`` for float extraction) live in ``output_dir/details.parquet``
  (and ``details_shards/``), keyed by the same identifier columns.

Reproducibility: every shard and the merged parquet carry a ``every_query_llm_baseline``
key in their parquet metadata holding a JSON blob with the model name/revision, sampling
config (``n_samples`` / ``temperature`` / ``smoothing`` / ``guided_choice``), the prompt
style with its template hash and style-specific knobs (``style_fields`` — ``max_events`` and
code descriptions for ``event_stream``; panel digest, form and imputation for
``llm4healthcare``), seed, and a hash of the resolved config — a run is reconstructable from
its output alone, and ``resume`` refuses to extend shards whose fingerprint differs.
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path

import hydra
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from meds_torchdata.config import MEDSTorchDataConfig
from omegaconf import DictConfig, OmegaConf

from every_query.data.dataset import EveryQueryPytorchDataset
from every_query.data.schema import TaskQuerySchema
from every_query.llm_baseline.llm_predict import LLMPredictor, SampleResult
from every_query.llm_baseline.prompt_style import PromptStyle, build_style
from every_query.llm_baseline.serialize import SerializedHistory, load_code_descriptions
from every_query.predict.predict import _identifiers_from_schema_df, _validate_tasks_dir
from every_query.predict.schema import PredictionSchema

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

CONFIGS = str(files("every_query") / "llm_baseline" / "configs")

METADATA_KEY = b"every_query_llm_baseline"

_ID_COLS = [
    TaskQuerySchema.subject_id_name,
    TaskQuerySchema.prediction_time_name,
    TaskQuerySchema.query_name,
    TaskQuerySchema.duration_days_name,
]


def _config_hash(cfg: DictConfig) -> str:
    """Short stable fingerprint of the resolved run config."""
    return hashlib.sha256(OmegaConf.to_yaml(cfg, resolve=True).encode()).hexdigest()[:16]


# Metadata fields that determine per-row predictions.  ``resume`` refuses to extend shards
# whose fingerprint over these fields differs from the current run's — identifier columns
# alone can't tell a Qwen shard from a Llama shard.
# ``style_fields`` holds the style-specific knobs (max_events / panel / form / …), which vary
# by prompt_style and so cannot be enumerated statically; it is fingerprinted as a whole.
_FINGERPRINT_FIELDS = (
    "model",
    "model_revision",
    "prompt_style",
    "prompt_template_version",
    "prompt_template_hash",
    "response_format",
    "style_fields",
    "n_samples",
    "temperature",
    "smoothing",
    "guided_choice",
    "seed",
    "fallback_prob",
    "extra_body",
)


def _run_payload(cfg: DictConfig, style: PromptStyle) -> dict:
    """Build the reproducibility payload recorded (as JSON) on every output file."""
    payload = {
        "model": cfg.model,
        "model_revision": cfg.model_revision,
        "prompt_style": style.name,
        "prompt_template_version": style.template_version,
        "prompt_template_hash": style.template_hash(),
        "response_format": style.response_format,
        "style_fields": style.payload_fields(),
        "n_samples": int(cfg.n_samples),
        "temperature": float(cfg.temperature),
        "smoothing": float(cfg.smoothing),
        "guided_choice": bool(cfg.guided_choice),
        "seed": int(cfg.seed),
        "fallback_prob": None if cfg.fallback_prob is None else float(cfg.fallback_prob),
        "extra_body": None if cfg.extra_body is None else OmegaConf.to_container(cfg.extra_body),
    }
    fingerprint_subset = {k: payload[k] for k in _FINGERPRINT_FIELDS}
    payload["run_fingerprint"] = hashlib.sha256(
        json.dumps(fingerprint_subset, sort_keys=True).encode()
    ).hexdigest()[:16]
    payload["config_hash"] = _config_hash(cfg)
    payload["created_at"] = datetime.now(UTC).isoformat()
    return payload


def _run_metadata(payload: dict) -> dict[bytes, bytes]:
    """Wrap the reproducibility payload as parquet key-value metadata."""
    return {METADATA_KEY: json.dumps(payload).encode()}


def _validate_resume_fingerprint(shards_dir: Path, payload: dict) -> None:
    """Refuse to resume onto shards produced under a different prediction-affecting config.

    Compares the ``run_fingerprint`` recorded in each existing shard's metadata against the
    current run's; on mismatch, raises with the differing fields spelled out.
    """
    for fp in sorted(shards_dir.glob("*.parquet")):
        blob = (pq.read_schema(fp).metadata or {}).get(METADATA_KEY)
        existing = json.loads(blob) if blob else {}
        if existing.get("run_fingerprint") == payload["run_fingerprint"]:
            continue
        diffs = {
            k: {"shard": existing.get(k), "current": payload.get(k)}
            for k in _FINGERPRINT_FIELDS
            if existing.get(k) != payload.get(k)
        }
        raise ValueError(
            f"resume=true, but existing shard {fp.name} was produced under a different "
            f"configuration — refusing to silently mix predictions.  Differing fields: "
            f"{json.dumps(diffs, indent=2)}\nPoint output_dir at a new path, or rerun with "
            f"the original configuration."
        )


@dataclass
class _HistorySerializer:
    """Serializes one ``(subject_id, prediction_time)`` group's history per call.

    Wraps the dataset's base-class ``load_subject_data`` (full untruncated pre-prediction window, no prepended
    query token) and hands the dense arrays to the configured :class:`PromptStyle`.  Callers are responsible
    for calling once per group and sharing the returned string across that group's ``(query, duration_days)``
    rows — no cache is kept here, so serialized histories are released as soon as the caller drops them
    (memory stays bounded on large cohorts).  Tracks truncation and empty-history stats for the per-run log
    line; under a panel style an "empty" history means the window held no measurement of any panel feature.
    """

    dataset: EveryQueryPytorchDataset
    style: PromptStyle

    def __post_init__(self) -> None:
        self._index_to_code = {idx: code for code, idx in self.dataset.code_to_index.items()}
        self.n_serialized = 0
        self.n_truncated = 0
        self.n_empty = 0

    def history_for_row(self, row_idx: int) -> SerializedHistory:
        """Serialize the history for dataset row ``row_idx`` (call once per group)."""
        subject_id, end_idx = self.dataset.index[row_idx]
        dynamic_data, static = self.dataset.load_subject_data(subject_id=subject_id, st=0, end=end_idx)
        static_codes = list(static.code) if static is not None else None
        history = self.style.history_from_dense(dynamic_data.to_dense(), static_codes, self._index_to_code)
        self.n_serialized += 1
        self.n_truncated += int(history.truncated)
        self.n_empty += int(history.n_events_total == 0)
        return history

    def log_truncation_stats(self) -> None:
        if self.n_serialized == 0:
            return
        logger.info(
            f"Serialized {self.n_serialized} unique (subject, prediction_time) histories; "
            f"{self.n_truncated} ({self.n_truncated / self.n_serialized:.1%}) were truncated."
        )
        if self.n_empty:
            logger.warning(
                f"{self.n_empty} of {self.n_serialized} histories "
                f"({self.n_empty / self.n_serialized:.1%}) serialized to an empty record — under "
                f"the 'llm4healthcare' style this means the window contained no measurement of "
                f"any panel feature, so the model is asked to predict from demographics alone."
            )


class _ShardedWriter:
    """Flushes completed rows to ``shards/`` (predictions) + ``details_shards/`` (sidecar).

    Each flush writes one parquet per directory via a ``.tmp`` sibling + atomic rename, so a crash mid-write
    never leaves a half-written ``.parquet`` to corrupt a later ``resume`` scan.  Prediction shards are
    ``PredictionSchema.align``-ed and carry the run-metadata blob; the sidecar carries the same identifier
    columns plus ``occurs_prob``, ``parse_failed``, and the Yes/No/unparsed vote counts.
    """

    def __init__(
        self,
        identifiers: pl.DataFrame,
        shards_dir: Path,
        details_dir: Path,
        metadata: dict[bytes, bytes],
        run_id: str,
    ) -> None:
        self._identifiers = identifiers
        self._shards_dir = shards_dir
        self._details_dir = details_dir
        self._metadata = metadata
        self._run_id = run_id
        self._n_shards = 0
        self.n_rows_written = 0
        shards_dir.mkdir(parents=True, exist_ok=True)
        details_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _atomic_write(table: pa.Table, target: Path) -> None:
        # Dot-prefixed staging name: pyarrow dataset discovery skips '.'-prefixed files and
        # the resume scan globs '*.parquet', so a leftover from a hard kill mid-write can
        # neither be swept into the final merge nor mistaken for a completed shard.
        tmp = target.with_name("." + target.name + ".tmp")
        pq.write_table(table, tmp)
        tmp.replace(target)

    def flush(self, results: list[tuple[int, SampleResult]]) -> None:
        """Write one prediction shard + one details shard for ``(row_idx, result)`` pairs."""
        if not results:
            return
        row_idxs = [row_idx for row_idx, _ in results]
        ids = self._identifiers.select(pl.all().gather(row_idxs))
        probs = pl.DataFrame(
            {
                PredictionSchema.censor_prob_name: pl.Series([0.0] * len(results), dtype=pl.Float32),
                PredictionSchema.occurs_prob_name: pl.Series(
                    [res.prob for _, res in results], dtype=pl.Float32
                ),
            }
        )
        aligned = PredictionSchema.align(ids.hstack(probs).to_arrow())
        aligned = aligned.replace_schema_metadata(self._metadata)

        details = ids.select(_ID_COLS).hstack(
            pl.DataFrame(
                {
                    PredictionSchema.occurs_prob_name: pl.Series(
                        [res.prob for _, res in results], dtype=pl.Float32
                    ),
                    "parse_failed": [res.parse_failed for _, res in results],
                    "n_yes": [res.n_yes for _, res in results],
                    "n_no": [res.n_no for _, res in results],
                    "n_unparsed": [res.n_unparsed for _, res in results],
                    "n_parsed": [res.n_parsed for _, res in results],
                }
            )
        )

        shard_name = f"shard_{self._run_id}_{self._n_shards:05d}.parquet"
        # Details first: resume completeness is keyed on prediction shards, so the shard the
        # resume scan trusts must be the last artifact published — a crash between the two
        # writes then re-runs the batch instead of leaving a permanent sidecar hole.
        self._atomic_write(
            details.to_arrow().replace_schema_metadata(self._metadata), self._details_dir / shard_name
        )
        self._atomic_write(aligned, self._shards_dir / shard_name)
        self._n_shards += 1
        self.n_rows_written += len(results)
        logger.info(f"Flushed shard {shard_name} ({len(results)} rows; {self.n_rows_written} total)")


def _completed_keys(shards_dir: Path) -> pl.DataFrame | None:
    """Read identifier columns of every completed shard, or ``None`` when there are none."""
    shard_fps = sorted(shards_dir.glob("*.parquet"))
    if not shard_fps:
        return None
    return pl.concat([pl.from_arrow(pq.read_table(fp, columns=_ID_COLS)) for fp in shard_fps], how="vertical")


def _pending_row_idxs(identifiers: pl.DataFrame, shards_dir: Path, resume: bool) -> list[int]:
    """Row indices still to run; under ``resume``, anti-join out rows already in shards."""
    with_idx = identifiers.select(_ID_COLS).with_row_index("__row_idx")
    if resume:
        done = _completed_keys(shards_dir)
        if done is not None:
            before = with_idx.height
            with_idx = with_idx.join(done, on=_ID_COLS, how="anti")
            logger.info(f"Resume: skipping {before - with_idx.height} already-completed rows")
    return sorted(with_idx["__row_idx"].to_list())


def _merge_shards(shards_dir: Path, out_fp: Path, metadata: dict[bytes, bytes]) -> int:
    """Concatenate all shards into a single parquet at ``out_fp``; returns the row count.

    Reads only ``*.parquet`` files — never directory-level dataset discovery, which would
    sweep in leftover staging files from a hard-killed run.
    """
    tables = [pq.read_table(fp) for fp in sorted(shards_dir.glob("*.parquet"))]
    merged = pa.concat_tables(tables).replace_schema_metadata(metadata)
    tmp = out_fp.with_name(out_fp.name + ".tmp")
    pq.write_table(merged, tmp)
    tmp.replace(out_fp)
    return merged.num_rows


def _print_dry_run_prompts(
    serializer: _HistorySerializer,
    style: PromptStyle,
    identifiers: pl.DataFrame,
    pending: list[int],
    n_prompts: int = 3,
    n_stats_groups: int = 200,
) -> None:
    """Print prompts for the first few distinct histories; serialize more to sample the stats.

    Printing stops at ``n_prompts``, but serialization continues to ``n_stats_groups`` so the
    caller's truncation / empty-record rates are measured over a sample rather than over the
    three histories that happened to print.  A single empty window says nothing about whether
    the panel matches the cohort; 200 of them do.  Bounded rather than sweeping all of
    ``pending`` so a dry run over a multi-million-row cohort still returns in seconds.
    """
    seen_groups: set[tuple[int, datetime]] = set()
    shown = 0
    for row_idx in pending:
        if len(seen_groups) >= n_stats_groups:
            break
        subject_id = identifiers[TaskQuerySchema.subject_id_name][row_idx]
        prediction_time = identifiers[TaskQuerySchema.prediction_time_name][row_idx]
        if (subject_id, prediction_time) in seen_groups:
            continue
        seen_groups.add((subject_id, prediction_time))

        history = serializer.history_for_row(row_idx)
        if shown >= n_prompts:
            continue
        query_code = identifiers[TaskQuerySchema.query_name][row_idx]
        duration = float(identifiers[TaskQuerySchema.duration_days_name][row_idx])
        print(
            f"\n{'=' * 80}\n--- prompt {shown + 1} (subject={subject_id}, "
            f"prediction_time={prediction_time}, style={style.name}) ---"
        )
        print(f"[system]\n{style.system_prompt()}\n")
        print(f"[user]\n{style.user_prompt(history.text, query_code, duration)}")
        shown += 1
    print(
        f"\n{'=' * 80}\nDry run: {shown} prompt(s) shown over {serializer.n_serialized} "
        f"serialized histories; no requests were sent."
    )
    serializer.log_truncation_stats()


def _group_pending_rows(identifiers: pl.DataFrame, pending: list[int]) -> list[list[int]]:
    """Group pending row indices by ``(subject_id, prediction_time)``, preserving row order."""
    groups: dict[tuple, list[int]] = {}
    subject_ids = identifiers[TaskQuerySchema.subject_id_name]
    prediction_times = identifiers[TaskQuerySchema.prediction_time_name]
    for row_idx in pending:
        groups.setdefault((subject_ids[row_idx], prediction_times[row_idx]), []).append(row_idx)
    return list(groups.values())


async def _run_async(
    cfg: DictConfig,
    predictor: LLMPredictor,
    serializer: _HistorySerializer,
    identifiers: pl.DataFrame,
    pending: list[int],
    writer: _ShardedWriter,
) -> None:
    """Stream pending rows through a bounded queue of LLM workers, flushing shards as they finish.

    The producer serializes each ``(subject_id, prediction_time)`` group's history exactly once
    and enqueues that group's rows sharing the one string; the bounded queue then backpressures
    the producer, so peak memory scales with ``max_concurrency`` + the queue bound + one flush
    buffer — never with the cohort size.
    """
    n_workers = int(cfg.max_concurrency)
    shard_size = int(cfg.batch_size)
    queue: asyncio.Queue[tuple[int, str] | None] = asyncio.Queue(maxsize=max(2 * n_workers, 32))
    buffer: list[tuple[int, SampleResult]] = []

    async def producer() -> None:
        for row_idxs in _group_pending_rows(identifiers, pending):
            history = serializer.history_for_row(row_idxs[0])
            for row_idx in row_idxs:
                await queue.put((row_idx, history.text))
        for _ in range(n_workers):
            await queue.put(None)

    async def worker() -> None:
        while (item := await queue.get()) is not None:
            row_idx, history_text = item
            query = identifiers[TaskQuerySchema.query_name][row_idx]
            duration = float(identifiers[TaskQuerySchema.duration_days_name][row_idx])
            result = await predictor.predict_prob(history_text, query, duration)
            buffer.append((row_idx, result))
            if len(buffer) >= shard_size:
                writer.flush(buffer)
                buffer.clear()

    tasks = [asyncio.create_task(producer())] + [asyncio.create_task(worker()) for _ in range(n_workers)]
    try:
        await asyncio.gather(*tasks)
    except BaseException:
        # A transport failure (or Ctrl-C) aborts the run; flush what already completed so
        # resume=true can pick up from here, then cancel the still-running remainder.
        writer.flush(buffer)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    writer.flush(buffer)
    serializer.log_truncation_stats()


@hydra.main(version_base="1.3", config_path=CONFIGS, config_name="llm_predict")
def main(cfg: DictConfig) -> None:
    """Serialize task histories, query the LLM, write PredictionSchema parquet shards."""
    tasks_dir = Path(cfg.tasks_dir)
    output_dir = Path(cfg.output_dir)
    shards_dir = output_dir / "shards"
    details_dir = output_dir / "details_shards"

    _validate_tasks_dir(tasks_dir)
    if shards_dir.exists() and any(shards_dir.glob("*.parquet")) and not (cfg.resume or cfg.dry_run):
        raise FileExistsError(
            f"{shards_dir} already contains prediction shards.  Pass resume=true to continue an "
            f"interrupted run, or point output_dir at a new path — EQ_llm_predict refuses to "
            f"silently mix runs."
        )

    code_descriptions = load_code_descriptions(Path(cfg.code_descriptions)) if cfg.code_descriptions else None
    if code_descriptions is not None:
        logger.info(f"Loaded {len(code_descriptions)} code descriptions from {cfg.code_descriptions}")

    # 'include' rather than 'omit': the llm4healthcare style renders the patient's sex from the
    # GENDER//* static code.  The extra per-subject read is one short list column and the
    # event_stream style simply ignores the returned statics.
    dataset_cfg = MEDSTorchDataConfig(
        tensorized_cohort_dir=str(cfg.tensorized_cohort_dir),
        task_labels_dir=str(tasks_dir),
        max_seq_len=int(cfg.max_events),  # unused by our load_subject_data path, but required
        seq_sampling_strategy="to_end",
        static_inclusion_mode="include",
        batch_mode="SM",
    )
    dataset = EveryQueryPytorchDataset(dataset_cfg, split=cfg.split)
    identifiers = _identifiers_from_schema_df(dataset.schema_df)
    logger.info(
        f"Loaded {identifiers.height} task rows across "
        f"{identifiers[TaskQuerySchema.query_name].n_unique()} query codes"
    )

    pending = _pending_row_idxs(identifiers, shards_dir, resume=bool(cfg.resume))
    if cfg.limit is not None:
        pending = pending[: int(cfg.limit)]
        logger.info(f"limit={cfg.limit}: capped to {len(pending)} pending rows")

    n_groups = (
        identifiers.select(pl.all().gather(pending))
        .select(TaskQuerySchema.subject_id_name, TaskQuerySchema.prediction_time_name)
        .unique()
        .height
    )
    if n_groups:
        logger.info(
            f"{len(pending)} rows pending over {n_groups} unique (subject, prediction_time) "
            f"histories ({len(pending) / n_groups:.1f} queries per serialized history)"
        )
    else:
        logger.info("0 rows pending — nothing to do")

    style = build_style(cfg, code_descriptions)
    serializer = _HistorySerializer(dataset=dataset, style=style)

    if cfg.dry_run:
        _print_dry_run_prompts(serializer, style, identifiers, pending)
        return

    payload = _run_payload(cfg, style)
    metadata = _run_metadata(payload)
    if cfg.resume:
        _validate_resume_fingerprint(shards_dir, payload)
    predictor = LLMPredictor(
        style=style,
        base_url=cfg.base_url,
        api_key=cfg.api_key,
        model=cfg.model,
        n_samples=int(cfg.n_samples),
        temperature=float(cfg.temperature),
        smoothing=float(cfg.smoothing),
        guided_choice=bool(cfg.guided_choice),
        seed=int(cfg.seed),
        max_concurrency=int(cfg.max_concurrency),
        max_retries=int(cfg.max_retries),
        request_timeout=float(cfg.request_timeout),
        fallback_prob=0.5 if cfg.fallback_prob is None else float(cfg.fallback_prob),
        extra_body=None if cfg.extra_body is None else OmegaConf.to_container(cfg.extra_body),
    )

    # Microsecond precision: two runs into the same output_dir (e.g. an immediate resume
    # after a fast partial run) must never collide on shard filenames.
    run_id = datetime.now(UTC).strftime("%Y%m%d%H%M%S%f")
    writer = _ShardedWriter(identifiers, shards_dir, details_dir, metadata, run_id)

    t0 = time.perf_counter()
    asyncio.run(_run_async(cfg, predictor, serializer, identifiers, pending, writer))
    elapsed = time.perf_counter() - t0

    if predictor.n_parse_failures:
        rate = predictor.n_parse_failures / max(len(pending), 1)
        logger.warning(
            f"{predictor.n_parse_failures} rows ({rate:.1%}) hit total parse failure and were "
            f"written with the fallback probability (parse_failed=True in details.parquet)."
        )
    logger.info(
        f"Completed {writer.n_rows_written} rows in {elapsed:.1f}s "
        f"({predictor.n_requests} requests, {predictor.n_parse_failures} parse failures, "
        f"{predictor.n_unparsed_samples} unparsed samples)"
    )

    n_merged = _merge_shards(shards_dir, output_dir / "predictions.parquet", metadata)
    n_details = _merge_shards(details_dir, output_dir / "details.parquet", metadata)
    if n_details != n_merged:
        logger.warning(
            f"details.parquet has {n_details} rows but predictions.parquet has {n_merged} — "
            f"a crash between shard writes on an earlier run likely left extra details rows; "
            f"diagnostics only, predictions are unaffected."
        )
    logger.info(f"Merged {n_merged} predictions to {output_dir / 'predictions.parquet'}")


if __name__ == "__main__":
    main()
