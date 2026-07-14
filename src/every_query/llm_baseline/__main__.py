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
- **Query-axis deduplication.**  A single ``(subject, prediction_time)`` commonly carries
  many ``(query, duration_days)`` pairs; the history is serialized exactly once per group
  and reused across that group's queries.  Each query is still its own LLM request — one
  probability per request keeps the logprob method clean.
- **Raw histories, not model inputs.**  ``EveryQueryPytorchDataset._seeded_getitem``
  prepends the query's vocab token to the sequence (a model-input convention); the baseline
  bypasses it via the base-class ``load_subject_data``, which also yields the full
  untruncated pre-prediction-time history so serialization-side truncation counts are exact.
- **Sharded, resumable output.**  A full 70B run will die partway through; completed rows
  are flushed to ``output_dir/shards/`` as they finish (atomic tmp-rename per shard) and
  ``resume=true`` anti-joins already-written rows out of the pending set.  On completion,
  shards are merged into ``output_dir/predictions.parquet``.
- **``censor_prob`` is a constant 0.0.**  Censoring is a data artifact, not a clinical
  prediction, and the LLM is not asked about it.  Downstream this only touches
  ``EQ_evaluate``'s ``censor_auroc`` sanity metric, which degrades to a meaningless-but-
  well-defined 0.5 for constant scores; ``occurs_auroc`` (the headline metric) drops
  censored rows before scoring and is unaffected.
- **Sidecar diagnostics.**  ``PredictionSchema.align`` rejects extra columns, so per-row
  ``parse_failed`` / ``method_used`` live in ``output_dir/details.parquet`` (and
  ``details_shards/``), keyed by the same identifier columns.

Reproducibility: every shard and the merged parquet carry a ``every_query_llm_baseline``
key in their parquet metadata holding a JSON blob with the model name/revision, method,
prompt-template hash, ``max_events``, whether code descriptions were used, seed, and a hash
of the resolved config — a run is reconstructable from its output alone.
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
from every_query.llm_baseline.llm_predict import LLMPredictor, ProbabilityResult
from every_query.llm_baseline.serialize import (
    PROMPT_TEMPLATE_VERSION,
    SYSTEM_PROMPT,
    SerializedHistory,
    build_user_prompt,
    events_from_jnrt_dense,
    load_code_descriptions,
    prompt_template_hash,
    serialize_history,
    serialize_question,
)
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

# MTD schema_df column holding the timestamp of the last event inside each row's window
# (populated because we set ``include_window_last_observed_in_schema=True``); used to anchor
# serialized event times at the prediction time rather than at the last observed event.
_LAST_TIME_COL = "window_last_observed"


def _config_hash(cfg: DictConfig) -> str:
    """Short stable fingerprint of the resolved run config."""
    return hashlib.sha256(OmegaConf.to_yaml(cfg, resolve=True).encode()).hexdigest()[:16]


def _run_metadata(cfg: DictConfig) -> dict[bytes, bytes]:
    """Build the parquet key-value metadata blob recorded on every output file."""
    payload = {
        "model": cfg.model,
        "model_revision": cfg.model_revision,
        "method": cfg.method,
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "prompt_template_hash": prompt_template_hash(),
        "max_events": int(cfg.max_events),
        "code_descriptions_used": cfg.code_descriptions is not None,
        "code_descriptions_path": cfg.code_descriptions,
        "seed": int(cfg.seed),
        "temperature": float(cfg.temperature),
        "extra_body": None if cfg.extra_body is None else OmegaConf.to_container(cfg.extra_body),
        "config_hash": _config_hash(cfg),
        "created_at": datetime.now(UTC).isoformat(),
    }
    return {METADATA_KEY: json.dumps(payload).encode()}


def _prevalence_by_task(identifiers: pl.DataFrame) -> dict[tuple[str, float], float]:
    """Per-``(query, duration_days)`` marginal prevalence among non-censored rows.

    Used as the parse-failure fallback probability when ``fallback_prob`` is null.  Tasks
    whose rows are all censored (or unlabeled inputs) get no entry — those rows fall back to
    0.5 via the predictor default.

    Examples:
        >>> ids = pl.DataFrame({
        ...     "query": ["A", "A", "A", "B"],
        ...     "duration_days": pl.Series([30.0, 30.0, 30.0, 30.0], dtype=pl.Float32),
        ...     "boolean_value": [True, False, None, None],
        ... })
        >>> _prevalence_by_task(ids)
        {('A', 30.0): 0.5}

        Unlabeled input (no ``boolean_value`` column) yields no prevalences:

        >>> _prevalence_by_task(ids.drop("boolean_value"))
        {}
    """
    if TaskQuerySchema.boolean_value_name not in identifiers.columns:
        return {}
    prevalence = (
        identifiers.drop_nulls(TaskQuerySchema.boolean_value_name)
        .group_by(TaskQuerySchema.query_name, TaskQuerySchema.duration_days_name)
        .agg(pl.col(TaskQuerySchema.boolean_value_name).mean().alias("prevalence"))
    )
    return {
        (row[TaskQuerySchema.query_name], float(row[TaskQuerySchema.duration_days_name])): float(
            row["prevalence"]
        )
        for row in prevalence.iter_rows(named=True)
    }


@dataclass
class _HistorySerializer:
    """Serializes each ``(subject_id, prediction_time)`` group's history exactly once.

    Wraps the dataset's base-class ``load_subject_data`` (full untruncated pre-prediction window, no prepended
    query token) and caches by dataset row group so the many ``(query, duration_days)`` rows sharing one
    prediction context reuse a single serialized string.  Tracks truncation stats for the per-run log line.
    """

    dataset: EveryQueryPytorchDataset
    max_events: int
    code_descriptions: dict[str, str] | None

    def __post_init__(self) -> None:
        self._index_to_code = {idx: code for code, idx in self.dataset.code_to_index.items()}
        self._cache: dict[tuple[int, datetime], SerializedHistory] = {}
        self._gap_days = self._gaps_from_schema_df()
        self.n_serialized = 0
        self.n_truncated = 0

    def _gaps_from_schema_df(self) -> list[float]:
        """Per-row gap (days) between the last observed event and the prediction time."""
        schema_df = self.dataset.schema_df
        if _LAST_TIME_COL not in schema_df.columns:
            logger.warning(
                f"schema_df has no {_LAST_TIME_COL!r} column; serialized event times will be "
                f"anchored at the last observed event instead of the prediction time."
            )
            return [0.0] * schema_df.height
        gaps = (
            (schema_df[TaskQuerySchema.prediction_time_name] - schema_df[_LAST_TIME_COL])
            .dt.total_seconds()
            .cast(pl.Float64)
            / 86400.0
        ).fill_null(0.0)
        return [max(g, 0.0) for g in gaps]

    def history_for_row(self, row_idx: int) -> SerializedHistory:
        """Return the serialized history for dataset row ``row_idx``, cached per group."""
        subject_id, end_idx = self.dataset.index[row_idx]
        prediction_time = self.dataset.schema_df[TaskQuerySchema.prediction_time_name][row_idx]
        key = (subject_id, prediction_time)
        if key in self._cache:
            return self._cache[key]

        dynamic_data, _static = self.dataset.load_subject_data(subject_id=subject_id, st=0, end=end_idx)
        events = events_from_jnrt_dense(
            dynamic_data.to_dense(), self._index_to_code, gap_days=self._gap_days[row_idx]
        )
        history = serialize_history(
            events, max_events=self.max_events, code_descriptions=self.code_descriptions
        )
        self._cache[key] = history
        self.n_serialized += 1
        self.n_truncated += int(history.truncated)
        return history

    def log_truncation_stats(self) -> None:
        if self.n_serialized == 0:
            return
        fraction = self.n_truncated / self.n_serialized
        logger.info(
            f"Serialized {self.n_serialized} unique (subject, prediction_time) histories; "
            f"{self.n_truncated} ({fraction:.1%}) were clipped to the most recent "
            f"{self.max_events} events."
        )


class _ShardedWriter:
    """Flushes completed rows to ``shards/`` (predictions) + ``details_shards/`` (sidecar).

    Each flush writes one parquet per directory via a ``.tmp`` sibling + atomic rename, so a crash mid-write
    never leaves a half-written ``.parquet`` to corrupt a later ``resume`` scan.  Prediction shards are
    ``PredictionSchema.align``-ed and carry the run-metadata blob; the sidecar carries the same identifier
    columns plus ``occurs_prob``, ``parse_failed``, and ``method_used``.
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
        tmp = target.with_name(target.name + ".tmp")
        pq.write_table(table, tmp)
        tmp.replace(target)

    def flush(self, results: list[tuple[int, ProbabilityResult]]) -> None:
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
                    "method_used": [res.method_used for _, res in results],
                }
            )
        )

        shard_name = f"shard_{self._run_id}_{self._n_shards:05d}.parquet"
        self._atomic_write(aligned, self._shards_dir / shard_name)
        self._atomic_write(
            details.to_arrow().replace_schema_metadata(self._metadata), self._details_dir / shard_name
        )
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
    """Concatenate all shards into a single parquet at ``out_fp``; returns the row count."""
    merged = pq.read_table(shards_dir).replace_schema_metadata(metadata)
    tmp = out_fp.with_name(out_fp.name + ".tmp")
    pq.write_table(merged, tmp)
    tmp.replace(out_fp)
    return merged.num_rows


def _print_dry_run_prompts(
    serializer: _HistorySerializer,
    identifiers: pl.DataFrame,
    pending: list[int],
    method: str,
    code_descriptions: dict[str, str] | None,
    n_prompts: int = 3,
) -> None:
    """Print fully serialized prompts for the first few distinct histories, then return."""
    seen_groups: set[tuple[int, datetime]] = set()
    shown = 0
    for row_idx in pending:
        subject_id = identifiers[TaskQuerySchema.subject_id_name][row_idx]
        prediction_time = identifiers[TaskQuerySchema.prediction_time_name][row_idx]
        if (subject_id, prediction_time) in seen_groups:
            continue
        seen_groups.add((subject_id, prediction_time))

        history = serializer.history_for_row(row_idx)
        question = serialize_question(
            identifiers[TaskQuerySchema.query_name][row_idx],
            float(identifiers[TaskQuerySchema.duration_days_name][row_idx]),
            code_descriptions,
        )
        print(
            f"\n{'=' * 80}\n--- prompt {shown + 1} (subject={subject_id}, "
            f"prediction_time={prediction_time}) ---"
        )
        print(f"[system]\n{SYSTEM_PROMPT}\n")
        print(f"[user]\n{build_user_prompt(history.text, question, method)}")
        shown += 1
        if shown >= n_prompts:
            break
    print(f"\n{'=' * 80}\nDry run: {shown} prompt(s) shown; no requests were sent.")


async def _run_async(
    cfg: DictConfig,
    predictor: LLMPredictor,
    serializer: _HistorySerializer,
    identifiers: pl.DataFrame,
    pending: list[int],
    writer: _ShardedWriter,
    prevalences: dict[tuple[str, float], float],
    code_descriptions: dict[str, str] | None,
) -> None:
    """Fan pending rows out to the LLM (semaphore-bounded) and flush shards as they finish."""

    async def one_row(row_idx: int) -> tuple[int, ProbabilityResult]:
        history = serializer.history_for_row(row_idx)
        query = identifiers[TaskQuerySchema.query_name][row_idx]
        duration = float(identifiers[TaskQuerySchema.duration_days_name][row_idx])
        question = serialize_question(query, duration, code_descriptions)
        result = await predictor.predict_prob(
            history.text, question, fallback_prob=prevalences.get((query, duration))
        )
        return row_idx, result

    # Serialize all histories up front (sequential disk IO, cached per group) so truncation
    # stats are complete before the first request and event-loop workers stay IO-only.
    for row_idx in pending:
        serializer.history_for_row(row_idx)
    serializer.log_truncation_stats()

    buffer: list[tuple[int, ProbabilityResult]] = []
    shard_size = int(cfg.batch_size)
    tasks = [asyncio.create_task(one_row(row_idx)) for row_idx in pending]
    try:
        for future in asyncio.as_completed(tasks):
            buffer.append(await future)
            if len(buffer) >= shard_size:
                writer.flush(buffer)
                buffer = []
    except BaseException:
        # A transport failure (or Ctrl-C) aborts the run; flush what already completed so
        # resume=true can pick up from here, then cancel the in-flight remainder.
        writer.flush(buffer)
        for task in tasks:
            task.cancel()
        raise
    writer.flush(buffer)


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

    dataset_cfg = MEDSTorchDataConfig(
        tensorized_cohort_dir=str(cfg.tensorized_cohort_dir),
        task_labels_dir=str(tasks_dir),
        max_seq_len=int(cfg.max_events),  # unused by our load_subject_data path, but required
        seq_sampling_strategy="to_end",
        static_inclusion_mode="omit",
        batch_mode="SM",
        include_window_last_observed_in_schema=True,
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

    serializer = _HistorySerializer(
        dataset=dataset, max_events=int(cfg.max_events), code_descriptions=code_descriptions
    )

    if cfg.dry_run:
        _print_dry_run_prompts(serializer, identifiers, pending, cfg.method, code_descriptions)
        return

    metadata = _run_metadata(cfg)
    predictor = LLMPredictor(
        base_url=cfg.base_url,
        api_key=cfg.api_key,
        model=cfg.model,
        method=cfg.method,
        temperature=float(cfg.temperature),
        seed=int(cfg.seed),
        max_concurrency=int(cfg.max_concurrency),
        max_retries=int(cfg.max_retries),
        request_timeout=float(cfg.request_timeout),
        top_logprobs=int(cfg.top_logprobs),
        fallback_prob=0.5 if cfg.fallback_prob is None else float(cfg.fallback_prob),
        extra_body=None if cfg.extra_body is None else OmegaConf.to_container(cfg.extra_body),
    )
    # Per-task marginal prevalence is the parse-failure fallback only when the user hasn't
    # pinned an explicit fallback_prob.
    prevalences = _prevalence_by_task(identifiers) if cfg.fallback_prob is None else {}

    run_id = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    writer = _ShardedWriter(identifiers, shards_dir, details_dir, metadata, run_id)

    t0 = time.perf_counter()
    asyncio.run(
        _run_async(cfg, predictor, serializer, identifiers, pending, writer, prevalences, code_descriptions)
    )
    elapsed = time.perf_counter() - t0

    if predictor.n_parse_failures:
        rate = predictor.n_parse_failures / max(len(pending), 1)
        logger.warning(
            f"{predictor.n_parse_failures} rows ({rate:.1%}) hit total parse failure and were "
            f"written with the fallback probability (parse_failed=True in details.parquet)."
        )
    logger.info(
        f"Completed {writer.n_rows_written} rows in {elapsed:.1f}s "
        f"({predictor.n_requests} requests, {predictor.n_guided_fallbacks} guided fallbacks, "
        f"{predictor.n_parse_failures} parse failures)"
    )

    n_merged = _merge_shards(shards_dir, output_dir / "predictions.parquet", metadata)
    _merge_shards(details_dir, output_dir / "details.parquet", metadata)
    logger.info(f"Merged {n_merged} predictions to {output_dir / 'predictions.parquet'}")


if __name__ == "__main__":
    main()
