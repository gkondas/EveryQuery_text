# LLM baseline (`EQ_llm_predict`)

Serializes patient event histories as text, asks a locally-served LLM for per-row
event-occurrence probabilities, and writes `PredictionSchema`-conformant parquets — a
drop-in alternative to `EQ_predict` whose output feeds `EQ_evaluate` unchanged.

## Serving the model (vLLM)

Launch the server yourself before running `EQ_llm_predict`; the CLI only talks to an
OpenAI-compatible endpoint (`base_url`, default `http://localhost:8000/v1`).

**Dev / pipeline debugging** — Llama 3.1 8B Instruct, single GPU:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --port 8000 \
    --max-model-len 16384
```

**Real baseline run** — Llama 3.1 70B Instruct across 4 GPUs:

```bash
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --port 8000 \
    --tensor-parallel-size 4 \
    --max-model-len 16384
```

Pin `--revision` when serving and record it via `model_revision=...` so the run metadata
captures exactly which weights answered.

## Running

Eyeball prompts first (no GPU / server contact):

```bash
EQ_llm_predict \
    tasks_dir=/path/to/tasks \
    tensorized_cohort_dir=/path/to/tensorized_cohort \
    output_dir=/path/to/out \
    dry_run=true
```

Smoke run against the 8B server:

```bash
EQ_llm_predict \
    tasks_dir=/path/to/tasks \
    tensorized_cohort_dir=/path/to/tensorized_cohort \
    output_dir=/path/to/out \
    model=meta-llama/Llama-3.1-8B-Instruct \
    limit=100
```

Full 70B run (restartable — rerun the same command with `resume=true` after a crash;
completed rows in `{output_dir}/shards/` are skipped):

```bash
EQ_llm_predict \
    tasks_dir=/path/to/tasks \
    tensorized_cohort_dir=/path/to/tensorized_cohort \
    output_dir=/path/to/out \
    model=meta-llama/Llama-3.1-70B-Instruct \
    max_concurrency=32 \
    resume=true
```

Then evaluate exactly as for `EQ_predict` output:

```bash
EQ_evaluate predictions_parquet=/path/to/out/predictions.parquet metrics_parquet=/path/to/metrics.parquet
```

## Probability extraction

Two methods behind the `method` config field (see `llm_predict.py`):

- **`logprob` (default).** The model is instructed to answer with exactly one word — `Yes`
  or `No` — and the top-20 logprobs of the first generated token are requested. The Yes/No
  token logprobs (robust to leading-space / BPE-marker / casing variants) are softmaxed
  pairwise into a continuous probability. This avoids the mode collapse of free-text
  numeric answers. If only one of the two tokens appears in the top-k, the other side is
  bounded by the smallest returned logprob; if neither appears, the sample falls back to
  the guided method.
- **`guided` (fallback / comparison).** Free-text numeric answer constrained with vLLM's
  `guided_regex` structured output (`(0\.\d{1,4}|1\.0|0|1)`), then parsed.

On total parse failure the configured `fallback_prob` is written (default 0.5; set it
explicitly to inject an externally-estimated prevalence — it is never computed from the
evaluated split's labels, which would leak ground truth), and the row is flagged
`parse_failed=True` in `details.parquet`. Transport failures (after bounded retries with
exponential backoff) abort the run instead of writing fallbacks — restart with
`resume=true`.

`censor_prob` is a constant `0.0`: censoring is a data artifact, not a clinical
prediction. In `EQ_evaluate` this only affects the `censor_auroc` sanity metric, which
becomes a well-defined-but-uninformative 0.5 for constant scores; the headline
`occurs_auroc` drops censored rows before scoring and is unaffected.

## Ablations

- **Code descriptions**: raw codes like `ICD//E11.9` are near-opaque to a general instruct
  model. Point `code_descriptions=` at a `.json` mapping or a `.parquet` with
  `code`/`description` columns (a MEDS `metadata/codes.parquet` works directly) to render
  `ICD//E11.9 (Type 2 diabetes mellitus without complications)`. Leave `null` (default) for
  the raw-code arm.
- **History length**: `max_events` controls left-truncation (most recent events kept, with
  an omission marker). Per-run truncation counts/fractions are logged.

## Reproducibility

Every shard and the merged `predictions.parquet` / `details.parquet` carry a JSON blob
under the `every_query_llm_baseline` parquet metadata key: model name/revision, `method`,
prompt-template version + hash, `max_events`, whether code descriptions were used, seed,
temperature, and a hash of the resolved config. Read it back with:

```python
import json, pyarrow.parquet as pq
json.loads(pq.read_schema("predictions.parquet").metadata[b"every_query_llm_baseline"])
```

The prompt template lives only in `serialize.py`, is versioned
(`PROMPT_TEMPLATE_VERSION`), and any wording change changes the recorded hash.
