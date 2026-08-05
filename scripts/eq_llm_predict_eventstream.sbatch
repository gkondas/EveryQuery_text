#!/bin/bash
#SBATCH --job-name=eq-llm-es
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/eq_llm_es_%j.out
#SBATCH --requeue
#
# EQ_llm_predict, event_stream style, against the OpenAI API — CPU-only, no model serving.
#
#   mkdir -p logs    # SLURM opens --output BEFORE this script runs, so it must already
#                    # exist, and it resolves relative to the SUBMISSION directory.
#   sbatch scripts/eq_llm_predict_eventstream.sbatch
#
# Differs from eq_llm_predict_openai.sbatch in more than prompt_style. That script reproduces
# the llm4healthcare paper: one greedy sample, parse a float. event_stream is a Yes/No style,
# so probability comes from the repeated-sampling vote instead — n_samples samples at a
# non-zero temperature, score = fraction of Yes. n_samples=1 at temperature 0 would collapse
# every prediction to exactly 0.0 or 1.0, which is why those knobs are set differently here.
#
# Verify with scripts/eq_llm_predict_eventstream_dryrun.sbatch first — event_stream prompts
# are substantially longer than the panel prompt, and the input tokens are what you pay for.
#
# Self-contained by design: this does NOT `source env.sh`. Everything is in the EDIT ME block.

set -euo pipefail

# ─── EDIT ME ────────────────────────────────────────────────────────────────────────────

# Repo checkout. The job may start in any directory, so we cd here explicitly.
export REPO_DIR="$HOME/EveryQuery_text"

# Inputs. Keep these identical to the dry-run script — the dry run is only evidence for the
# real run if both read the same cohort.
export TASKS_DIR="/users/gbk2114/nwicu-eq/eval_tasks_500_20tasks/eval/held_out"
export TENSORIZED_COHORT_DIR="/groups/mm6677_gp/data/NWICU/processed"
export SPLIT="held_out"

# Output. A NEW directory — prompt_style is part of the run_fingerprint, so resume refuses to
# extend shards written under the llm4healthcare arm. Do not aim this at the l4h output dir.
export OUTPUT_DIR="/users/gbk2114/eq-neurips-rebuttle/llm_baseline_out_eventstream_nwicu"

# Model. Must honor `max_tokens`, `temperature` and `n` on chat completions — reasoning models
# (o-series, gpt-5) reject the first and ignore the second.
export MODEL="gpt-4.1"

# Serialization. Keep in sync with the dry run you sized.
export MAX_EVENTS=50
export CODE_DESCRIPTIONS="${TENSORIZED_COHORT_DIR}/metadata/codes.parquet"

# Answer extraction — the repeated-sampling Yes/No vote. The score takes one of N_SAMPLES + 1
# discrete values, so 20 gives 0.05 resolution; all N_SAMPLES samples ride on ONE request
# (`n=`), so the long history prefix is billed and prefilled once, not 20 times.
export N_SAMPLES=20
# MUST be > 0 with N_SAMPLES > 1 — the code raises otherwise. Worth sweeping.
export TEMPERATURE=0.7
# Symmetric Laplace prior on the vote. 0.0 = raw fraction (fine for AUROC, which is what
# EQ_evaluate reports); set ~0.5 to keep log-loss / Brier finite off the 0/1 endpoints.
export SMOOTHING=0.0
# Written only on TOTAL parse failure (no sample yielded a Yes or No). "" leaves it null → 0.5.
# Never derive this from the evaluated split's own labels.
export FALLBACK_PROB=""

# Throughput. Tune to your OpenAI rate-limit tier: exhausting the retry budget on 429s aborts
# the run by design (transport failures are never silently written as fallback probabilities).
# Note this arm requests N_SAMPLES completions per call, so tokens-per-minute bites sooner
# than on the single-sample l4h arm even at the same concurrency.
export MAX_CONCURRENCY=4
export MAX_RETRIES=8

# Rows per shard flush. Deliberately small: the in-flight buffer is flushed on Ctrl-C but NOT
# on SLURM's time-limit SIGTERM, so this is the worst-case number of rows whose API spend is
# lost on a timeout.
export BATCH_SIZE=25

# Smoke test: set to a row count for a short verification run, or "" for the full pass. Start
# with 20 and check n_yes/n_no in details.parquet — if every row is 20/0 or 0/20, the model is
# not actually sampling and TEMPERATURE or the `n` parameter is not being honored.
export LIMIT="20"

# API key. Kept OUT of this file on purpose — scripts/ is tracked by git and a literal key
# here is one `git push` from a leak. Store it mode-600 instead:
#     mkdir -p ~/.config/openai && umask 077 && cat > ~/.config/openai/key
export OPENAI_KEY_FILE="$HOME/.config/openai/key"

# ─── END EDIT ME ────────────────────────────────────────────────────────────────────────

cd "$REPO_DIR"

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} started=$(date -Is)"
echo "style=event_stream  model=$MODEL  n_samples=$N_SAMPLES  temperature=$TEMPERATURE"

# Resolve the key from the file unless it was set inline in the environment.
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    if [[ ! -r "$OPENAI_KEY_FILE" ]]; then
        echo "ERROR: no API key. Set OPENAI_API_KEY inline, or create $OPENAI_KEY_FILE:" >&2
        echo "       mkdir -p ~/.config/openai && umask 077 && cat > $OPENAI_KEY_FILE" >&2
        exit 1
    fi
    OPENAI_API_KEY="$(< "$OPENAI_KEY_FILE")"
    export OPENAI_API_KEY
fi

# Fail fast and loudly if compute nodes are firewalled off from the API — otherwise the first
# symptom is a retry storm that burns the budget before aborting. 401 = reachable (we send no
# credentials here); 000 = no egress.
http_code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 \
    https://api.openai.com/v1/models || echo 000)
if [[ "$http_code" == "000" ]]; then
    echo "ERROR: no egress to api.openai.com from $(hostname) (curl got 000)." >&2
    echo "       Compute nodes are likely firewalled; check with your cluster admins." >&2
    exit 1
fi
echo "egress ok (api.openai.com -> $http_code)"

# Optional args assembled conditionally so unset ones fall back to the config defaults.
extra_args=()
[[ -n "$LIMIT" ]] && extra_args+=("limit=$LIMIT")
[[ -n "$FALLBACK_PROB" ]] && extra_args+=("fallback_prob=$FALLBACK_PROB")
[[ -n "$CODE_DESCRIPTIONS" ]] && extra_args+=("code_descriptions=$CODE_DESCRIPTIONS")

# guided_choice=false: constraining the answer to exactly "Yes"/"No" is a vLLM extension the
# OpenAI API does not implement. The leading Yes/No word is parsed leniently instead — watch
# n_unparsed in details.parquet. Flip to true (and repoint base_url) when serving via vLLM.
#
# No srun and no job array. Do NOT run array tasks against one OUTPUT_DIR: there is no
# shard/offset knob, so every task would compute the same pending set and duplicate rows.
# Parallelize by splitting TASKS_DIR into separate OUTPUT_DIRs instead.
#
# resume=true is safe on a fresh OUTPUT_DIR (no-op) and makes this idempotent under --requeue.
uv run EQ_llm_predict \
    tasks_dir="$TASKS_DIR" \
    tensorized_cohort_dir="$TENSORIZED_COHORT_DIR" \
    output_dir="$OUTPUT_DIR" \
    split="$SPLIT" \
    prompt_style=event_stream \
    max_events="$MAX_EVENTS" \
    base_url=https://api.openai.com/v1 \
    api_key="$OPENAI_API_KEY" \
    model="$MODEL" \
    guided_choice=false \
    n_samples="$N_SAMPLES" \
    temperature="$TEMPERATURE" \
    smoothing="$SMOOTHING" \
    max_concurrency="$MAX_CONCURRENCY" \
    max_retries="$MAX_RETRIES" \
    batch_size="$BATCH_SIZE" \
    resume=true \
    "${extra_args[@]}"

echo "finished=$(date -Is)"
echo "predictions: $OUTPUT_DIR/predictions.parquet"
echo "diagnostics: $OUTPUT_DIR/details.parquet  (parse_failed, n_yes/n_no/n_unparsed)"
