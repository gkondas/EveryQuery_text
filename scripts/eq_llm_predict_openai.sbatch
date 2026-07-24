#!/bin/bash
#SBATCH --job-name=eq-llm-openai
#SBATCH --partition=<FILL>
#SBATCH --account=<FILL>
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/eq_llm_%j.out
#SBATCH --requeue
#
# EQ_llm_predict against the OpenAI API — CPU-only, no GPU, no model serving.
#
#   mkdir -p logs    # SLURM opens --output BEFORE this script runs, so it must already
#                    # exist, and it resolves relative to the SUBMISSION directory.
#   sbatch scripts/eq_llm_predict_openai.sbatch
#
# Self-contained by design: this does NOT `source env.sh` (the convention documented in
# README "Paths & environment"). Everything is set in the EDIT ME block below, so one file
# describes the whole run. The tradeoff is that path changes have to be made here as well
# as in env.sh if you use both.

set -euo pipefail

# ─── EDIT ME ────────────────────────────────────────────────────────────────────────────

# Repo checkout. The job may start in any directory, so we cd here explicitly.
export REPO_DIR="$HOME/EveryQuery_text"

# Inputs. TENSORIZED_COHORT_DIR is the EQ_process_data `final/` output; TASKS_DIR holds
# TaskQuerySchema parquets for the split named below.
export TASKS_DIR="$HOME/data/MIMIC_MEDS_DEMO/eq_tasks/eval/held_out"
export TENSORIZED_COHORT_DIR="$HOME/data/MIMIC_MEDS_DEMO/eq_processed/final"
export SPLIT="held_out"

# Output. Use a NEW directory per configuration — resume refuses to extend shards written
# under a different model/prompt/sampling config (the run_fingerprint check).
export OUTPUT_DIR="$HOME/data/MIMIC_MEDS_DEMO/llm_baseline_out_gpt"

# Model. MUST support n>1 on chat completions — the whole method is a 20-way sampled
# Yes/No vote in a single request. Reasoning/o-series models do not qualify.
export MODEL="<chat-model-with-n-support>"

# Serialization. max_events counts unique timestamps, NOT code lines — one ICU timestamp
# can expand to 60+ codes, so prompt length scales with codes-per-event. Set
# CODE_DESCRIPTIONS to "" for the raw-code ablation arm.
export MAX_EVENTS=50
export CODE_DESCRIPTIONS="${TENSORIZED_COHORT_DIR}/metadata/codes.parquet"

# Sampling vote. TEMPERATURE must be > 0 (at 0 every sample is identical and the
# probability collapses to 0 or 1; the code raises on this).
export N_SAMPLES=20
export TEMPERATURE=0.7
export SMOOTHING=0.0

# Throughput. Tune MAX_CONCURRENCY to your OpenAI rate-limit tier: exhausting the retry
# budget on 429s aborts the run by design (transport failures are never silently written
# as fallback probabilities).
export MAX_CONCURRENCY=4
export MAX_RETRIES=8

# Rows per shard flush. Deliberately small: the in-flight buffer is flushed on Ctrl-C but
# NOT on SLURM's time-limit SIGTERM (Python terminates without raising), so this is the
# worst-case number of rows whose API spend is lost on a timeout. 25 keeps that cheap.
export BATCH_SIZE=25

# Smoke test: set to a row count for a short verification run, or "" for the full pass.
# Start with 20 to confirm n>1 works and that parsing survives guided_choice=false.
export LIMIT="20"

# API key. Kept OUT of this file on purpose — scripts/ is tracked by git and a literal key
# here is one `git push` from a leak. Store it mode-600 instead:
#     mkdir -p ~/.config/openai && umask 077 && cat > ~/.config/openai/key
export OPENAI_KEY_FILE="$HOME/.config/openai/key"
# Inline alternative — only if this file is untracked / .gitignore'd. NOT recommended:
# export OPENAI_API_KEY="sk-..."

# ─── END EDIT ME ────────────────────────────────────────────────────────────────────────

cd "$REPO_DIR"

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} started=$(date -Is)"

# Resolve the key from the file unless it was set inline above.
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    if [[ ! -r "$OPENAI_KEY_FILE" ]]; then
        echo "ERROR: no API key. Set OPENAI_API_KEY inline, or create $OPENAI_KEY_FILE:" >&2
        echo "       mkdir -p ~/.config/openai && umask 077 && cat > $OPENAI_KEY_FILE" >&2
        exit 1
    fi
    OPENAI_API_KEY="$(< "$OPENAI_KEY_FILE")"
    export OPENAI_API_KEY
fi

# Fail fast and loudly if compute nodes are firewalled off from the API — otherwise the
# first symptom is a retry storm that burns the budget before aborting. 401 = reachable
# (we deliberately send no credentials here); 000 = no egress.
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
[[ -n "$CODE_DESCRIPTIONS" ]] && extra_args+=("code_descriptions=$CODE_DESCRIPTIONS")

# No srun and no job array, matching the single-node/single-task convention in
# generate_tasks/redesign-spec.md. Do NOT run array tasks against one OUTPUT_DIR: there is
# no shard/offset knob, so every task would compute the same pending set and duplicate
# rows into shards/. Parallelize by splitting TASKS_DIR into separate OUTPUT_DIRs instead.
#
# resume=true is safe on a fresh OUTPUT_DIR (no-op) and makes this idempotent under
# --requeue: completed rows in shards/ are anti-joined out of the pending set.
uv run EQ_llm_predict \
    tasks_dir="$TASKS_DIR" \
    tensorized_cohort_dir="$TENSORIZED_COHORT_DIR" \
    output_dir="$OUTPUT_DIR" \
    split="$SPLIT" \
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
