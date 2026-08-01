#!/bin/bash
#SBATCH --job-name=eq-llm-openai
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
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

# Inputs. Keep these identical to eq_llm_predict_dryrun.sbatch — the dry run is only
# evidence for the real run if both read the same cohort.
export TASKS_DIR="/users/gbk2114/mimic-eq/eval_tasks_500_id/eval/held_out"
export TENSORIZED_COHORT_DIR="/groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort/processed"
export SPLIT="held_out"

# Output. Use a NEW directory per configuration — resume refuses to extend shards written
# under a different model/prompt/sampling config (the run_fingerprint check).
#
# Also new per TASKS_DIR, which the fingerprint does NOT cover: the ID-filtered subsample is
# an independent draw, so resuming into llm_baseline_out_gpt_alltasks would union two
# different patient populations under the same task.
export OUTPUT_DIR="/users/gbk2114/eq-neurips-rebuttle/llm_baseline_out_gpt_id"

# Model. Must honor `max_tokens` and `temperature` on chat completions — reasoning models
# (o-series, gpt-5) reject the first and ignore the second, and their answer would be
# reasoning rather than a bare float. Raising N_SAMPLES above 1 additionally requires `n`.
export MODEL="gpt-4.1"

# Which serialization to send. Match what you verified in the dry run.
export PROMPT_STYLE="llm4healthcare"

# --- event_stream knobs (ignored under llm4healthcare) ---
export MAX_EVENTS=50
# event_stream only. build_style does not accept descriptions for llm4healthcare, whose
# question template renders the raw MEDS code either way — so under this arm the model is
# asked about `LAB//51067//mg/24hr//value_[1260.0,1405.0)` verbatim, with no gloss.
export CODE_DESCRIPTIONS="${TENSORIZED_COHORT_DIR}/metadata/codes.parquet"

# --- llm4healthcare knobs (ignored under event_stream) ---
export L4H_PANEL="mimic_icu17"
export L4H_FORM="string"
export L4H_IMPUTE="locf"
export L4H_MAX_VISITS=48
export L4H_RECORD_TIME_MODE="days_before"
# float = the reference paper's format (average a parsed probability over the samples);
# yes_no = the better-calibrated vote, but no longer their prompt.
export L4H_RESPONSE_FORMAT="float"

# Answer extraction — the reference paper's single-shot setup: one completion, one parsed
# float, greedy. TEMPERATURE=0 is only legal because N_SAMPLES=1; the code raises on
# n_samples>1 at temperature 0 (identical samples collapse every probability to 0 or 1),
# so raise TEMPERATURE alongside N_SAMPLES if you ever sample more.
export N_SAMPLES=1
export TEMPERATURE=0
# Their constant for the "I do not know" escape hatch — not a number, so it lands here as
# a parse failure. Not derived from this split's labels.
export FALLBACK_PROB=0.501
# No-op in float mode (the Laplace prior only touches the Yes/No vote); kept for the
# yes_no arm.
export SMOOTHING=0.0

# ~77% of histories on this eval grid hold no mimic_icu17 measurement at all, so their prompt
# is age + sex + "no measurements were recorded". true writes FALLBACK_PROB for those rows
# without sending a request — most of the API bill, none of the signal. They are flagged
# `skipped` in details.parquet; drop them before reading their AUROC, since a constant
# probability contributes only ties. false = ask the model anyway (the reference setup).
export SKIP_EMPTY_HISTORIES=true

# Throughput. Tune MAX_CONCURRENCY to your OpenAI rate-limit tier: exhausting the retry
# budget on 429s aborts the run by design (transport failures are never silently written
# as fallback probabilities).
export MAX_CONCURRENCY=8
export MAX_RETRIES=8

# Rows per shard flush. Deliberately small: the in-flight buffer is flushed on Ctrl-C but
# NOT on SLURM's time-limit SIGTERM (Python terminates without raising), so this is the
# worst-case number of rows whose API spend is lost on a timeout. 25 keeps that cheap.
export BATCH_SIZE=25

# Smoke test: set to a row count for a short verification run, or "" for the full pass.
# Confirms the float parser survives real completions — check parse_failed in details.parquet
# before committing to the full pass.
#
# Use ~500, not 20. limit truncates the PENDING ROW LIST before histories are grouped, and
# that list starts with empty-history rows, which SKIP_EMPTY_HISTORIES=true answers without
# an API call. A small limit therefore sends zero requests and exits 0 having proven nothing.
export LIMIT="500"

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

# Optional args assembled conditionally so unset ones fall back to the config defaults,
# and so the unused style's knobs never reach the config.
extra_args=()
[[ -n "$LIMIT" ]] && extra_args+=("limit=$LIMIT")
if [[ "$PROMPT_STYLE" == "llm4healthcare" ]]; then
    extra_args+=("l4h.panel=$L4H_PANEL" "l4h.form=$L4H_FORM" "l4h.impute=$L4H_IMPUTE")
    extra_args+=("l4h.max_visits=$L4H_MAX_VISITS")
    extra_args+=("l4h.record_time_mode=$L4H_RECORD_TIME_MODE")
    extra_args+=("l4h.response_format=$L4H_RESPONSE_FORMAT")
else
    extra_args+=("max_events=$MAX_EVENTS")
    [[ -n "$CODE_DESCRIPTIONS" ]] && extra_args+=("code_descriptions=$CODE_DESCRIPTIONS")
fi

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
    prompt_style="$PROMPT_STYLE" \
    base_url=https://api.openai.com/v1 \
    api_key="$OPENAI_API_KEY" \
    model="$MODEL" \
    guided_choice=false \
    n_samples="$N_SAMPLES" \
    temperature="$TEMPERATURE" \
    smoothing="$SMOOTHING" \
    fallback_prob="$FALLBACK_PROB" \
    skip_empty_histories="$SKIP_EMPTY_HISTORIES" \
    max_concurrency="$MAX_CONCURRENCY" \
    max_retries="$MAX_RETRIES" \
    batch_size="$BATCH_SIZE" \
    resume=true \
    "${extra_args[@]}"

echo "finished=$(date -Is)"
echo "predictions: $OUTPUT_DIR/predictions.parquet"
echo "diagnostics: $OUTPUT_DIR/details.parquet  (parse_failed, skipped, n_parsed, n_yes/n_no)"
echo "NOTE: EQ_evaluate reads predictions.parquet, which has no 'skipped' column — join"
echo "      details.parquet on subject_id/prediction_time/query/duration_days and drop the"
echo "      skipped rows before reporting per-task AUROC."
