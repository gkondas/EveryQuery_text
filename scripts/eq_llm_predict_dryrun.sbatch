#!/bin/bash
#SBATCH --job-name=eq-llm-dryrun
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/eq_llm_dryrun_%j.out
#
# EQ_llm_predict dry run — CPU-only, no GPU, no API key, no network egress.
#
#   mkdir -p logs    # SLURM opens --output BEFORE this script runs, so it must already
#                    # exist, and it resolves relative to the SUBMISSION directory.
#   sbatch scripts/eq_llm_predict_dryrun.sbatch
#
# What this proves, before you spend a single token:
#   - the compute node can import the package and read the cohort off shared storage
#   - the task parquets load and the identifier frame builds
#   - histories serialize under the chosen prompt style (panel resolves, values decode)
#   - what the model will actually be sent, verbatim
#
#   - exactly how many API requests the real run will send, and their total prompt size
#
# Only 3 prompts are PRINTED, but dry_run_max_groups=null below serializes every history and
# sizes every request, so the reported character total is exact rather than extrapolated.
# That is a full pass over the cohort's windows — if it outgrows the 30-minute --time above,
# either raise it or drop dry_run_max_groups to sample (200 is the default).
#
# It writes nothing — dry_run returns before the output directory is touched, so OUTPUT_DIR
# below is a placeholder the config requires rather than a path that gets created.
#
# Sibling script for the real run: scripts/eq_llm_predict_openai.sbatch

set -euo pipefail

# ─── EDIT ME ────────────────────────────────────────────────────────────────────────────

# Repo checkout. The job may start in any directory, so we cd here explicitly.
export REPO_DIR="$HOME/EveryQuery_text"

# Inputs. TENSORIZED_COHORT_DIR is the EQ_process_data `final/` output; TASKS_DIR holds
# TaskQuerySchema parquets for the split named below. Both must be readable from the
# COMPUTE node — a path that only exists on the login node is exactly what this catches.
export TASKS_DIR="/users/gbk2114/mimic-eq/eval_tasks_500_20tasks/eval/held_out"
export TENSORIZED_COHORT_DIR="/groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort/processed"
export SPLIT="held_out"

# Required by the config but never created under dry_run. Point it at the directory the
# real run will use, so the printed banner matches what you are about to launch.
export OUTPUT_DIR="/users/gbk2114/eq-neurips-rebuttle/llm_baseline_out_dryrun"

# Which serialization to render: event_stream | llm4healthcare
export PROMPT_STYLE="llm4healthcare"

# --- event_stream knobs (ignored under llm4healthcare) ---
# max_events counts unique timestamps, NOT code lines — one ICU timestamp can expand to
# 60+ codes, so prompt length scales with codes-per-event. Set CODE_DESCRIPTIONS to ""
# for the raw-code ablation arm.
export MAX_EVENTS=50
export CODE_DESCRIPTIONS="${TENSORIZED_COHORT_DIR}/metadata/codes.parquet"

# --- llm4healthcare knobs (ignored under event_stream) ---
# Panel is bundled by name; pass a .yaml path instead to target a different cohort.
export L4H_PANEL="mimic_icu17"
export L4H_FORM="string"          # string | list | batches
export L4H_IMPUTE="locf"          # none | locf | locf_marked
export L4H_MAX_VISITS=48
# index reproduces the reference prompts (`0, 1, 2`) but tells the model nothing about
# spacing; days_before renders each visit's real distance from the prediction time, which a
# MEDS window has and their resampled ICU panels do not.
export L4H_RECORD_TIME_MODE="days_before"   # index | days_before

# ─── END EDIT ME ────────────────────────────────────────────────────────────────────────

cd "$REPO_DIR"

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} started=$(date -Is)"
echo "style=$PROMPT_STYLE  split=$SPLIT"

# Style-specific overrides, assembled conditionally so the unused arm's knobs never reach
# the config (they are ignored, but leaving them out keeps the printed command honest).
extra_args=()
if [[ "$PROMPT_STYLE" == "llm4healthcare" ]]; then
    extra_args+=("l4h.panel=$L4H_PANEL" "l4h.form=$L4H_FORM" "l4h.impute=$L4H_IMPUTE")
    extra_args+=("l4h.max_visits=$L4H_MAX_VISITS")
    extra_args+=("l4h.record_time_mode=$L4H_RECORD_TIME_MODE")
else
    extra_args+=("max_events=$MAX_EVENTS")
    [[ -n "$CODE_DESCRIPTIONS" ]] && extra_args+=("code_descriptions=$CODE_DESCRIPTIONS")
fi

# No base_url, no api_key, no model: dry_run returns before the LLM client is built.
uv run EQ_llm_predict \
    tasks_dir="$TASKS_DIR" \
    tensorized_cohort_dir="$TENSORIZED_COHORT_DIR" \
    output_dir="$OUTPUT_DIR" \
    split="$SPLIT" \
    prompt_style="$PROMPT_STYLE" \
    dry_run=true \
    dry_run_max_groups=null \
    "${extra_args[@]}"

echo "finished=$(date -Is)"
echo "No requests were sent and no files were written."
