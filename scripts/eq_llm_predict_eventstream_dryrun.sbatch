#!/bin/bash
#SBATCH --job-name=eq-llm-es-dryrun
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/eq_llm_es_dryrun_%j.out
#
# EQ_llm_predict dry run, event_stream style — CPU-only, no GPU, no API key, no egress.
#
#   mkdir -p logs    # SLURM opens --output BEFORE this script runs, so it must already
#                    # exist, and it resolves relative to the SUBMISSION directory.
#   sbatch scripts/eq_llm_predict_eventstream_dryrun.sbatch
#
# Same job as eq_llm_predict_dryrun.sbatch but pinned to event_stream, so the l4h knobs are
# gone rather than sitting in a dead branch. Run this BEFORE the real pass: event_stream
# prompts are much longer than the 17-feature panel (every code at every kept timestamp, not
# 17 rows), and this is what tells you by how much.
#
# Only 3 prompts are PRINTED, but dry_run_max_groups=null serializes every history and sizes
# every request, so the reported character total is exact rather than extrapolated. That is a
# full pass over the cohort's windows — if it outgrows the 30-minute --time above, either
# raise it or drop DRY_RUN_MAX_GROUPS to 200 to sample.
#
# It writes nothing — dry_run returns before the output directory is touched.
#
# Sibling script for the real run: scripts/eq_llm_predict_eventstream.sbatch

set -euo pipefail

# ─── EDIT ME ────────────────────────────────────────────────────────────────────────────

# Repo checkout. The job may start in any directory, so we cd here explicitly.
export REPO_DIR="$HOME/EveryQuery_text"

# Inputs. Both must be readable from the COMPUTE node — a path that only exists on the login
# node is exactly what this catches.
export TASKS_DIR="/users/gbk2114/nwicu-eq/eval_tasks_500_20tasks/eval/held_out"
export TENSORIZED_COHORT_DIR="/groups/mm6677_gp/data/NWICU/processed"
export SPLIT="held_out"

# Required by the config but never created under dry_run. Point it at the directory the real
# run will use, so the printed banner matches what you are about to launch.
export OUTPUT_DIR="/users/gbk2114/eq-neurips-rebuttle/llm_baseline_out_eventstream_nwicu"

# max_events counts unique timestamps, NOT code lines — one ICU timestamp can expand to 60+
# codes, so prompt length scales with codes-per-event. 50 matches the l4h dry run's window
# depth; the config default is 256. Check the reported char total before raising it.
export MAX_EVENTS=50
# Set to "" for the raw-code ablation arm (codes render as `LAB//220045` with no gloss).
export CODE_DESCRIPTIONS="${TENSORIZED_COHORT_DIR}/metadata/codes.parquet"

# null = size every pending row exactly (a full pass). 200 samples instead, much faster.
export DRY_RUN_MAX_GROUPS="null"

# ─── END EDIT ME ────────────────────────────────────────────────────────────────────────

cd "$REPO_DIR"

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} started=$(date -Is)"
echo "style=event_stream  split=$SPLIT  max_events=$MAX_EVENTS"

extra_args=()
[[ -n "$CODE_DESCRIPTIONS" ]] && extra_args+=("code_descriptions=$CODE_DESCRIPTIONS")

# No base_url, no api_key, no model: dry_run returns before the LLM client is built.
uv run EQ_llm_predict \
    tasks_dir="$TASKS_DIR" \
    tensorized_cohort_dir="$TENSORIZED_COHORT_DIR" \
    output_dir="$OUTPUT_DIR" \
    split="$SPLIT" \
    prompt_style=event_stream \
    max_events="$MAX_EVENTS" \
    dry_run=true \
    dry_run_max_groups="$DRY_RUN_MAX_GROUPS" \
    "${extra_args[@]}"

echo "finished=$(date -Is)"
echo "No requests were sent and no files were written."
