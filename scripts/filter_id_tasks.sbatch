#!/bin/bash
#SBATCH --job-name=eq-filter-id
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=logs/eq_filter_id_%j.out
#
# Like subsample_tasks.sbatch, but keeps ONLY the in-distribution codes listed in CODES.
# The raw eval grid mixes ID and OOD codes; only the ID half is scored.
#
#   mkdir -p logs    # SLURM opens --output BEFORE this script runs, so it must already
#                    # exist, and it resolves relative to the SUBMISSION directory.
#   sbatch scripts/filter_id_tasks.sbatch
#
# Reads the whole input into memory (polars, single pass) and writes ONE parquet file.
# Feed OUTPUT_FILE's parent directory to EQ_llm_predict as tasks_dir.
#
# NOTE this is a fresh draw, NOT a subset of eval_tasks_500: the per-group shuffle sees a
# different frame once OOD rows are gone, so it shares no rows with that run. Point it at a
# NEW llm_predict output_dir -- resume does not fingerprint tasks_dir and would otherwise
# union two different patient draws under one task.

set -euo pipefail

# ─── EDIT ME ────────────────────────────────────────────────────────────────────────────

# Repo checkout. The job may start in any directory, so we cd here explicitly.
export REPO_DIR="$HOME/EveryQuery_text"

# Input: either a single TaskQuerySchema parquet or a directory of shards (read recursively).
# The RAW grid, before any code filtering.
export INPUT="/users/gbk2114/nwicu-eq/eval-new-sampler/eval/held_out"

# The ID half of the code split. A YAML `codes:` list (or a flat list). MUST be the NWICU
# split, not the MIMIC one — the script raises if nothing matches, but a PARTIAL match is
# silent apart from the "N absent from input" line, so read that line before trusting the run.
export CODES="/groups/mm6677_gp/data/NWICU/code-split/split__seed42__pool3cc17c86f45e__oodb0e525c8bf23__id14c9331281ce/id_eval_codes.yaml"

# Output: a single parquet FILE (its parent directory is created by the script).
export OUTPUT_FILE="/users/gbk2114/nwicu-eq/eval_tasks_500_id/eval/held_out/tasks.parquet"

# Rows kept per label per task — 250 + 250 = 500 patients per task.
export PER_CLASS=250

# Drop tasks that can't field this many in EITHER class. 1 drops the degenerate
# single-class tasks (zero positives or zero negatives — no AUROC, nothing to learn).
export MIN_PER_CLASS=1

# Change to re-draw a different random sample.
export SEED=0

# Cap the task count to bound API spend. 0 = every ID task. Raise MIN_PER_CLASS or set this
# if the dry run's token estimate comes back too large.
export N_TASKS=0

# ─── END EDIT ME ────────────────────────────────────────────────────────────────────────

cd "$REPO_DIR"

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} started=$(date -Is)"
echo "in=$INPUT  codes=$CODES"
echo "out=$OUTPUT_FILE  per_class=$PER_CLASS  min_per_class=$MIN_PER_CLASS  seed=$SEED  n_tasks=$N_TASKS"

extra_args=()
[[ "$N_TASKS" -gt 0 ]] && extra_args+=(--n-tasks "$N_TASKS")

uv run scripts/filter_id_tasks.py \
    "$INPUT" \
    "$OUTPUT_FILE" \
    --codes "$CODES" \
    --per-class "$PER_CLASS" \
    --min-per-class "$MIN_PER_CLASS" \
    --seed "$SEED" \
    "${extra_args[@]}"

echo "finished=$(date -Is)"
