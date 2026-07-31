#!/usr/bin/env python3
"""Subsample a TaskQuery parquet to ID codes only.

Same sampling as ``subsample_tasks.py`` -- N positives + N negatives per (query,
duration_days) task -- but first drops every row whose ``query`` code is absent from a
codes YAML (the in-distribution half of a code split).  The eval grid mixes ID and OOD
codes; only the ID half should be scored.

    uv run scripts/filter_id_tasks.py eval_tasks/eval/held_out out.parquet \\
        --codes code-split/.../id_eval_codes.yaml
"""

import argparse
import sys
from pathlib import Path

import polars as pl

from every_query.generate_tasks.sample_tasks import read_query_codes
from subsample_tasks import class_counts, pick_tasks, subsample


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--codes", type=Path, required=True, help="YAML: a `codes:` list, or a flat list")
    p.add_argument("--per-class", type=int, default=250, help="rows per label per task (default 250)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--min-per-class",
        type=int,
        default=0,
        help="drop tasks with fewer than this many positives or negatives (default 0 = keep all)",
    )
    p.add_argument(
        "--n-tasks",
        type=int,
        default=0,
        help="keep only this many random (query, duration_days) tasks (default 0 = keep all)",
    )
    args = p.parse_args()

    codes = read_query_codes(args.codes)
    # a TASKS_DIR of shards is as common an input here as a single file
    df = pl.read_parquet(args.input / "**/*.parquet" if args.input.is_dir() else args.input)
    kept = df.filter(pl.col("query").is_in(codes))

    present = set(kept["query"].unique().to_list())
    missing = sorted(c for c in codes if c not in present)
    print(f"{len(codes)} codes in {args.codes.name}, {len(present)} present in input")
    if missing:
        # a renamed/stale code silently shrinks the task grid -- say so rather than
        # quietly evaluating a smaller split than the one that was asked for
        head = ", ".join(missing[:10])
        print(f"{len(missing)} absent from input: {head}{' ...' if len(missing) > 10 else ''}")
    if kept.is_empty():
        raise SystemExit(
            f"no rows matched the {len(codes)} codes in {args.codes} -- wrong code split, or "
            f"wrong input dir?"
        )
    print(f"{len(df)} -> {len(kept)} rows after the code filter")

    out = subsample(kept, args.per_class, args.seed, args.min_per_class)
    # after the per-class sampling, not before, so a --n-tasks run is a strict row-subset
    # of the full run at the same --seed
    if args.n_tasks:
        out = pick_tasks(out, args.n_tasks, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(args.output)

    counts = class_counts(out)
    short = counts.filter((pl.col("n_pos") < args.per_class) | (pl.col("n_neg") < args.per_class))
    print(f"{len(kept)} -> {len(out)} rows over {len(counts)} tasks")
    print(f"{len(short)} tasks short of {args.per_class}/class:")
    print(short.sort("n_pos"))


def _selfcheck() -> None:
    import tempfile
    from datetime import datetime

    df = pl.DataFrame(
        {
            "subject_id": list(range(600)),
            "prediction_time": [datetime(2023, 1, 1)] * 600,
            "boolean_value": [True, False] * 300,
            "query": ["DIAGNOSIS//ICD//10//A419"] * 300 + ["OOD//CODE"] * 300,
            "duration_days": [30.0] * 600,
        }
    )
    with tempfile.TemporaryDirectory() as d:
        tasks, yml, out = Path(d) / "t.parquet", Path(d) / "id.yaml", Path(d) / "sub" / "o.parquet"
        df.write_parquet(tasks)
        yml.write_text("codes:\n- DIAGNOSIS//ICD//10//A419\n- NEVER//SEEN\n")

        sys.argv = ["x", str(tasks), str(out), "--codes", str(yml), "--per-class", "2"]
        main()
        got = pl.read_parquet(out)  # mkdir -p of a missing parent is part of the contract
        assert got["query"].unique().to_list() == ["DIAGNOSIS//ICD//10//A419"], "OOD code survived"
        assert got.height == 4, got.height  # 2 per class, one task

        # an empty result is an error, not a 0-row parquet handed to EQ_llm_predict
        yml.write_text("codes:\n- NEVER//SEEN\n")
        try:
            main()
        except SystemExit:
            pass
        else:
            raise AssertionError("empty filter result did not raise")
    print("ok")


if __name__ == "__main__":
    _selfcheck() if sys.argv[1:2] == ["--selfcheck"] else main()
