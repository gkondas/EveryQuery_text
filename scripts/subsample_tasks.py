#!/usr/bin/env python3
"""Subsample a TaskQuery parquet to N positives + N negatives per (query, duration_days) task.

Censored rows (``boolean_value`` null) are dropped.  Tasks with fewer than ``--per-class``
rows in a class keep whatever they have -- nothing is upsampled.

    uv run scripts/subsample_tasks.py tasks.parquet tasks_500.parquet
"""

import argparse
from pathlib import Path

import polars as pl


TASK = ["query", "duration_days"]


def class_counts(df: pl.DataFrame) -> pl.DataFrame:
    """Per-task positive / negative row counts."""
    return df.group_by(TASK).agg(
        pl.col("boolean_value").sum().alias("n_pos"),
        (~pl.col("boolean_value")).sum().alias("n_neg"),
    )


def pick_tasks(df: pl.DataFrame, n_tasks: int, seed: int = 0) -> pl.DataFrame:
    """Keep a random ``n_tasks`` distinct (query, duration_days) tuples; all of them if fewer exist."""
    tasks = df.select(TASK).unique()
    return df.join(tasks.sample(n=min(n_tasks, tasks.height), seed=seed), on=TASK, how="semi")


def subsample(df: pl.DataFrame, per_class: int = 250, seed: int = 0, min_per_class: int = 0) -> pl.DataFrame:
    """Randomly keep up to ``per_class`` rows per (query, duration_days, boolean_value) group.

    Tasks short of ``per_class`` in a class keep whatever they have (so they come out smaller
    and class-imbalanced); tasks below ``min_per_class`` in either class are dropped entirely.
    """
    out = (
        df.filter(pl.col("boolean_value").is_not_null())
        # shuffle once, then head(n) per group == a uniform sample per group
        .sample(fraction=1.0, shuffle=True, seed=seed)
        # ponytail: one row per (task, subject) so "500 patients" means 500 distinct
        # patients even when a subject has several prediction times for the same task.
        .unique(subset=["query", "duration_days", "subject_id"], keep="first", maintain_order=True)
        .group_by("query", "duration_days", "boolean_value", maintain_order=True)
        .head(per_class)
    )
    if min_per_class:
        keep = class_counts(out).filter(
            (pl.col("n_pos") >= min_per_class) & (pl.col("n_neg") >= min_per_class)
        )
        out = out.join(keep.select(TASK), on=TASK, how="semi")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
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

    # a TASKS_DIR of shards is as common an input here as a single file
    df = pl.read_parquet(args.input / "**/*.parquet" if args.input.is_dir() else args.input)
    out = subsample(df, args.per_class, args.seed, args.min_per_class)
    # after the per-class sampling, not before, so a --n-tasks run is a strict row-subset
    # of the full run at the same --seed
    if args.n_tasks:
        out = pick_tasks(out, args.n_tasks, args.seed)
    out.write_parquet(args.output)

    counts = class_counts(out)
    short = counts.filter((pl.col("n_pos") < args.per_class) | (pl.col("n_neg") < args.per_class))
    print(f"{len(df)} -> {len(out)} rows over {len(counts)} tasks")
    print(f"{len(short)} tasks short of {args.per_class}/class:")
    print(short.sort("n_pos"))


def _selfcheck() -> None:
    from datetime import datetime

    df = pl.DataFrame(
        {
            "subject_id": [i % 7 for i in range(600)],  # repeated subjects
            "prediction_time": [datetime(2023, 1, 1)] * 600,
            "boolean_value": [True, False, None] * 200,
            "query": ["A"] * 300 + ["B"] * 300,
            "duration_days": [30.0] * 600,
        }
    )
    out = subsample(df, per_class=2, seed=1)
    assert out["boolean_value"].null_count() == 0
    assert out.group_by("query", "duration_days", "boolean_value").len()["len"].max() == 2
    assert out.select("query", "duration_days", "subject_id").is_duplicated().sum() == 0

    # a task with only 1 positive: kept as-is by default, dropped by --min-per-class 2
    thin = pl.DataFrame(
        {
            "subject_id": [100, 101, 102],
            "prediction_time": [datetime(2023, 1, 1)] * 3,
            "boolean_value": [True, False, False],
            "query": ["C"] * 3,
            "duration_days": [30.0] * 3,
        }
    )
    both = pl.concat([df, thin])
    assert subsample(both, per_class=2, seed=1).filter(pl.col("query") == "C").height == 3
    assert subsample(both, per_class=2, seed=1, min_per_class=2).filter(pl.col("query") == "C").height == 0

    full = subsample(both, per_class=2, seed=1)
    one = pick_tasks(full, 1, seed=1)
    assert one.select(TASK).n_unique() == 1
    assert one.join(full, on=one.columns, how="semi").height == one.height  # strict subset
    assert pick_tasks(full, 99, seed=1).height == full.height  # asking for more than exist is fine
    print("ok")


if __name__ == "__main__":
    main()
