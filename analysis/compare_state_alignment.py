#!/usr/bin/env python3
"""Compare two state-alignment evaluations on shared held-out samples."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def load_successful(path: Path) -> dict[str, dict]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {
        row["sample_key"]: row
        for row in rows
        if not row.get("error") and row.get("state_alignment")
    }


def score(row: dict, dimension: str | None = None) -> float:
    alignment = row["state_alignment"]
    if dimension is None:
        return float(alignment["overall_state_alignment"])
    return float(alignment["dimension_scores"][dimension]["score"])


def compare(base: dict[str, dict], trained: dict[str, dict]) -> dict:
    shared = sorted(set(base) & set(trained))
    dimensions = sorted(
        set.intersection(
            *[
                set(row["state_alignment"]["dimension_scores"])
                for row in list(base.values()) + list(trained.values())
            ]
        )
    )

    def summarize(dimension: str | None) -> dict:
        deltas = [score(trained[key], dimension) - score(base[key], dimension) for key in shared]
        return {
            "count": len(deltas),
            "base_mean": round(statistics.mean(score(base[key], dimension) for key in shared), 4),
            "trained_mean": round(
                statistics.mean(score(trained[key], dimension) for key in shared), 4
            ),
            "mean_delta": round(statistics.mean(deltas), 4),
            "median_delta": round(statistics.median(deltas), 4),
            "trained_wins": sum(delta > 0 for delta in deltas),
            "ties": sum(delta == 0 for delta in deltas),
            "base_wins": sum(delta < 0 for delta in deltas),
        }

    return {
        "shared_successful_samples": len(shared),
        "base_only_successful_samples": len(set(base) - set(trained)),
        "trained_only_successful_samples": len(set(trained) - set(base)),
        "overall": summarize(None),
        "dimensions": {dimension: summarize(dimension) for dimension in dimensions},
    }


def markdown(result: dict) -> str:
    overall = result["overall"]
    lines = [
        "# Paired State Alignment Comparison",
        "",
        f"- Shared successful samples: {result['shared_successful_samples']}",
        f"- Base-only successful samples: {result['base_only_successful_samples']}",
        f"- Trained-only successful samples: {result['trained_only_successful_samples']}",
        "",
        "| Metric | Base mean | Training-two mean | Mean delta | Median delta | Training-two wins | Ties | Base wins |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| Overall | {overall['base_mean']} | {overall['trained_mean']} | "
        f"{overall['mean_delta']} | {overall['median_delta']} | "
        f"{overall['trained_wins']} | {overall['ties']} | {overall['base_wins']} |",
    ]
    for dimension, values in result["dimensions"].items():
        lines.append(
            f"| {dimension} | {values['base_mean']} | {values['trained_mean']} | "
            f"{values['mean_delta']} | {values['median_delta']} | "
            f"{values['trained_wins']} | {values['ties']} | {values['base_wins']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--trained", required=True, type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args()

    result = compare(load_successful(args.base), load_successful(args.trained))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(markdown(result), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
