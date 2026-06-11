#!/usr/bin/env python3
"""Summarize DSV4Pro state-alignment judgments from Qwen rollouts."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize(rows: list[dict]) -> dict:
    scores = defaultdict(list)
    overall = []
    missing = Counter()
    unsupported = Counter()
    errors = Counter()

    for row in rows:
        if row.get("error"):
            errors[row["error"].split(":", 1)[0]] += 1
            continue
        judgment = row.get("state_alignment", {})
        for dimension, value in judgment.get("dimension_scores", {}).items():
            score = value.get("score") if isinstance(value, dict) else None
            if isinstance(score, (int, float)):
                scores[dimension].append(float(score))
        if isinstance(judgment.get("overall_state_alignment"), (int, float)):
            overall.append(float(judgment["overall_state_alignment"]))
        missing.update(str(item) for item in judgment.get("missing_state", []))
        unsupported.update(str(item) for item in judgment.get("redundant_unsupported_state", []))

    def stats(values: list[float]) -> dict:
        return {
            "count": len(values),
            "mean": round(statistics.mean(values), 4) if values else None,
            "median": round(statistics.median(values), 4) if values else None,
            "min": round(min(values), 4) if values else None,
            "max": round(max(values), 4) if values else None,
        }

    return {
        "rows": len(rows),
        "successful_judgments": len(rows) - sum(errors.values()),
        "errors": dict(errors),
        "overall_state_alignment": stats(overall),
        "dimension_scores": {dimension: stats(values) for dimension, values in scores.items()},
        "top_missing_states": missing.most_common(20),
        "top_redundant_unsupported_states": unsupported.most_common(20),
    }


def markdown(result: dict) -> str:
    lines = [
        "# State Alignment Evaluation",
        "",
        f"- Rows: {result['rows']}",
        f"- Successful judgments: {result['successful_judgments']}",
        f"- Overall mean: {result['overall_state_alignment']['mean']}",
        "",
        "## Dimension Scores",
        "",
        "| Dimension | Count | Mean | Median | Min | Max |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dimension, values in sorted(result["dimension_scores"].items()):
        lines.append(
            f"| {dimension} | {values['count']} | {values['mean']} | "
            f"{values['median']} | {values['min']} | {values['max']} |"
        )
    for title, key in (
        ("Missing States", "top_missing_states"),
        ("Redundant / Unsupported States", "top_redundant_unsupported_states"),
    ):
        lines.extend(["", f"## {title}", "", "| State | Count |", "|---|---:|"])
        lines.extend(f"| {state} | {count} |" for state, count in result[key])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args()

    result = summarize(load_jsonl(args.input))
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text, encoding="utf-8")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(markdown(result), encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
