#!/usr/bin/env python3
"""Summarize computable JUSThink state proxies and macro behavior metrics."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize(rows: list[dict]) -> dict:
    proxies = Counter()
    participant_samples = Counter()
    action_sequences = Counter()
    best_gaps = []
    submission_counts = []
    found_optimal = 0

    for row in rows:
        proxies.update(row.get("computable_state_proxies", {}))
        participant_samples[row["participant"]] += 1
        actions = row["ground_truth"]["actions_before_next_human_utterance"]
        sequence = tuple(
            event.get("matching_label") or event.get("verb", "unknown") for event in actions
        )
        if sequence:
            action_sequences[sequence] += 1
        performance = row["environment_state"]["performance_so_far"]
        if performance["best_cost_gap"] is not None:
            best_gaps.append(performance["best_cost_gap"])
        submission_counts.append(performance["submission_count"])
        found_optimal += int(performance["found_optimal_so_far"])

    transitions = Counter()
    for sequence, count in action_sequences.items():
        for left, right in zip(sequence, sequence[1:]):
            transitions[f"{left} -> {right}"] += count

    return {
        "samples": len(rows),
        "samples_by_participant": dict(participant_samples),
        "action_and_alignment_proxy_counts": dict(proxies),
        "performance_so_far_distribution": {
            "best_cost_gaps": best_gaps,
            "submission_counts": submission_counts,
            "found_optimal_so_far_count": found_optimal,
        },
        "action_transition_counts": dict(transitions),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = summarize(load_jsonl(args.input))
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
