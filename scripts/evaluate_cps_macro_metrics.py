#!/usr/bin/env python3
"""Compute human JUSThink trajectory metrics from team bundles."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

try:
    from .cps_humanlm_utils import action_type, discourse_type, matching_label
    from .prepare_cps_humanlm_data import attach_submit_results
except ImportError:
    from cps_humanlm_utils import action_type, discourse_type, matching_label
    from prepare_cps_humanlm_data import attach_submit_results


def relative_position(index: int | None, length: int) -> float | None:
    return round(index / max(length - 1, 1), 4) if index is not None else None


def first_index(values: list[str | None], targets: set[str]) -> int | None:
    return next((idx for idx, value in enumerate(values) if value in targets), None)


def summarize_bundle(bundle: dict, repair_window: int) -> dict:
    events = attach_submit_results(bundle["annotated_corpus"], bundle["corpus"])
    action_codes = [action_type(event) for event in events]
    alignment_codes = [matching_label(event) for event in events]
    discourse_codes = [discourse_type(event) for event in events]
    submit_results = [
        event["submit_result"]
        for event in events
        if event.get("subject") == "T" and event.get("submit_result")
    ]
    gaps = [
        float(result["abs_error"])
        for result in submit_results
        if result.get("abs_error") is not None
    ]
    matches = Counter(code for code in alignment_codes if code)
    mismatch_indices = [idx for idx, code in enumerate(alignment_codes) if code == "mismatch"]
    turn_ids = {
        (event.get("attempt_no"), event.get("turn_no"))
        for event in events
        if event.get("turn_no") is not None
    }
    repaired = sum(
        any(
            code == "match"
            for code in alignment_codes[idx + 1 : idx + repair_window + 1]
        )
        for idx in mismatch_indices
    )
    combined_codes = []
    for discourse, alignment, action in zip(discourse_codes, alignment_codes, action_codes):
        combined_codes.extend(code for code in (discourse, alignment, action) if code)
    transitions = Counter(
        f"{left} -> {right}" for left, right in zip(combined_codes, combined_codes[1:])
    )

    return {
        "team_no": int(bundle["team_no"]),
        "task_performance": {
            "submission_count": len(submit_results),
            "best_cost_gap": min(gaps) if gaps else None,
            "found_optimal": any(gap == 0 for gap in gaps),
            "cost_gap_trajectory": gaps,
        },
        "behavior_rhythm": {
            "event_count": len(events),
            "turn_count": len(turn_ids),
            "human_utterance_count": sum(
                event.get("subject") in {"A", "B"} and event.get("verb") == "says"
                for event in events
            ),
            "first_edit_relative_position": relative_position(
                first_index(action_codes, {"edit_add", "edit_remove", "edit_load"}),
                len(events),
            ),
            "first_submit_relative_position": relative_position(
                first_index(action_codes, {"submit"}), len(events)
            ),
            "first_match_relative_position": relative_position(
                first_index(alignment_codes, {"match"}), len(events)
            ),
            "action_counts": dict(Counter(code for code in action_codes if code)),
            "discourse_counts": dict(Counter(code for code in discourse_codes if code)),
        },
        "language_action_alignment": {
            "counts": dict(matches),
            "match_to_mismatch_ratio": (
                round(matches["match"] / matches["mismatch"], 4)
                if matches["mismatch"]
                else None
            ),
            "mismatch_repair_within_n": repaired,
            "mismatch_repair_probability": (
                round(repaired / len(mismatch_indices), 4) if mismatch_indices else None
            ),
            "repair_window_events": repair_window,
        },
        "transition_counts": dict(transitions),
    }


def aggregate(team_rows: list[dict]) -> dict:
    transitions = Counter()
    action_counts = Counter()
    discourse_counts = Counter()
    best_gaps = []
    submissions = []
    optimal = 0
    repair_rates = []
    turn_counts = []
    edit_counts = []
    match_counts = Counter()
    first_submit_progress = []
    first_match_progress = []
    for row in team_rows:
        performance = row["task_performance"]
        behavior = row["behavior_rhythm"]
        alignment = row["language_action_alignment"]
        transitions.update(row["transition_counts"])
        action_counts.update(behavior["action_counts"])
        discourse_counts.update(behavior["discourse_counts"])
        submissions.append(performance["submission_count"])
        turn_counts.append(behavior["turn_count"])
        edit_counts.append(
            sum(behavior["action_counts"].get(key, 0) for key in ("edit_add", "edit_remove", "edit_load"))
        )
        match_counts.update(alignment["counts"])
        if behavior["first_submit_relative_position"] is not None:
            first_submit_progress.append(behavior["first_submit_relative_position"])
        if behavior["first_match_relative_position"] is not None:
            first_match_progress.append(behavior["first_match_relative_position"])
        optimal += int(performance["found_optimal"])
        if performance["best_cost_gap"] is not None:
            best_gaps.append(performance["best_cost_gap"])
        if alignment["mismatch_repair_probability"] is not None:
            repair_rates.append(alignment["mismatch_repair_probability"])
    return {
        "team_count": len(team_rows),
        "found_optimal_rate": round(optimal / len(team_rows), 4) if team_rows else None,
        "mean_submission_count": round(statistics.mean(submissions), 4) if submissions else None,
        "mean_turn_count": round(statistics.mean(turn_counts), 4) if turn_counts else None,
        "mean_edit_count": round(statistics.mean(edit_counts), 4) if edit_counts else None,
        "mean_best_cost_gap": round(statistics.mean(best_gaps), 4) if best_gaps else None,
        "alignment_counts": dict(match_counts),
        "match_to_mismatch_ratio": (
            round(match_counts["match"] / match_counts["mismatch"], 4)
            if match_counts["mismatch"]
            else None
        ),
        "mean_first_submit_progress": (
            round(statistics.mean(first_submit_progress), 4) if first_submit_progress else None
        ),
        "mean_first_match_progress": (
            round(statistics.mean(first_match_progress), 4) if first_match_progress else None
        ),
        "mean_mismatch_repair_probability": (
            round(statistics.mean(repair_rates), 4) if repair_rates else None
        ),
        "action_counts": dict(action_counts),
        "discourse_counts": dict(discourse_counts),
        "transition_counts": dict(transitions),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repair-window-events", default=10, type=int)
    args = parser.parse_args()

    paths = sorted(args.input_dir.rglob("team_*_bundle_atc21s_full.json"))
    if not paths:
        raise FileNotFoundError(f"No team bundles found under {args.input_dir}")
    teams = [
        summarize_bundle(json.loads(path.read_text()), args.repair_window_events)
        for path in paths
    ]
    result = {"aggregate": aggregate(teams), "teams": teams}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
