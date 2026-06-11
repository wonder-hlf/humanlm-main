#!/usr/bin/env python3
"""Generate human and simulated JUSThink macro-alignment reports."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_cps_macro_metrics import aggregate, summarize_bundle


CORE_METRICS = [
    ("mean_turn_count", "Mean turn count"),
    ("mean_edit_count", "Mean edit count"),
    ("mean_submission_count", "Mean submit count"),
    ("mean_best_cost_gap", "Mean best cost gap"),
    ("match_to_mismatch_ratio", "Match-to-mismatch ratio"),
    ("mean_first_submit_progress", "Mean first submit progress"),
    ("mean_first_match_progress", "Mean first match progress"),
    ("mean_mismatch_repair_probability", "Repair-after-mismatch rate"),
    ("found_optimal_rate", "Found optimal rate"),
]
MATRIX_CODES = [
    "route_instruction",
    "confirmation",
    "disagreement",
    "repair_talk",
    "feedback_interpretation",
    "match",
    "mismatch",
    "nonmatch",
    "edit_add",
    "edit_remove",
    "edit_load",
    "check",
    "submit",
]


def load_bundles(input_dir: Path, repair_window: int) -> dict:
    paths = sorted(input_dir.rglob("team_*_bundle_atc21s_full.json"))
    if not paths:
        paths = sorted(input_dir.rglob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No JSON bundles found under {input_dir}")
    teams = [
        summarize_bundle(json.loads(path.read_text()), repair_window)
        for path in paths
    ]
    return {"aggregate": aggregate(teams), "teams": teams}


def fmt(value: object) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def normalize(counter: dict[str, int]) -> dict[str, float]:
    total = sum(counter.values())
    return {key: value / total for key, value in counter.items()} if total else {}


def js_divergence(left: dict[str, int], right: dict[str, int]) -> float | None:
    p, q = normalize(left), normalize(right)
    keys = set(p) | set(q)
    if not keys:
        return None
    midpoint = {key: (p.get(key, 0) + q.get(key, 0)) / 2 for key in keys}

    def kl(source: dict[str, float]) -> float:
        return sum(
            probability * math.log2(probability / midpoint[key])
            for key, probability in source.items()
            if probability > 0
        )

    return round((kl(p) + kl(q)) / 2, 6)


def top_transitions(transitions: dict[str, int], limit: int = 25) -> list[tuple[str, int]]:
    return Counter(transitions).most_common(limit)


def transition_matrix_table(transitions: dict[str, int]) -> list[str]:
    lines = [
        "| From / To | " + " | ".join(MATRIX_CODES) + " |",
        "|---|" + "|".join("---:" for _ in MATRIX_CODES) + "|",
    ]
    for source in MATRIX_CODES:
        values = [str(transitions.get(f"{source} -> {target}", 0)) for target in MATRIX_CODES]
        lines.append(f"| {source} | " + " | ".join(values) + " |")
    return lines


def metric_table(aggregate_row: dict) -> list[str]:
    lines = ["| Metric | Human value |", "|---|---:|"]
    for key, label in CORE_METRICS:
        lines.append(f"| {label} | {fmt(aggregate_row.get(key))} |")
    counts = aggregate_row.get("alignment_counts", {})
    for key in ("match", "mismatch", "nonmatch"):
        lines.append(f"| Total {key} | {fmt(counts.get(key, 0))} |")
    return lines


def human_report(result: dict, repair_window: int) -> str:
    aggregate_row = result["aggregate"]
    lines = [
        "# Human Macro Statistics",
        "",
        f"Computed from {aggregate_row['team_count']} human JUSThink team trajectories.",
        f"Repair-after-mismatch uses a window of {repair_window} subsequent events.",
        "",
        "## Requested Metrics",
        "",
        *metric_table(aggregate_row),
        "",
        "Progress values are normalized event positions from 0 (task start) to 1 (task end).",
        "",
        "## Additional Metrics",
        "",
        f"- Action counts: `{json.dumps(aggregate_row.get('action_counts', {}), ensure_ascii=False)}`",
        f"- Discourse counts: `{json.dumps(aggregate_row.get('discourse_counts', {}), ensure_ascii=False)}`",
        "- Additional retained signals: event count, utterance count, first edit progress, "
        "cost-gap trajectory, found-optimal rate, check/press counts, and discourse categories.",
        "",
        "## Top Behavior Transitions",
        "",
        "| Transition | Count |",
        "|---|---:|",
    ]
    lines.extend(
        f"| `{transition}` | {count} |"
        for transition, count in top_transitions(aggregate_row["transition_counts"])
    )
    lines.extend(
        [
            "",
            "## Core Behavior Transition Matrix",
            "",
            *transition_matrix_table(aggregate_row["transition_counts"]),
        ]
    )
    return "\n".join(lines) + "\n"


def comparison_report(human: dict, simulated: dict | None) -> str:
    lines = [
        "# Simulated vs Human Macro Alignment",
        "",
        "The human reference metrics are generated from real JUSThink trajectories.",
        "",
    ]
    if simulated is None:
        return "\n".join(
            lines
            + [
                "Simulation metrics are not available yet. Generate complete simulated "
                "team trajectories, then rerun this script with `--sim-input-dir`.",
                "",
                "The current Qwen rollout file contains independent next-turn samples; "
                "it is not a complete interactive trajectory and therefore must not be "
                "used to claim macro behavioral alignment.",
            ]
        ) + "\n"

    human_agg, sim_agg = human["aggregate"], simulated["aggregate"]
    lines.extend(["| Metric | Human | Simulated | Absolute gap |", "|---|---:|---:|---:|"])
    for key, label in CORE_METRICS:
        human_value, sim_value = human_agg.get(key), sim_agg.get(key)
        gap = abs(human_value - sim_value) if human_value is not None and sim_value is not None else None
        lines.append(f"| {label} | {fmt(human_value)} | {fmt(sim_value)} | {fmt(gap)} |")
    divergence = js_divergence(human_agg["transition_counts"], sim_agg["transition_counts"])
    lines.extend(
        [
            "",
            "## Structural Alignment",
            "",
            f"- Behavior-transition Jensen-Shannon divergence: `{fmt(divergence)}`",
            "- Lower divergence indicates more similar transition structure.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--human-input-dir", required=True, type=Path)
    parser.add_argument("--sim-input-dir", type=Path)
    parser.add_argument("--human-report", default=ROOT / "reports/human_macro_stats.md", type=Path)
    parser.add_argument(
        "--comparison-report",
        default=ROOT / "reports/sim_vs_human_macro_alignment.md",
        type=Path,
    )
    parser.add_argument("--repair-window-events", default=10, type=int)
    args = parser.parse_args()

    human = load_bundles(args.human_input_dir, args.repair_window_events)
    simulated = (
        load_bundles(args.sim_input_dir, args.repair_window_events)
        if args.sim_input_dir
        else None
    )
    args.human_report.parent.mkdir(parents=True, exist_ok=True)
    args.comparison_report.parent.mkdir(parents=True, exist_ok=True)
    args.human_report.write_text(
        human_report(human, args.repair_window_events),
        encoding="utf-8",
    )
    args.comparison_report.write_text(comparison_report(human, simulated), encoding="utf-8")
    print(f"Wrote {args.human_report}")
    print(f"Wrote {args.comparison_report}")


if __name__ == "__main__":
    main()
