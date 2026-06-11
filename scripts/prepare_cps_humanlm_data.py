#!/usr/bin/env python3
"""Build leakage-safe, multi-turn JUSThink data for HumanLM-style training."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd

try:
    from .cps_humanlm_utils import (
        CPS_STATE_DIMENSIONS,
        HUMAN_PARTICIPANTS,
        build_proxy_labels,
        prefix_environment_state,
        prefix_role_state,
        prefix_student_profile,
        serialize_event,
        target_actions_after_utterance,
    )
    from .merge_cps_consecutive_utterances import merge_consecutive_utterances
    from .prepare_cps_sft_data import split_samples
except ImportError:
    from cps_humanlm_utils import (
        CPS_STATE_DIMENSIONS,
        HUMAN_PARTICIPANTS,
        build_proxy_labels,
        prefix_environment_state,
        prefix_role_state,
        prefix_student_profile,
        serialize_event,
        target_actions_after_utterance,
    )
    from merge_cps_consecutive_utterances import merge_consecutive_utterances
    from prepare_cps_sft_data import split_samples


def attach_submit_results(annotated: list[dict], corpus: list[dict]) -> list[dict]:
    """Copy submit results into the aligned annotated stream without future lookup."""
    if len(annotated) != len(corpus):
        raise ValueError(
            "annotated_corpus and corpus must align by index: "
            f"{len(annotated)} != {len(corpus)}"
        )
    result = []
    for idx, event in enumerate(annotated):
        copied = dict(event)
        if corpus[idx].get("submit_result"):
            copied["submit_result"] = corpus[idx]["submit_result"]
        result.append(copied)
    return result


def make_sample(bundle: dict, events: list[dict], idx: int, context_window: int) -> dict | None:
    target_event = events[idx]
    participant = target_event.get("subject")
    if participant not in HUMAN_PARTICIPANTS or target_event.get("verb") != "says":
        return None
    if target_event.get("is_incomplete_fragment"):
        return None
    utterance = str(target_event.get("object", "")).strip()
    if not utterance:
        return None

    prefix = events[:idx]
    history = prefix[-context_window:] if context_window > 0 else prefix
    actions = target_actions_after_utterance(events, idx, str(participant))
    team_no = int(bundle["team_no"])

    return {
        "team_no": team_no,
        "participant": str(participant),
        "attempt_no": target_event.get("attempt_no"),
        "turn_no": target_event.get("turn_no"),
        "source_index": idx,
        "student_profile": prefix_student_profile(prefix, str(participant)),
        "role_state": prefix_role_state(prefix, str(participant)),
        "environment_state": prefix_environment_state(prefix),
        "dialogue_and_action_history": [serialize_event(event) for event in history],
        "state_dimensions": CPS_STATE_DIMENSIONS,
        "optional_steering_config": None,
        "ground_truth": {
            "utterance": utterance,
            "actions_before_next_human_utterance": actions,
        },
        "computable_state_proxies": build_proxy_labels(actions),
        "qwen_rollout_request": {
            "input_fields": [
                "student_profile",
                "role_state",
                "environment_state",
                "dialogue_and_action_history",
                "state_dimensions",
            ],
            "required_output": {
                "latent_states": list(CPS_STATE_DIMENSIONS),
                "utterance": "string",
                "action_intent": "null or task action intent object",
            },
            "ground_truth_visible_to_qwen": False,
        },
        "judge_request": {
            "context_fields": [
                "student_profile",
                "role_state",
                "environment_state",
                "dialogue_and_action_history",
            ],
            "ground_truth_fields": [
                "ground_truth.utterance",
                "ground_truth.actions_before_next_human_utterance",
            ],
            "dimensions": CPS_STATE_DIMENSIONS,
            "judge_role": "score_qwen_rollout_only",
            "required_output": {
                "dimension_scores": {
                    dimension["cps_name"]: {"score": "0-1", "rationale": "string"}
                    for dimension in CPS_STATE_DIMENSIONS.values()
                },
                "overall_state_alignment": "0-1",
                "missing_state": "list of strings",
                "redundant_unsupported_state": "list of strings",
            },
        },
    }


def write_split(rows: list[dict], output_dir: Path, split: str) -> None:
    jsonl_path = output_dir / f"{split}.jsonl"
    jsonl_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    frame = pd.DataFrame(
        {
            "team_no": [row["team_no"] for row in rows],
            "participant": [row["participant"] for row in rows],
            "attempt_no": [row["attempt_no"] for row in rows],
            "turn_no": [row["turn_no"] for row in rows],
            "sample_json": [json.dumps(row, ensure_ascii=False) for row in rows],
        }
    )
    frame.to_parquet(output_dir / f"{split}.parquet", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--context-window", default=80, type=int)
    parser.add_argument("--max-samples-per-team", default=20, type=int)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    samples = []
    total_merged_fragments = 0
    total_excluded_incomplete_targets = 0
    input_files = sorted(args.input_dir.rglob("team_*_bundle_atc21s_full.json"))
    if not input_files:
        raise FileNotFoundError(f"No team bundles found under {args.input_dir}")

    for path in input_files:
        bundle = json.loads(path.read_text())
        annotated = attach_submit_results(bundle["annotated_corpus"], bundle["corpus"])
        events, stats = merge_consecutive_utterances(
            annotated,
            team_no=bundle.get("team_no"),
        )
        total_merged_fragments += stats["merged_fragments"]
        total_excluded_incomplete_targets += sum(
            event.get("subject") in HUMAN_PARTICIPANTS
            and event.get("verb") == "says"
            and event.get("is_incomplete_fragment", False)
            for event in events
        )
        team_samples = [
            sample
            for idx in range(len(events))
            if (sample := make_sample(bundle, events, idx, args.context_window))
        ]
        rng.shuffle(team_samples)
        if args.max_samples_per_team > 0:
            team_samples = team_samples[: args.max_samples_per_team]
        samples.extend(team_samples)
        print(f"{path.name}: {len(team_samples)} samples")

    splits, split_teams = split_samples(samples, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        write_split(rows, args.output_dir, split)
        print(f"{split}: {len(rows)}")

    manifest = {
        "schema": "justhink_humanlm_v1",
        "input_dir": str(args.input_dir),
        "context_window": args.context_window,
        "max_samples_per_team": args.max_samples_per_team,
        "total_samples": len(samples),
        "merged_fragments": total_merged_fragments,
        "excluded_incomplete_targets": total_excluded_incomplete_targets,
        "split_teams": split_teams,
        "splits": {split: len(rows) for split, rows in splits.items()},
        "future_information_policy": (
            "All profile, role, environment, and performance fields are computed "
            "only from events before the target utterance."
        ),
        "role_limitation": (
            "Current cost-view/edit-view assignment is not available in the bundle; "
            "only capabilities observed in the event prefix are recorded."
        ),
        "state_dimensions": CPS_STATE_DIMENSIONS,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
