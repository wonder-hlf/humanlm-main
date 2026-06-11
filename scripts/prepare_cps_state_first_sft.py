#!/usr/bin/env python3
"""Select judged latent states and build state-first SFT parquet files."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

try:
    from .cps_humanlm_utils import action_type
    from .run_cps_dsv4pro_alignment import qwen_rollout_messages, sample_key
except ImportError:
    from cps_humanlm_utils import action_type
    from run_cps_dsv4pro_alignment import qwen_rollout_messages, sample_key


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def candidate_score(row: dict) -> tuple[float, float]:
    alignment = row.get("state_alignment") or {}
    overall = float(alignment.get("overall_state_alignment", -1))
    scores = alignment.get("dimension_scores") or {}
    values = [
        float(value["score"])
        for value in scores.values()
        if isinstance(value, dict) and isinstance(value.get("score"), (int, float))
    ]
    mean_dimension = sum(values) / len(values) if values else -1
    return overall, mean_dimension


def select_candidates(rows: list[dict], minimum_score: float) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if not row.get("error") and row.get("qwen_rollout") and row.get("state_alignment"):
            grouped[row["sample_key"]].append(row)
    selected = {}
    for key, candidates in grouped.items():
        best = max(candidates, key=candidate_score)
        if candidate_score(best)[0] >= minimum_score:
            selected[key] = best
    return selected


def ground_truth_action_intent(actions: list[dict]) -> dict | None:
    if not actions:
        return None
    action = actions[0]
    verb = action.get("verb")
    intent_type = {
        "adds": "add_track",
        "removes": "remove_track",
        "loads": "load_solution",
        "submits": "submit",
        "presses": "press",
    }.get(verb)
    classified = action_type(action)
    if classified in {"submit", "check"}:
        intent_type = classified
    if not intent_type:
        return None
    intent = {"type": intent_type}
    if verb in {"adds", "removes"}:
        intent["track"] = action.get("object")
    elif verb == "loads":
        intent["solution"] = action.get("object")
    elif action.get("object") not in (None, ""):
        intent["object"] = action.get("object")
    return intent


def make_state_first_row(sample: dict, candidate: dict) -> dict:
    rollout = candidate["qwen_rollout"]
    target = {
        "latent_states": rollout["latent_states"],
        "utterance": sample["ground_truth"]["utterance"],
        "action_intent": ground_truth_action_intent(
            sample["ground_truth"]["actions_before_next_human_utterance"]
        ),
    }
    messages = qwen_rollout_messages(sample)
    prompt = [
        {"role": message["role"], "name": "", "content": message["content"]}
        for message in messages
    ]
    return {
        "prompt": prompt,
        "generation": json.dumps(target, ensure_ascii=False, separators=(",", ":")),
        "sample_key": sample_key(sample),
        "team_no": sample["team_no"],
        "participant": sample["participant"],
        "attempt_no": sample["attempt_no"],
        "turn_no": sample["turn_no"],
        "source_index": sample["source_index"],
        "selected_candidate_index": candidate["candidate_index"],
        "selected_overall_state_alignment": candidate_score(candidate)[0],
        "state_alignment": candidate["state_alignment"],
    }


def write_split(rows: list[dict], output_dir: Path, split: str) -> None:
    pd.DataFrame(
        [{"prompt": row["prompt"], "generation": row["generation"]} for row in rows]
    ).to_parquet(output_dir / f"{split}.parquet", index=False)
    (output_dir / f"{split}.selected.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    (output_dir / f"{split}.preview.json").write_text(
        json.dumps(rows[:5], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--alignment-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--minimum-score", default=0.5, type=float)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "justhink_state_first_sft_v1",
        "source_dir": str(args.source_dir),
        "alignment_dir": str(args.alignment_dir),
        "minimum_score": args.minimum_score,
        "splits": {},
        "target_policy": (
            "Selected Qwen latent states plus ground-truth human utterance and "
            "ground-truth nearby action intent. Qwen-generated utterances/actions "
            "are never used as SFT targets."
        ),
    }
    for split in ("train", "val", "test"):
        samples = load_jsonl(args.source_dir / f"{split}.jsonl")
        candidates = load_jsonl(args.alignment_dir / f"{split}.jsonl")
        selected = select_candidates(candidates, args.minimum_score)
        rows = [
            make_state_first_row(sample, selected[key])
            for sample in samples
            if (key := sample_key(sample)) in selected
        ]
        write_split(rows, args.output_dir, split)
        manifest["splits"][split] = {
            "source_samples": len(samples),
            "selected_samples": len(rows),
            "excluded_samples": len(samples) - len(rows),
        }
        print(f"{split}: selected {len(rows)}/{len(samples)}")
        if split in {"train", "val"} and not rows:
            raise RuntimeError(
                f"No {split} samples passed the state-alignment threshold. "
                "Inspect candidate errors or lower --minimum-score."
            )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
