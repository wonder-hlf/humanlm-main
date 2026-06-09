#!/usr/bin/env python3
"""Build HumanLM-compatible SFT parquet files from CPS team bundles."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd


SYSTEM_PROMPT = """You are simulating a real human participant in a collaborative problem-solving task.

The participant works with a teammate to build train tracks. They can see only part of the task state, talk with the teammate, and perform actions such as adding/removing tracks, pressing help, or submitting. Given the conversation and action history, produce the next utterance for the target participant.

Participant profile:
{persona}

Respond naturally as HUMAN. Output only the next utterance."""


def event_to_text(event: dict) -> str:
    subject = str(event.get("subject", "UNK"))
    verb = str(event.get("verb", "acts"))
    obj = str(event.get("object", "")).strip()
    turn = event.get("turn_no")
    attempt = event.get("attempt_no")
    prefix = f"[attempt {attempt}, turn {turn}] {subject}"
    if verb == "says":
        return f'{prefix} says: "{obj}"'
    return f"{prefix} {verb}: {obj}"


def make_persona(team_no: int, participant: str, bundle: dict) -> str:
    learning = bundle.get("learning_features", {})
    log_features = bundle.get("log_features", {})
    participant_prefix = f"{participant}_"

    participant_features = {
        key[len(participant_prefix) :]: value
        for key, value in {**log_features, **learning}.items()
        if key.startswith(participant_prefix)
    }

    lines = [
        f"Team: {team_no}",
        f"Participant: {participant}",
        "Role: A human collaborator in a two-person team task.",
        "Communication style: infer from the dialogue history and continue consistently.",
    ]
    if participant_features:
        compact = ", ".join(f"{k}={v}" for k, v in sorted(participant_features.items()))
        lines.append(f"Observed participant features: {compact}")
    return "\n".join(lines)


def make_sample(bundle: dict, idx: int, context_window: int) -> dict | None:
    event = bundle["annotated_corpus"][idx]
    target = event.get("subject")
    if target not in {"A", "B"} or event.get("verb") != "says":
        return None

    response = str(event.get("object", "")).strip()
    if not response:
        return None

    team_no = bundle.get("team_no", "unknown")
    start = max(0, idx - context_window)
    history = bundle["annotated_corpus"][start:idx]
    context = "\n".join(event_to_text(item) for item in history)
    if not context:
        context = "The task has just started."

    user_content = (
        f"Task: collaborative train-track problem solving.\n"
        f"Target participant: {target} (refer to this participant as HUMAN).\n"
        f"Team number: {team_no}.\n\n"
        f"Recent history:\n{context}\n\n"
        f"Now generate HUMAN's next utterance."
    )

    return {
        "prompt": [
            {
                "role": "system",
                "name": "",
                "content": SYSTEM_PROMPT.format(
                    persona=make_persona(int(team_no), str(target), bundle)
                ),
            },
            {"role": "user", "name": "", "content": user_content},
        ],
        "generation": {"role": "user", "name": "HUMAN", "content": response},
        "team_no": int(team_no),
        "target": str(target),
        "attempt_no": event.get("attempt_no"),
        "turn_no": event.get("turn_no"),
        "source_index": idx,
    }


def split_samples(samples: list[dict], seed: int) -> tuple[dict[str, list[dict]], dict[str, list[int]]]:
    rng = random.Random(seed)
    team_ids = sorted({row["team_no"] for row in samples})
    rng.shuffle(team_ids)
    if len(team_ids) < 3:
        raise ValueError("At least three teams are required for team-held-out splits.")

    split_teams = {
        "train": sorted(team_ids[:-2]),
        "val": [team_ids[-2]],
        "test": [team_ids[-1]],
    }
    splits = {
        name: [row for row in samples if row["team_no"] in teams]
        for name, teams in split_teams.items()
    }
    return splits, split_teams


def write_split(rows: list[dict], path: Path) -> None:
    frame = pd.DataFrame(
        [{"prompt": row["prompt"], "generation": row["generation"]} for row in rows]
    )
    frame.to_parquet(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("data/cps_team_sft"), type=Path)
    parser.add_argument("--context-window", default=40, type=int)
    parser.add_argument("--max-samples-per-team", default=20, type=int)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    all_samples: list[dict] = []

    input_files = sorted(args.input_dir.rglob("team_*_bundle_atc21s_full.json"))
    if not input_files:
        raise FileNotFoundError(
            f"No team bundle JSON files found under: {args.input_dir}"
        )

    for file_path in input_files:
        bundle = json.loads(file_path.read_text())
        bundle["annotated_corpus"] = bundle.get("annotated_corpus") or bundle.get("corpus", [])
        team_samples = [
            sample
            for idx in range(len(bundle["annotated_corpus"]))
            if (sample := make_sample(bundle, idx, args.context_window)) is not None
        ]
        rng.shuffle(team_samples)
        if args.max_samples_per_team > 0:
            team_samples = team_samples[: args.max_samples_per_team]
        all_samples.extend(team_samples)
        print(f"{file_path.name}: {len(team_samples)} samples")

    splits, split_teams = split_samples(all_samples, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for split, rows in splits.items():
        write_split(rows, args.output_dir / f"{split}.parquet")
        preview_path = args.output_dir / f"{split}.preview.json"
        preview_path.write_text(
            json.dumps(rows[:5], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"{split}: {len(rows)} rows -> {args.output_dir / f'{split}.parquet'}")

    manifest = {
        "input_dir": str(args.input_dir),
        "context_window": args.context_window,
        "max_samples_per_team": args.max_samples_per_team,
        "seed": args.seed,
        "total_samples": len(all_samples),
        "splits": {name: len(rows) for name, rows in splits.items()},
        "split_teams": split_teams,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
