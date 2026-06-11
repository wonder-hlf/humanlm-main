#!/usr/bin/env python3
"""Merge consecutive utterance fragments from the same CPS participant."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


HUMAN_PARTICIPANTS = {"A", "B"}
CONTINUATION_STARTERS = {
    "and",
    "at",
    "because",
    "but",
    "for",
    "from",
    "in",
    "of",
    "on",
    "or",
    "than",
    "that",
    "then",
    "to",
    "which",
    "with",
}


def is_speech(event: dict) -> bool:
    return event.get("verb") == "says"


def join_utterances(parts: list[str]) -> str:
    return " ".join(part.strip() for part in parts if part.strip())


def is_continuation_fragment(text: str) -> bool:
    words = text.strip().lower().split()
    return bool(words) and len(words) <= 8 and words[0].strip(".,!?") in CONTINUATION_STARTERS


def merge_consecutive_utterances(events: list[dict]) -> tuple[list[dict], dict[str, int]]:
    """Merge consecutive speech and short continuation fragments."""
    merged_events: list[dict] = []
    merged_fragments = 0
    merged_groups = 0
    current_turn: tuple[object, object] | None = None
    last_human_speaker: str | None = None
    last_speech_index: dict[str, int] = {}

    for original_event in events:
        event = copy.deepcopy(original_event)
        turn = (event.get("attempt_no"), event.get("turn_no"))
        if turn != current_turn:
            current_turn = turn
            last_human_speaker = None
            last_speech_index.clear()

        speaker = event.get("subject")
        if not is_speech(event):
            merged_events.append(event)
            continue
        if speaker not in HUMAN_PARTICIPANTS:
            merged_events.append(event)
            last_human_speaker = None
            last_speech_index.clear()
            continue

        text = str(event.get("object", "")).strip()
        previous_index = last_speech_index.get(str(speaker))
        should_merge = previous_index is not None and (
            last_human_speaker == speaker or is_continuation_fragment(text)
        )
        if should_merge:
            previous = merged_events[previous_index]
            previous["object"] = join_utterances(
                [str(previous.get("object", "")), text]
            )
            previous_count = int(previous.get("merged_utterance_count", 1))
            previous["merged_utterance_count"] = previous_count + 1
            merged_fragments += 1
            if previous_count == 1:
                merged_groups += 1
        else:
            event["merged_utterance_count"] = 1
            merged_events.append(event)
            last_speech_index[str(speaker)] = len(merged_events) - 1

        last_human_speaker = str(speaker)

    return merged_events, {
        "input_events": len(events),
        "output_events": len(merged_events),
        "merged_groups": merged_groups,
        "merged_fragments": merged_fragments,
    }


def merge_bundle(bundle: dict) -> tuple[dict, dict[str, int]]:
    merged_bundle = copy.deepcopy(bundle)
    source_key = "annotated_corpus" if bundle.get("annotated_corpus") else "corpus"
    merged, stats = merge_consecutive_utterances(bundle.get(source_key, []))
    merged_bundle[source_key] = merged
    return merged_bundle, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    input_files = sorted(args.input_dir.rglob("team_*_bundle_atc21s_full.json"))
    if not input_files:
        raise FileNotFoundError(
            f"No team bundle JSON files found under: {args.input_dir}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    totals = {"input_events": 0, "output_events": 0, "merged_groups": 0, "merged_fragments": 0}

    for input_path in input_files:
        merged_bundle, stats = merge_bundle(json.loads(input_path.read_text()))
        output_path = args.output_dir / input_path.name
        output_path.write_text(
            json.dumps(merged_bundle, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        for key in totals:
            totals[key] += stats[key]
        print(
            f"{input_path.name}: {stats['merged_groups']} groups, "
            f"{stats['merged_fragments']} fragments merged"
        )

    print(
        f"Total: {totals['merged_groups']} groups, "
        f"{totals['merged_fragments']} fragments merged, "
        f"{totals['input_events']} -> {totals['output_events']} events"
    )


if __name__ == "__main__":
    main()
