#!/usr/bin/env python3
"""Merge consecutive utterance fragments from the same CPS participant."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
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
INCOMPLETE_ENDERS = {
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
    "then",
    "to",
    "with",
}
FRAGMENT_ENDERS = INCOMPLETE_ENDERS | {
    "a",
    "do",
    "go",
    "just",
    "so",
    "the",
    "uh",
    "um",
}
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REVIEW_FLAGS = ROOT / "configs/cps_merge_review_flags.json"


def is_speech(event: dict) -> bool:
    return event.get("verb") == "says"


def join_utterances(parts: list[str]) -> str:
    return " ".join(part.strip() for part in parts if part.strip())


def clean_utterance_text(text: str) -> str:
    text = re.sub(r"\s+([,.;:!?])", r"\1", text.strip())
    return re.sub(r"\s{2,}", " ", text)


def is_likely_incomplete_fragment(text: str) -> bool:
    stripped = clean_utterance_text(text).lower()
    words = re.findall(r"[a-z0-9']+", stripped)
    if not words:
        return True
    return words[-1] in FRAGMENT_ENDERS


def review_key(team_no: object, event: dict) -> str:
    payload = json.dumps(
        {
            "team_no": team_no,
            "attempt_no": event.get("attempt_no"),
            "turn_no": event.get("turn_no"),
            "subject": event.get("subject"),
            "parts": event.get("merged_utterance_parts", [event.get("object", "")]),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def load_review_flags(path: Path = DEFAULT_REVIEW_FLAGS) -> set[str]:
    if not path.exists():
        return set()
    return set(json.loads(path.read_text(encoding="utf-8"))["needs_review"])


def is_continuation_fragment(text: str) -> bool:
    words = text.strip().lower().split()
    return bool(words) and len(words) <= 5 and words[0].strip(".,!?") in CONTINUATION_STARTERS


def is_incomplete_utterance(text: str) -> bool:
    stripped = text.strip().lower()
    words = stripped.split()
    if not words:
        return False
    final_word = words[-1].strip(".,!?")
    return (
        final_word in INCOMPLETE_ENDERS
        or stripped[-1] not in ".!?"
    )


def merge_consecutive_utterances(
    events: list[dict],
    team_no: object = None,
    review_flags: set[str] | None = None,
) -> tuple[list[dict], dict[str, int]]:
    """Merge consecutive speech and short continuation fragments."""
    merged_events: list[dict] = []
    merged_fragments = 0
    merged_groups = 0
    current_turn: tuple[object, object] | None = None
    last_speech_index: dict[str, int] = {}

    for original_event in events:
        event = copy.deepcopy(original_event)
        turn = (event.get("attempt_no"), event.get("turn_no"))
        if turn != current_turn:
            current_turn = turn
            last_speech_index.clear()

        speaker = event.get("subject")
        if not is_speech(event):
            merged_events.append(event)
            continue
        if speaker not in HUMAN_PARTICIPANTS:
            merged_events.append(event)
            last_speech_index.clear()
            continue

        text = str(event.get("object", "")).strip()
        previous_index = last_speech_index.get(str(speaker))
        previous_text = (
            str(merged_events[previous_index].get("object", ""))
            if previous_index is not None
            else ""
        )
        should_merge = previous_index is not None and (
            is_continuation_fragment(text) or is_incomplete_utterance(previous_text)
        )
        if should_merge:
            previous = merged_events[previous_index]
            previous.setdefault(
                "merged_utterance_parts",
                [str(previous.get("object", "")).strip()],
            ).append(text)
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
            event["merged_utterance_parts"] = [text]
            merged_events.append(event)
            last_speech_index[str(speaker)] = len(merged_events) - 1

    incomplete_fragments = 0
    flags = load_review_flags() if review_flags is None else review_flags
    for event in merged_events:
        if is_speech(event) and event.get("subject") in HUMAN_PARTICIPANTS:
            event["object"] = clean_utterance_text(str(event.get("object", "")))
            human_review_flag = review_key(team_no, event) in flags
            event["human_review_status"] = (
                "needs_review" if human_review_flag else "not_flagged"
            )
            event["is_incomplete_fragment"] = (
                human_review_flag
                or is_likely_incomplete_fragment(str(event.get("object", "")))
            )
            incomplete_fragments += int(event["is_incomplete_fragment"])

    return merged_events, {
        "input_events": len(events),
        "output_events": len(merged_events),
        "merged_groups": merged_groups,
        "merged_fragments": merged_fragments,
        "incomplete_fragments": incomplete_fragments,
    }


def merge_bundle(bundle: dict, review_flags: set[str] | None = None) -> tuple[dict, dict[str, int]]:
    merged_bundle = copy.deepcopy(bundle)
    source_key = "annotated_corpus" if bundle.get("annotated_corpus") else "corpus"
    merged, stats = merge_consecutive_utterances(
        bundle.get(source_key, []),
        team_no=bundle.get("team_no"),
        review_flags=review_flags,
    )
    merged_bundle[source_key] = merged
    return merged_bundle, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--audit-output", type=Path)
    args = parser.parse_args()

    input_files = sorted(args.input_dir.rglob("team_*_bundle_atc21s_full.json"))
    if not input_files:
        raise FileNotFoundError(
            f"No team bundle JSON files found under: {args.input_dir}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    review_flags = load_review_flags()
    totals = {
        "input_events": 0,
        "output_events": 0,
        "merged_groups": 0,
        "merged_fragments": 0,
        "incomplete_fragments": 0,
    }
    audit_rows = []

    for input_path in input_files:
        merged_bundle, stats = merge_bundle(
            json.loads(input_path.read_text()),
            review_flags=review_flags,
        )
        output_path = args.output_dir / input_path.name
        output_path.write_text(
            json.dumps(merged_bundle, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        source_key = "annotated_corpus" if merged_bundle.get("annotated_corpus") else "corpus"
        audit_rows.extend(
            {
                "team_no": merged_bundle.get("team_no"),
                "attempt_no": event.get("attempt_no"),
                "turn_no": event.get("turn_no"),
                "subject": event.get("subject"),
                "merged_utterance_parts": event.get("merged_utterance_parts"),
                "merged_utterance": event.get("object"),
                "is_incomplete_fragment": event.get("is_incomplete_fragment", False),
                "training_eligible": not event.get("is_incomplete_fragment", False),
            }
            for event in merged_bundle[source_key]
            if event.get("merged_utterance_count", 1) > 1
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
        f"{totals['input_events']} -> {totals['output_events']} events, "
        f"{totals['incomplete_fragments']} incomplete speech events"
    )
    if args.audit_output:
        args.audit_output.parent.mkdir(parents=True, exist_ok=True)
        args.audit_output.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in audit_rows),
            encoding="utf-8",
        )
        print(f"Audit: {len(audit_rows)} rows -> {args.audit_output}")


if __name__ == "__main__":
    main()
