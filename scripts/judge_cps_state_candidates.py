#!/usr/bin/env python3
"""Judge existing Qwen state candidates from an internet-connected login node."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
from pathlib import Path

try:
    from .run_cps_dsv4pro_alignment import (
        call_chat_completion,
        judge_messages,
        sample_key,
        validate_judge_output,
        validate_qwen_rollout,
    )
except ImportError:
    from run_cps_dsv4pro_alignment import (
        call_chat_completion,
        judge_messages,
        sample_key,
        validate_judge_output,
        validate_qwen_rollout,
    )


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout", default=180, type=int)
    parser.add_argument("--sleep-seconds", default=0.5, type=float)
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("DSV4PRO_API_KEY")
    base_url = os.environ.get("DSV4PRO_BASE_URL")
    model = os.environ.get("DSV4PRO_MODEL")
    missing = [
        name
        for name, value in {
            "DSV4PRO_API_KEY": api_key,
            "DSV4PRO_BASE_URL": base_url,
            "DSV4PRO_MODEL": model,
        }.items()
        if not value
    ]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

    samples = {sample_key(row): row for row in load_jsonl(args.source)}
    candidates = load_jsonl(args.candidates)
    completed: set[tuple[str, int]] = set()
    if args.append and args.output.exists():
        for row in load_jsonl(args.output):
            if not row.get("error") and row.get("state_alignment"):
                completed.add((row["sample_key"], int(row["candidate_index"])))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append else "w"
    with args.output.open(mode, encoding="utf-8") as output:
        for index, candidate in enumerate(candidates, start=1):
            key = candidate["sample_key"]
            candidate_key = (key, int(candidate["candidate_index"]))
            if candidate_key in completed:
                print(f"{index}/{len(candidates)} skip {key} candidate {candidate_key[1]}")
                continue
            result = dict(candidate)
            result.pop("error", None)
            try:
                rollout = candidate["qwen_rollout"]
                validate_qwen_rollout(rollout)
                judgment = call_chat_completion(
                    base_url=str(base_url),
                    api_key=str(api_key),
                    model=str(model),
                    messages=judge_messages(samples[key], rollout),
                    timeout=args.timeout,
                )
                validate_judge_output(judgment)
                result["state_alignment"] = judgment
            except (
                urllib.error.URLError,
                TimeoutError,
                ValueError,
                KeyError,
            ) as error:
                result["state_alignment"] = None
                result["error"] = f"{type(error).__name__}: {error}"
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()
            print(f"{index}/{len(candidates)} {key} candidate {candidate_key[1]}")
            time.sleep(args.sleep_seconds)


if __name__ == "__main__":
    main()
