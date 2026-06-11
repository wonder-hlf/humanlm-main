#!/usr/bin/env python3
"""Generate CPS latent states and judge alignment with a configurable DSV4Pro API."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"Model did not return a JSON object: {text[:200]}")
    return json.loads(text[start : end + 1])


def call_chat_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict],
    timeout: int,
) -> dict:
    endpoint = base_url.rstrip("/")
    if not endpoint.endswith("/chat/completions"):
        endpoint += "/chat/completions"
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
    ).encode()
    request = urllib.request.Request(
        endpoint,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read())
    return extract_json(body["choices"][0]["message"]["content"])


def context_payload(sample: dict) -> dict:
    return {
        "student_profile": sample["student_profile"],
        "role_state": sample["role_state"],
        "environment_state": sample["environment_state"],
        "dialogue_and_action_history": sample["dialogue_and_action_history"],
        "state_dimensions": sample["state_dimensions"],
    }


def latent_messages(sample: dict) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "Infer the target student's current latent CPS state using only the "
                "provided prefix context. Never use or predict final task outcomes. "
                "Return JSON with exactly six keys: belief, goal, value, stance, "
                "emotion, communication. Each value must be a concise string."
            ),
        },
        {"role": "user", "content": json.dumps(context_payload(sample), ensure_ascii=False)},
    ]


def judge_messages(sample: dict, latent_states: dict) -> list[dict]:
    payload = context_payload(sample)
    payload["ground_truth"] = sample["ground_truth"]
    payload["generated_latent_states"] = latent_states
    payload["computable_state_proxies"] = sample["computable_state_proxies"]
    return [
        {
            "role": "system",
            "content": (
                "Judge how well each generated latent state aligns with the human's "
                "ground-truth next utterance and nearby action. Return JSON keyed by "
                "task_understanding, strategy_goal, collaboration_value, "
                "interaction_stance, error_repair_state, communication_style. Each "
                "value must contain score (number from 0 to 1) and rationale (string). "
                "Do not reward claims that rely on future information."
            ),
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", default=0, type=int)
    parser.add_argument("--timeout", default=180, type=int)
    parser.add_argument("--sleep-seconds", default=0.5, type=float)
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

    rows = load_jsonl(args.input)
    if args.limit > 0:
        rows = rows[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as output:
        for index, sample in enumerate(rows, start=1):
            try:
                latent_states = call_chat_completion(
                    base_url=str(base_url),
                    api_key=str(api_key),
                    model=str(model),
                    messages=latent_messages(sample),
                    timeout=args.timeout,
                )
                alignment = call_chat_completion(
                    base_url=str(base_url),
                    api_key=str(api_key),
                    model=str(model),
                    messages=judge_messages(sample, latent_states),
                    timeout=args.timeout,
                )
                result = {
                    "team_no": sample["team_no"],
                    "participant": sample["participant"],
                    "source_index": sample["source_index"],
                    "latent_states": latent_states,
                    "state_alignment": alignment,
                }
            except (urllib.error.URLError, TimeoutError, ValueError, KeyError) as error:
                result = {
                    "team_no": sample["team_no"],
                    "participant": sample["participant"],
                    "source_index": sample["source_index"],
                    "error": f"{type(error).__name__}: {error}",
                }
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()
            print(f"{index}/{len(rows)}")
            time.sleep(args.sleep_seconds)


if __name__ == "__main__":
    main()
