#!/usr/bin/env python3
"""Run Qwen HumanLM rollouts and judge their state alignment with DSV4Pro."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


STATE_KEYS = {"belief", "goal", "value", "stance", "emotion", "communication"}
JUDGE_DIMENSION_KEYS = {
    "task_understanding",
    "strategy_goal",
    "collaboration_value",
    "interaction_stance",
    "error_repair_state",
    "communication_style",
}
ACTION_TYPES = {"add_track", "remove_track", "load_solution", "check", "submit", "press"}
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GENERATION_PROMPT = ROOT / "prompts/state_generation_prompt.md"
DEFAULT_JUDGE_PROMPT = ROOT / "prompts/state_judge_prompt.md"


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


def validate_qwen_rollout(rollout: dict) -> None:
    states = rollout.get("latent_states")
    if not isinstance(states, dict) or set(states) != STATE_KEYS:
        raise ValueError(f"Qwen rollout must contain exactly six latent states: {STATE_KEYS}")
    if not all(isinstance(states[key], str) and states[key].strip() for key in STATE_KEYS):
        raise ValueError("Every Qwen latent state must be a non-empty string.")
    if not isinstance(rollout.get("utterance"), str) or not rollout["utterance"].strip():
        raise ValueError("Qwen rollout utterance must be a non-empty string.")
    action = rollout.get("action_intent")
    if action is not None and (
        not isinstance(action, dict) or action.get("type") not in ACTION_TYPES
    ):
        raise ValueError(f"Qwen action_intent must be null or one of: {ACTION_TYPES}")


def validate_judge_output(judgment: dict) -> None:
    scores = judgment.get("dimension_scores")
    if not isinstance(scores, dict) or set(scores) != JUDGE_DIMENSION_KEYS:
        raise ValueError(
            f"Judge output must contain exactly these dimension_scores: {JUDGE_DIMENSION_KEYS}"
        )
    for value in scores.values():
        score = value.get("score") if isinstance(value, dict) else None
        if not isinstance(score, (int, float)) or not 0 <= score <= 1:
            raise ValueError("Every judge dimension score must be between 0 and 1.")
    overall = judgment.get("overall_state_alignment")
    if not isinstance(overall, (int, float)) or not 0 <= overall <= 1:
        raise ValueError("Judge overall_state_alignment must be between 0 and 1.")
    for key in ("missing_state", "redundant_unsupported_state"):
        if not isinstance(judgment.get(key), list):
            raise ValueError(f"Judge output must contain list field: {key}")


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
        "optional_steering_config": sample.get("optional_steering_config"),
    }


def qwen_rollout_messages(sample: dict, prompt: str | None = None) -> list[dict]:
    return [
        {
            "role": "system",
            "content": prompt or DEFAULT_GENERATION_PROMPT.read_text(encoding="utf-8"),
        },
        {"role": "user", "content": json.dumps(context_payload(sample), ensure_ascii=False)},
    ]


def judge_messages(sample: dict, qwen_rollout: dict, prompt: str | None = None) -> list[dict]:
    payload = context_payload(sample)
    payload["ground_truth"] = sample["ground_truth"]
    payload["qwen_rollout"] = qwen_rollout
    payload["computable_state_proxies"] = sample["computable_state_proxies"]
    return [
        {
            "role": "system",
            "content": prompt or DEFAULT_JUDGE_PROMPT.read_text(encoding="utf-8"),
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
    parser.add_argument("--generation-prompt", default=DEFAULT_GENERATION_PROMPT, type=Path)
    parser.add_argument("--judge-prompt", default=DEFAULT_JUDGE_PROMPT, type=Path)
    args = parser.parse_args()

    qwen_api_key = os.environ.get("QWEN_API_KEY", "EMPTY")
    qwen_base_url = os.environ.get("QWEN_BASE_URL")
    qwen_model = os.environ.get("QWEN_MODEL")
    judge_api_key = os.environ.get("DSV4PRO_API_KEY")
    judge_base_url = os.environ.get("DSV4PRO_BASE_URL")
    judge_model = os.environ.get("DSV4PRO_MODEL")
    missing = [
        name
        for name, value in {
            "QWEN_BASE_URL": qwen_base_url,
            "QWEN_MODEL": qwen_model,
            "DSV4PRO_API_KEY": judge_api_key,
            "DSV4PRO_BASE_URL": judge_base_url,
            "DSV4PRO_MODEL": judge_model,
        }.items()
        if not value
    ]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

    rows = load_jsonl(args.input)
    if args.limit > 0:
        rows = rows[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    generation_prompt = args.generation_prompt.read_text(encoding="utf-8")
    judge_prompt = args.judge_prompt.read_text(encoding="utf-8")

    with args.output.open("w", encoding="utf-8") as output:
        for index, sample in enumerate(rows, start=1):
            try:
                qwen_rollout = call_chat_completion(
                    base_url=str(qwen_base_url),
                    api_key=str(qwen_api_key),
                    model=str(qwen_model),
                    messages=qwen_rollout_messages(sample, generation_prompt),
                    timeout=args.timeout,
                )
                validate_qwen_rollout(qwen_rollout)
                alignment = call_chat_completion(
                    base_url=str(judge_base_url),
                    api_key=str(judge_api_key),
                    model=str(judge_model),
                    messages=judge_messages(sample, qwen_rollout, judge_prompt),
                    timeout=args.timeout,
                )
                validate_judge_output(alignment)
                result = {
                    "team_no": sample["team_no"],
                    "participant": sample["participant"],
                    "source_index": sample["source_index"],
                    "qwen_rollout": qwen_rollout,
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
