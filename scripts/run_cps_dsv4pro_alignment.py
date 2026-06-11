#!/usr/bin/env python3
"""Run Qwen HumanLM rollouts and judge their state alignment with DSV4Pro."""

from __future__ import annotations

import argparse
import json
import os
import random
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


class LocalQwenGenerator:
    def __init__(self, model_path: str, max_new_tokens: int):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
        )
        self.model.eval()

    def __call__(self, messages: list[dict], temperature: float, seed: int) -> dict:
        self.torch.manual_seed(seed)
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        generate_kwargs = {
            **inputs,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if temperature > 0:
            generate_kwargs["temperature"] = temperature
        with self.torch.inference_mode():
            output = self.model.generate(**generate_kwargs)
        generated = output[0, inputs["input_ids"].shape[1] :]
        return extract_json(self.tokenizer.decode(generated, skip_special_tokens=True))


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
    temperature: float = 0,
    seed: int | None = None,
) -> dict:
    endpoint = base_url.rstrip("/")
    if not endpoint.endswith("/chat/completions"):
        endpoint += "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }
    reasoning_effort = os.environ.get("DSV4PRO_REASONING_EFFORT")
    thinking_enabled = os.environ.get("DSV4PRO_THINKING_ENABLED", "").lower()
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
    if thinking_enabled in {"1", "true", "yes", "enabled"}:
        payload["thinking"] = {"type": "enabled"}
    if seed is not None:
        payload["seed"] = seed
    def send(request_payload: dict) -> dict:
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(request_payload).encode(),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())

    try:
        body = send(payload)
    except urllib.error.HTTPError as error:
        if error.code not in {400, 422}:
            raise
        fallback = dict(payload)
        fallback.pop("response_format", None)
        fallback.pop("seed", None)
        fallback.pop("reasoning_effort", None)
        fallback.pop("thinking", None)
        body = send(fallback)
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


def sample_key(sample: dict) -> str:
    return ":".join(
        str(sample[key])
        for key in ("team_no", "participant", "attempt_no", "turn_no", "source_index")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", default=0, type=int)
    parser.add_argument("--timeout", default=180, type=int)
    parser.add_argument("--sleep-seconds", default=0.5, type=float)
    parser.add_argument("--candidates-per-sample", default=1, type=int)
    parser.add_argument("--qwen-temperature", default=0.7, type=float)
    parser.add_argument(
        "--qwen-local-model",
        type=Path,
        help="Load Qwen locally instead of calling QWEN_BASE_URL.",
    )
    parser.add_argument("--qwen-max-new-tokens", default=1024, type=int)
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Generate Qwen candidates without calling the judge.",
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append and skip candidate keys already present in the output.",
    )
    parser.add_argument("--generation-prompt", default=DEFAULT_GENERATION_PROMPT, type=Path)
    parser.add_argument("--judge-prompt", default=DEFAULT_JUDGE_PROMPT, type=Path)
    args = parser.parse_args()
    if args.candidates_per_sample < 1:
        raise ValueError("--candidates-per-sample must be at least 1")

    qwen_api_key = os.environ.get("QWEN_API_KEY", "EMPTY")
    qwen_base_url = os.environ.get("QWEN_BASE_URL")
    qwen_model = os.environ.get("QWEN_MODEL")
    judge_api_key = os.environ.get("DSV4PRO_API_KEY")
    judge_base_url = os.environ.get("DSV4PRO_BASE_URL")
    judge_model = os.environ.get("DSV4PRO_MODEL")
    missing = [
        name
        for name, value in {
            "DSV4PRO_API_KEY": judge_api_key,
            "DSV4PRO_BASE_URL": judge_base_url,
            "DSV4PRO_MODEL": judge_model,
        }.items()
        if not value and not args.generate_only
    ]
    if not args.qwen_local_model:
        if not qwen_base_url:
            missing.append("QWEN_BASE_URL")
        if not qwen_model:
            missing.append("QWEN_MODEL")
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")
    local_qwen = (
        LocalQwenGenerator(str(args.qwen_local_model), args.qwen_max_new_tokens)
        if args.qwen_local_model
        else None
    )

    rows = load_jsonl(args.input)
    if args.limit > 0:
        rows = rows[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    generation_prompt = args.generation_prompt.read_text(encoding="utf-8")
    judge_prompt = args.judge_prompt.read_text(encoding="utf-8")
    completed: set[tuple[str, int]] = set()
    attempt_counts: dict[tuple[str, int], int] = {}
    if args.append and args.output.exists():
        for row in load_jsonl(args.output):
            if "sample_key" in row and "candidate_index" in row:
                candidate_key = (row["sample_key"], int(row["candidate_index"]))
                attempt_counts[candidate_key] = attempt_counts.get(candidate_key, 0) + 1
                if not row.get("error"):
                    completed.add(candidate_key)
    mode = "a" if args.append else "w"
    rng = random.Random(args.seed)
    total = len(rows) * args.candidates_per_sample
    processed = 0

    with args.output.open(mode, encoding="utf-8") as output:
        for sample in rows:
            key = sample_key(sample)
            for candidate_index in range(args.candidates_per_sample):
                processed += 1
                if (key, candidate_index) in completed:
                    print(f"{processed}/{total} skip {key} candidate {candidate_index}")
                    continue
                result = {
                    "sample_key": key,
                    "candidate_index": candidate_index,
                    "team_no": sample["team_no"],
                    "participant": sample["participant"],
                    "attempt_no": sample["attempt_no"],
                    "turn_no": sample["turn_no"],
                    "source_index": sample["source_index"],
                }
                qwen_rollout = None
                alignment = None
                error = None
                qwen_seed = (
                    rng.randrange(0, 2**31) + attempt_counts.get((key, candidate_index), 0)
                ) % (2**31)
                try:
                    messages = qwen_rollout_messages(sample, generation_prompt)
                    qwen_rollout = (
                        local_qwen(messages, args.qwen_temperature, qwen_seed)
                        if local_qwen
                        else call_chat_completion(
                            base_url=str(qwen_base_url),
                            api_key=str(qwen_api_key),
                            model=str(qwen_model),
                            messages=messages,
                            timeout=args.timeout,
                            temperature=args.qwen_temperature,
                            seed=qwen_seed,
                        )
                    )
                    validate_qwen_rollout(qwen_rollout)
                    if not args.generate_only:
                        alignment = call_chat_completion(
                            base_url=str(judge_base_url),
                            api_key=str(judge_api_key),
                            model=str(judge_model),
                            messages=judge_messages(sample, qwen_rollout, judge_prompt),
                            timeout=args.timeout,
                        )
                        validate_judge_output(alignment)
                except (urllib.error.URLError, TimeoutError, ValueError, KeyError) as caught:
                    error = f"{type(caught).__name__}: {caught}"
                result.update(
                    {
                        "qwen_seed": qwen_seed,
                        "qwen_rollout": qwen_rollout,
                        "state_alignment": alignment,
                    }
                )
                if error:
                    result["error"] = error
                output.write(json.dumps(result, ensure_ascii=False) + "\n")
                output.flush()
                print(f"{processed}/{total} {key} candidate {candidate_index}")
                time.sleep(args.sleep_seconds)


if __name__ == "__main__":
    main()
