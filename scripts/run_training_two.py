#!/usr/bin/env python3
"""Run the complete small-sample state-first SFT training-two pipeline."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


def load_server_config(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing server-only DSV4Pro config: {path}")
    spec = importlib.util.spec_from_file_location("dsv4pro_server_config", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load DSV4Pro config: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = {}
    for name in ("DSV4PRO_API_KEY", "DSV4PRO_BASE_URL", "DSV4PRO_MODEL"):
        value = getattr(module, name, None)
        if not isinstance(value, str) or not value.strip():
            raise RuntimeError(f"{path} must define non-empty {name}")
        result[name] = value
    optional = {
        "REASONING_EFFORT": "DSV4PRO_REASONING_EFFORT",
        "THINKING_ENABLED": "DSV4PRO_THINKING_ENABLED",
    }
    for config_name, env_name in optional.items():
        value = getattr(module, config_name, None)
        if value is not None:
            result[env_name] = str(value).lower() if isinstance(value, bool) else str(value)
    return result


def run(command: list[str], *, cwd: Path, env: dict[str, str], dry_run: bool) -> None:
    print("\n$", " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=cwd, env=env, check=True)


def latest_checkpoint(output_dir: Path) -> int | None:
    tracker = output_dir / "latest_checkpointed_iteration.txt"
    if not tracker.exists():
        return None
    return int(tracker.read_text().strip())


def build_commands(
    *,
    workspace: Path,
    candidates_per_sample: int,
    minimum_score: float,
    limit: int,
) -> list[list[str]]:
    repo = workspace / "humanlm-main"
    raw_data = workspace / "raw_data/team_bundles_atc21s_full/team_bundles_atc21s_full"
    return [
        [
            sys.executable,
            str(repo / "scripts/prepare_cps_humanlm_data.py"),
            "--input-dir",
            str(raw_data),
            "--output-dir",
            str(repo / "data/cps_humanlm/v1/20p"),
            "--max-samples-per-team",
            "20",
            "--context-window",
            "80",
        ],
        [
            "sbatch",
            "--wait",
            f"--export=ALL,CANDIDATES_PER_SAMPLE={candidates_per_sample},LIMIT={limit}",
            str(repo / "scripts/generate_cps_state_candidates_a6000.slurm"),
        ],
        ["bash", str(repo / "scripts/judge_and_build_cps_state_first_sft.sh")],
        ["sbatch", "--wait", str(repo / "scripts/train_cps_state_first_sft_a6000.slurm")],
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.home() / "workspace/hlf_workspace",
    )
    parser.add_argument("--candidates-per-sample", default=3, type=int)
    parser.add_argument("--minimum-score", default=0.5, type=float)
    parser.add_argument("--limit", default=0, type=int)
    parser.add_argument(
        "--config",
        type=Path,
        help="Server-only dsv4pro_config.py; defaults to WORKSPACE/dsv4pro_config.py",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    workspace = args.workspace.expanduser().resolve()
    repo = workspace / "humanlm-main"
    config_path = (args.config or workspace / "dsv4pro_config.py").expanduser().resolve()
    if args.candidates_per_sample < 1:
        raise ValueError("--candidates-per-sample must be at least 1")
    if not 0 <= args.minimum_score <= 1:
        raise ValueError("--minimum-score must be between 0 and 1")
    if not repo.exists():
        raise FileNotFoundError(f"Missing repository: {repo}")

    env = os.environ.copy()
    env.update(load_server_config(config_path))
    env["MINIMUM_SCORE"] = str(args.minimum_score)
    (workspace / "logs").mkdir(parents=True, exist_ok=True)

    commands = build_commands(
        workspace=workspace,
        candidates_per_sample=args.candidates_per_sample,
        minimum_score=args.minimum_score,
        limit=args.limit,
    )
    stage_names = [
        "Prepare leakage-safe state-first source data",
        "Generate Qwen3-8B latent-state candidates on a GPU node",
        "Judge candidates with DSV4Pro and build SFT parquet files",
        "Train state-first SFT from the original Qwen3-8B base model",
    ]
    for index, (stage, command) in enumerate(zip(stage_names, commands), start=1):
        print(f"\n=== Training Two Stage {index}/4: {stage} ===", flush=True)
        run(command, cwd=workspace, env=env, dry_run=args.dry_run)

    if args.dry_run:
        print("\nDry run complete; no commands were executed.")
        return

    manifest_path = repo / "data/cps_state_first_sft_base_qwen/20p/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    output_dir = workspace / "humanlm_outputs/cps_qwen3_8b_base_state_first_sft_smoke"
    checkpoint = latest_checkpoint(output_dir)
    if checkpoint is None:
        raise RuntimeError(f"Training finished without a checkpoint tracker: {output_dir}")

    print("\n=== Training Two Complete ===")
    print(f"Selected data manifest: {manifest_path}")
    print(f"Selected samples: {manifest['splits']}")
    print(f"Latest checkpoint: global_step_{checkpoint}")
    print(f"Training output: {output_dir}")


if __name__ == "__main__":
    main()
