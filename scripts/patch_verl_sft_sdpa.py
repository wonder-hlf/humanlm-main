#!/usr/bin/env python3
"""Patch VERL 0.6.x SFT trainer to use PyTorch SDPA instead of FlashAttention2."""

from __future__ import annotations

import argparse
from pathlib import Path

import verl.trainer.fsdp_sft_trainer as trainer


FLASH = 'attn_implementation="flash_attention_2"'
SDPA = 'attn_implementation="sdpa"'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--restore-flash-attn",
        action="store_true",
        help="Restore the original FlashAttention2 setting.",
    )
    args = parser.parse_args()

    path = Path(trainer.__file__).resolve()
    text = path.read_text(encoding="utf-8")
    old, new = (SDPA, FLASH) if args.restore_flash_attn else (FLASH, SDPA)

    if new in text and old not in text:
        print(f"Already configured: {new}")
        print(path)
        return
    if old not in text:
        raise RuntimeError(f"Expected setting not found in {path}: {old}")

    backup = path.with_suffix(path.suffix + ".bak")
    if not backup.exists():
        backup.write_text(text, encoding="utf-8")

    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"Patched: {old} -> {new}")
    print(path)


if __name__ == "__main__":
    main()
