#!/usr/bin/env python3
"""Convert existing CPS SFT parquet generation structs to plain response text."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def extract_text(value: object) -> str:
    if isinstance(value, dict):
        return str(value.get("content", ""))
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    args = parser.parse_args()

    for split in ("train", "val", "test"):
        path = args.data_dir / f"{split}.parquet"
        if not path.exists():
            print(f"skip missing: {path}")
            continue

        frame = pd.read_parquet(path)
        before = type(frame.iloc[0]["generation"]).__name__ if len(frame) else "empty"
        frame["generation"] = frame["generation"].map(extract_text)
        frame.to_parquet(path, index=False)
        after = type(pd.read_parquet(path).iloc[0]["generation"]).__name__ if len(frame) else "empty"
        print(f"{path}: {len(frame)} rows, generation {before} -> {after}")


if __name__ == "__main__":
    main()
