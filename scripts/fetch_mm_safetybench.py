#!/usr/bin/env python3
"""Fetch the official MM-SafetyBench repository containing questions and eval code."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO_URL = "https://github.com/isXinLiu/MM-SafetyBench.git"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=ROOT / "data/raw/mm_safetybench/source")
    args = parser.parse_args()
    if args.target.exists():
        if (args.target / ".git").exists():
            subprocess.run(["git", "-C", str(args.target), "pull", "--ff-only"], check=True)
            return
        if any(args.target.iterdir()):
            raise SystemExit(f"Target exists and is not an official git checkout: {args.target}")
        args.target.rmdir()
    args.target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(args.target)], check=True)
    print(f"Official repository fetched to {args.target}")
    print("Fetch and unpack the official MM-SafetyBench(imgs).zip under data/imgs, then run prepare_mm_safetybench.py.")


if __name__ == "__main__":
    main()
