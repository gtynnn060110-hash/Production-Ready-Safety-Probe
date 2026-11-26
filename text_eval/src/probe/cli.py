"""One small command-line entry point for all four stages."""

import argparse
import logging
from pathlib import Path

import yaml

from .config import load_config
from .pipeline import STAGES, plan, run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Frozen-backbone safety-probe experiments")
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument(
        "--config", type=Path, action="append", help="YAML file; repeat for overlays"
    )
    parser.add_argument(
        "--set", dest="overrides", action="append", help="Override a dotted key=value"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate and print plan; no I/O or GPU"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for namespace in ("httpx", "httpx2"):
        logging.getLogger(namespace).setLevel(logging.WARNING)
    try:
        config = load_config(args.config or [Path("configs/base.yaml")], args.overrides)
        if args.dry_run:
            print(yaml.safe_dump(plan(args.stage, config), sort_keys=False, allow_unicode=True))
            return 0
        run(args.stage, config)
    except (OSError, ValueError, NotImplementedError, yaml.YAMLError) as exc:
        logging.error("%s", exc)
        return 1
    return 0
