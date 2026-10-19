"""Deterministic stage paths; computing them never creates files."""

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

from .config import Config
from .contracts import HEADS

DATA_VERSION = 3  # per-head annotation majority (ties safe), then filtering/grouped splitting


def _key(value: dict) -> str:
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()[:12]


@dataclass(frozen=True)
class Artifacts:
    prepared: Path
    features: Path
    run: Path

    @classmethod
    def from_config(cls, config: Config) -> "Artifacts":
        root = Path(config.output_dir).expanduser().resolve()
        data_key = _key({"data": asdict(config.data), "heads": HEADS, "version": DATA_VERSION})
        feature_key = _key({"data": data_key, "features": asdict(config.features)})
        run_key = _key(
            {
                "features": feature_key,
                "probe": asdict(config.probe),
                "loss": asdict(config.loss),
                "training": asdict(config.training) | {"resume": True},
                "seed": config.seed,
            }
        )
        return cls(
            root / "prepared" / data_key,
            root / "features" / feature_key,
            root / "runs" / f"{config.name}-{run_key}",
        )
