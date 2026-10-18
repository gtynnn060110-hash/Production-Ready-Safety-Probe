"""Independent response-only token objectives; no sequence-pooling supervision."""

from ..config import ComponentConfig
from ..contracts import Loss


def build_loss(config: ComponentConfig) -> Loss:
    from .token import PlugGuardATC, SingProbeWeighted, TokenBCE

    classes = {
        "bce": TokenBCE,
        "singprobe_weighted": SingProbeWeighted,
        "plugguard_atc": PlugGuardATC,
    }
    if config.name not in classes:
        raise ValueError(f"Unknown loss: {config.name}")
    try:
        return classes[config.name](**config.params)
    except TypeError as exc:
        raise ValueError(f"Invalid {config.name} parameters: {exc}") from exc
