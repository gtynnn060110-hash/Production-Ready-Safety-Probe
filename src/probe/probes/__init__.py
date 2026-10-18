"""Three small architectures with a shared per-position logit contract."""

from ..config import ComponentConfig
from ..contracts import Probe


def build_probe(config: ComponentConfig, *, input_dim: int, num_heads: int) -> Probe:
    from .gemini import GeminiProbe
    from .linear import LinearProbe
    from .singprobe import SingProbe

    classes = {"linear": LinearProbe, "singprobe": SingProbe, "gemini": GeminiProbe}
    if config.name not in classes:
        raise ValueError(f"Unknown probe: {config.name}")
    try:
        return classes[config.name](input_dim=input_dim, num_heads=num_heads, **config.params)
    except TypeError as exc:
        raise ValueError(f"Invalid {config.name} parameters: {exc}") from exc
