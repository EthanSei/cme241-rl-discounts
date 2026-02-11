"""Parameter schema and YAML helpers for calibrated MDP parameters."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class CategoryParams:
    """Per-category parameter bundle."""

    name: str
    price: float
    beta_0: float


@dataclass(frozen=True)
class MDPParams:
    """Top-level calibrated MDP parameter bundle."""

    delta: float
    gamma: float
    alpha: float
    beta_p: float
    beta_l: float
    beta_m: float
    eta: float
    c0: float
    categories: tuple[CategoryParams, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert parameter object to a plain dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MDPParams":
        """Create parameter object from a plain dict."""
        categories = tuple(
            CategoryParams(**category_payload)
            for category_payload in payload["categories"]
        )
        metadata = payload.get("metadata", {})
        return cls(
            delta=float(payload["delta"]),
            gamma=float(payload["gamma"]),
            alpha=float(payload["alpha"]),
            beta_p=float(payload["beta_p"]),
            beta_l=float(payload["beta_l"]),
            beta_m=float(payload["beta_m"]),
            eta=float(payload["eta"]),
            c0=float(payload["c0"]),
            categories=categories,
            metadata=dict(metadata),
        )


def save_mdp_params(params: MDPParams, output_path: Path) -> None:
    """Serialize parameters to YAML."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(params.to_dict(), sort_keys=False))


def load_mdp_params(path: Path) -> MDPParams:
    """Load parameters from YAML."""
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("Expected a mapping in MDP parameter YAML.")
    return MDPParams.from_dict(payload)
