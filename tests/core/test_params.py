"""Tests for MDP parameter schema and YAML serialization."""

from __future__ import annotations

from pathlib import Path

from discount_engine.core.params import CategoryParams, MDPParams, load_mdp_params, save_mdp_params


def test_params_roundtrip_yaml(tmp_path: Path) -> None:
    params = MDPParams(
        delta=0.30,
        gamma=0.99,
        alpha=0.8,
        beta_p=1.1,
        beta_l=0.2,
        beta_m=0.3,
        eta=0.05,
        c0=0.1,
        categories=(
            CategoryParams(name="A", price=10.0, beta_0=0.1),
            CategoryParams(name="B", price=6.0, beta_0=-0.2),
        ),
        metadata={"source": "unit-test"},
    )
    output_path = tmp_path / "mdp_params.yaml"
    save_mdp_params(params=params, output_path=output_path)

    loaded = load_mdp_params(output_path)
    assert loaded == params
