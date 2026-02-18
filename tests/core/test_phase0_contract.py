"""Phase 0 contract tests: schema, ownership boundaries, and migration checklist."""

from __future__ import annotations

from pathlib import Path

from discount_engine.core.params import MDPParams
from discount_engine.core.types import ContinuousState, DiscreteState


def test_state_types_match_phase0_contract() -> None:
    continuous = ContinuousState(
        churn_propensity=0.2,
        discount_memory=(0.0, 1.0, 2.0),
        purchase_recency=(0.0, 1.0, 5.0),
    )
    discrete = DiscreteState(
        churn_bucket=0,
        memory_buckets=(0, 1, 2),
        recency_buckets=(0, 0, 1),
    )

    assert isinstance(continuous.churn_propensity, float)
    assert len(continuous.discount_memory) == len(continuous.purchase_recency)
    assert len(discrete.memory_buckets) == len(discrete.recency_buckets)


def test_mdp_params_schema_contains_frozen_phase0_fields() -> None:
    field_names = set(MDPParams.__dataclass_fields__.keys())
    required = {
        "delta",
        "gamma",
        "alpha",
        "beta_p",
        "beta_l",
        "beta_m",
        "eta",
        "c0",
        "categories",
    }
    assert required.issubset(field_names)


def test_core_modules_do_not_import_dp_or_rl() -> None:
    core_dir = Path("src/discount_engine/core")
    for py_file in core_dir.glob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        assert "from discount_engine.dp" not in content
        assert "import discount_engine.dp" not in content
        assert "from discount_engine.rl" not in content
        assert "import discount_engine.rl" not in content


def test_roadmap_migration_plan_is_marked_complete_for_moved_modules() -> None:
    roadmap = Path("specs/02_roadmap.md").read_text(encoding="utf-8")
    assert (
        "- [x] Move files from legacy folders (`agents`, `simulators`, `envs`) into new ownership boundaries."
        in roadmap
    )
