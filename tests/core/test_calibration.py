"""Tests for MDP parameter calibration."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml

from discount_engine.core import calibration as calib
from discount_engine.core.calibration import calibrate_mdp_params
from discount_engine.utils.io import ingest_raw_to_processed


def _write_raw_calibration_fixture(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    products = pd.DataFrame(
        {
            "Product ID": [100, 101, 200, 300],
            "Commodity Desc": ["A", "A", "B", "C"],
        }
    )

    rows: list[dict[str, float | int]] = []
    prices = {100: 10.0, 101: 11.0, 200: 7.0, 300: 5.0}
    for household in (1, 2, 3, 4):
        for day in range(1, 9):
            if (household + day) % 2 == 0:
                product_id = 100 if household % 2 == 0 else 101
                full_price = prices[product_id]
                retail_disc = -2.0 if day % 4 == 0 else 0.0
                rows.append(
                    {
                        "Household Key": household,
                        "Day": day,
                        "Product ID": product_id,
                        "Quantity": 1,
                        "Sales Value": full_price + retail_disc,
                        "Retail Disc": retail_disc,
                        "Coupon Disc": 0.0,
                        "Coupon Match Disc": 0.0,
                    }
                )
            if day in (3, 6) and household in (1, 3):
                product_id = 200
                full_price = prices[product_id]
                retail_disc = -0.7 if day == 6 else 0.0
                rows.append(
                    {
                        "Household Key": household,
                        "Day": day,
                        "Product ID": product_id,
                        "Quantity": 1,
                        "Sales Value": full_price + retail_disc,
                        "Retail Disc": retail_disc,
                        "Coupon Disc": 0.0,
                        "Coupon Match Disc": 0.0,
                    }
                )
            if day in (2, 5, 8) and household == 4:
                product_id = 300
                full_price = prices[product_id]
                rows.append(
                    {
                        "Household Key": household,
                        "Day": day,
                        "Product ID": product_id,
                        "Quantity": 1,
                        "Sales Value": full_price,
                        "Retail Disc": 0.0,
                        "Coupon Disc": 0.0,
                        "Coupon Match Disc": 0.0,
                    }
                )

    transactions = pd.DataFrame(rows)
    transactions.to_csv(raw_dir / "transaction_data.csv", index=False)
    products.to_csv(raw_dir / "product.csv", index=False)


def test_calibrate_mdp_params_writes_expected_yaml(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    output_path = processed_dir / "mdp_params.yaml"
    _write_raw_calibration_fixture(raw_dir)
    ingest_raw_to_processed(raw_dir=raw_dir, processed_dir=processed_dir, file_format="csv")

    payload = calibrate_mdp_params(
        processed_dir=processed_dir,
        output_path=output_path,
        n_categories=2,
        alpha_grid=(0.0, 0.5, 0.9),
    )

    assert output_path.exists()
    parsed = yaml.safe_load(output_path.read_text())
    assert isinstance(parsed, dict)
    assert parsed["alpha"] == payload["alpha"]
    assert parsed["delta"] == payload["delta"]
    assert len(parsed["categories"]) == 2
    assert parsed["beta_p"] > 0.0
    assert parsed["beta_l"] > 0.0
    assert parsed["beta_m"] > 0.0
    assert 0.0 <= parsed["alpha"] <= 1.0
    assert 0.0 < parsed["c0"] < 1.0
    assert parsed["metadata"]["alpha_selection_metric"] == "validation_nll"
    assert parsed["metadata"]["validation_fraction"] == 0.2
    assert parsed["metadata"]["deal_signal_mode"] == "positive_centered_anomaly"
    assert parsed["metadata"]["memory_mode"] == "gap_aware_ewma"
    assert parsed["metadata"]["fit_neg_log_likelihood"] == parsed["metadata"][
        "fit_val_neg_log_likelihood"
    ]
    for category in parsed["categories"]:
        assert category["price"] > 0.0
        assert "beta_0" in category
        assert "name" in category


def test_calibration_script_cli_runs(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    output_path = processed_dir / "mdp_params.yaml"
    _write_raw_calibration_fixture(raw_dir)
    ingest_raw_to_processed(raw_dir=raw_dir, processed_dir=processed_dir, file_format="csv")

    project_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(project_root / "src")

    cmd = [
        sys.executable,
        str(project_root / "scripts" / "data" / "calibrate_mdp_params.py"),
        "--processed-dir",
        str(processed_dir),
        "--output-path",
        str(output_path),
        "--n-categories",
        "2",
    ]
    result = subprocess.run(cmd, check=False, env=env, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert output_path.exists()


def test_time_split_masks_create_forward_validation_window() -> None:
    times = np.array([1, 2, 3, 4, 5], dtype=float)
    train_mask, val_mask, cutoff = calib._build_time_split_masks(
        times=times,
        validation_fraction=0.40,
    )

    assert cutoff > 3.0
    assert cutoff < 4.0
    assert train_mask.tolist() == [True, True, True, False, False]
    assert val_mask.tolist() == [False, False, False, True, True]


def test_gap_aware_memory_uses_elapsed_time() -> None:
    panel = pd.DataFrame(
        {
            "household_key": [1, 1, 1, 2, 2],
            "cat_idx": [0, 0, 0, 0, 0],
            "day": [1, 3, 6, 1, 2],
            "deal_signal": [0.2, 0.4, 0.1, 0.5, 0.0],
        }
    )
    memory = calib._compute_memory_feature(panel=panel, alpha=0.5)

    # Sequence 1 (household 1): gap-aware decay
    # m1 = 0.5**(3-1)*0 + 0.2 = 0.2
    # m2 = 0.5**(6-3)*0.2 + 0.4 = 0.425
    assert np.isclose(memory[0], 0.0)
    assert np.isclose(memory[1], 0.2)
    assert np.isclose(memory[2], 0.425)

    # Sequence 2 (household 2) should reset independently.
    assert np.isclose(memory[3], 0.0)
    assert np.isclose(memory[4], 0.5)


def test_build_purchase_panel_deal_signal_not_perfect_label_proxy(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    _write_raw_calibration_fixture(raw_dir)
    ingest_raw_to_processed(raw_dir=raw_dir, processed_dir=processed_dir, file_format="csv")

    tables, _ = calib._load_processed_tables_with_fallback(processed_dir)
    transactions = calib._prepare_transactions(tables["transaction_data"])
    products = calib._prepare_products(
        tables["product"],
        product_ids=set(transactions["product_id"]),
        category_column="commodity_desc",
    )

    merged = transactions.merge(
        products[["product_id", "commodity_desc"]],
        on="product_id",
        how="inner",
    )
    merged = calib._compute_price_and_discount_signals(merged)
    selected = calib._select_top_categories(merged, "commodity_desc", 2)
    merged = merged[merged["commodity_desc"].isin(selected)].copy()
    panel, _ = calib._build_purchase_panel(
        merged=merged,
        category_column="commodity_desc",
        selected_categories=selected,
        time_column=calib._detect_time_column(merged),
    )

    has_signal = panel["deal_signal"] > 1e-12
    assert bool((has_signal & (panel["purchased"] == 1)).any())
    assert bool((has_signal & (panel["purchased"] == 0)).any())
