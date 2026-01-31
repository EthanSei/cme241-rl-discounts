"""Tests for feature engineering utilities."""

import pandas as pd
import pytest

from discount_engine.utils.feature_engineering import build_basic_features, build_features


def test_build_basic_features() -> None:
    tables = {
        "transaction_data": pd.DataFrame(
            {
                "household_key": [1],
                "basket_id": [100],
                "week_no": [5],
                "product_id": [10],
                "quantity": [2],
                "sales_value": [10.0],
                "store_id": [3],
                "retail_disc": [1.0],
                "coupon_disc": [0.5],
                "coupon_match_disc": [0.0],
            }
        ),
        "product": pd.DataFrame(
            {
                "product_id": [10],
                "department": ["GROCERY"],
                "commodity_desc": ["CEREAL"],
                "brand": ["BRAND_A"],
            }
        ),
        "hh_demographic": pd.DataFrame(
            {
                "household_key": [1],
                "age_desc": ["25-34"],
                "income_desc": ["50-74K"],
                "homeowner_desc": ["HOMEOWNER"],
            }
        ),
    }

    features = build_basic_features(tables)

    assert features.loc[0, "price_per_unit"] == 5.0
    assert features.loc[0, "total_discount"] == 1.5
    assert round(features.loc[0, "discount_rate"], 4) == 0.15
    assert bool(features.loc[0, "promo_flag"]) is True
    assert features.loc[0, "department"] == "GROCERY"
    assert features.loc[0, "income_desc"] == "50-74K"


def test_build_features_column_controls() -> None:
    tables = {
        "transaction_data": pd.DataFrame(
            {
                "household_key": [1],
                "basket_id": [100],
                "week_no": [5],
                "product_id": [10],
                "quantity": [2],
                "sales_value": [10.0],
                "store_id": [3],
                "retail_disc": [1.0],
                "coupon_disc": [0.5],
                "coupon_match_disc": [0.0],
            }
        ),
        "product": pd.DataFrame(
            {
                "product_id": [10],
                "department": ["GROCERY"],
                "commodity_desc": ["CEREAL"],
                "brand": ["BRAND_A"],
            }
        ),
        "hh_demographic": pd.DataFrame(
            {
                "household_key": [1],
                "age_desc": ["25-34"],
                "income_desc": ["50-74K"],
                "homeowner_desc": ["HOMEOWNER"],
            }
        ),
    }

    features = build_features(
        tables,
        include_columns=["price_per_unit", "total_discount"],
        exclude_columns=["department"],
    )

    assert "price_per_unit" in features.columns
    assert "total_discount" in features.columns
    assert "department" not in features.columns
    assert "household_key" in features.columns


def test_build_features_unknown_set() -> None:
    tables = {
        "transaction_data": pd.DataFrame(
            {
                "household_key": [1],
                "basket_id": [100],
                "week_no": [5],
                "product_id": [10],
                "quantity": [2],
                "sales_value": [10.0],
                "store_id": [3],
                "retail_disc": [1.0],
                "coupon_disc": [0.5],
                "coupon_match_disc": [0.0],
            }
        ),
        "product": pd.DataFrame(
            {
                "product_id": [10],
                "department": ["GROCERY"],
                "commodity_desc": ["CEREAL"],
                "brand": ["BRAND_A"],
            }
        ),
        "hh_demographic": pd.DataFrame(
            {
                "household_key": [1],
                "age_desc": ["25-34"],
                "income_desc": ["50-74K"],
                "homeowner_desc": ["HOMEOWNER"],
            }
        ),
    }

    with pytest.raises(KeyError):
        build_features(tables, feature_sets=["missing"])
