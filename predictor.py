"""Exact BetaPredict CSV lookups using the user's seven-category feature encoding.

No model is trained here. Values come directly from the supplied CSVs.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Mapping

import pandas as pd

FEATURE_COLS = tuple(f"feature_{i}" for i in range(1, 16))

# The order of each tuple is the order of the radio choices in the UI.
# These are CUMULATIVE encodings, e.g. density category 2 => (0, 1, 1),
# NOT mutually exclusive / one-hot encodings.
ENCODINGS: dict[str, tuple[tuple[str, ...], tuple[tuple[int, ...], ...]]] = {
    "number_incident": (
        ("feature_1",),
        ((1,), (0,)),
    ),
    "length_incident": (
        ("feature_2", "feature_3"),
        ((1, 1), (0, 1), (0, 0)),
    ),
    "density": (
        ("feature_4", "feature_5", "feature_6"),
        ((1, 1, 1), (0, 1, 1), (0, 0, 1), (0, 0, 0)),
    ),
    "conflict": (
        ("feature_7", "feature_8"),
        ((1, 1), (0, 1), (0, 0)),
    ),
    "redlights": (
        ("feature_9", "feature_10"),
        ((1, 1), (0, 1), (0, 0)),
    ),
    "zonehour_typical_workload": (
        ("feature_11", "feature_12", "feature_13", "feature_14"),
        ((1, 1, 1, 1), (0, 1, 1, 1), (0, 0, 1, 1),
         (0, 0, 0, 1), (0, 0, 0, 0)),
    ),
    "peak": (
        ("feature_15",),
        ((0,), (1,)),
    ),
}

CATEGORY_LABELS = (
    ("number_incident", "Select incident count category",
     ("number_incident = 0", "number_incident > 0")),
    ("length_incident", "Select incident length category",
     ("length_incident = 0", "0 < length_incident ≤ 3", "length_incident > 3")),
    ("density", "Select traffic density category",
     ("density ≤ 0.15", "0.15 < density ≤ 0.28", "0.28 < density ≤ 0.45", "density > 0.45")),
    ("conflict", "Select conflict level category",
     ("conflict = 0", "0 < conflict ≤ 0.25", "conflict > 0.25")),
    ("redlights", "Select red light category",
     ("redlights = 0", "0 < redlights ≤ 0.25", "redlights > 0.25")),
    ("zonehour_typical_workload", "Select workload category",
     ("zonehour_typical_workload = 1",
      "zonehour_typical_workload = 2",
      "zonehour_typical_workload = 3",
      "zonehour_typical_workload = 4",
      "zonehour_typical_workload = 5")),
    ("peak", "Select peak hour category",
     ("peak = 0", "peak = 1")),
)


@dataclass(frozen=True)
class Prediction:
    harm_probability: float      # fractional probability: 0 to 1
    delay_seconds: float          # signed seconds per train section
    improvement_probability: float  # fractional probability: 0 to 1
    retained_trees_ratio: float   # fraction: 0 to 1


def feature_vector(choice_indices: Mapping[str, int]) -> dict[str, int]:
    """Convert zero-based radio indices into the 15 cumulative binary features."""
    result: dict[str, int] = {}
    for key, (columns, patterns) in ENCODINGS.items():
        if key not in choice_indices:
            raise ValueError(f"Missing selection for {key}")
        chosen = choice_indices[key]
        if not isinstance(chosen, int) or chosen < 0 or chosen >= len(patterns):
            raise ValueError(f"Invalid selection {chosen!r} for {key}")
        result.update(zip(columns, patterns[chosen]))
    if tuple(result) != FEATURE_COLS:
        raise AssertionError("Feature ordering changed unexpectedly")
    return result


def load_data(csv1: str | Path, csv2: str | Path) -> pd.DataFrame:
    """Merge the datasets by all 15 features; validate exact lookup coverage."""
    first = pd.read_csv(csv1)
    second = pd.read_csv(csv2)
    required_first = set(FEATURE_COLS) | {
        "predicted_change_in_train_delay",
        "directional_stability_improve",
        "directional_stability_harm",
    }
    required_second = set(FEATURE_COLS) | {"tree_used_ratio"}

    for name, frame, expected in (
        ("webapp.csv", first, required_first),
        ("webapp2.csv", second, required_second),
    ):
        missing = sorted(expected - set(frame.columns))
        if missing:
            raise ValueError(f"{name} is missing columns: {missing}")
        if frame.empty:
            raise ValueError(f"{name} is empty")
        if frame[list(expected)].isna().any().any():
            raise ValueError(f"{name} contains missing values")
        if frame.duplicated(subset=list(FEATURE_COLS)).any():
            raise ValueError(f"{name} has duplicate feature combinations")
        for col in FEATURE_COLS:
            if not frame[col].isin([0, 1]).all():
                raise ValueError(f"{name}: {col} must contain only 0 or 1")

    combined = first.merge(
        second, on=list(FEATURE_COLS),
        how="inner", validate="one_to_one",
    )
    if len(combined) != len(first) or len(combined) != len(second):
        raise ValueError(
            "The two CSVs do not cover the exact same feature combinations"
        )

    # Each possible selection should lead to exactly one row.
    expected_keys = set()
    for choices in product(*(range(len(pats)) for _, pats in ENCODINGS.values())):
        choice_map = dict(zip(ENCODINGS.keys(), choices))
        encoded = feature_vector(choice_map)
        expected_keys.add(tuple(encoded[col] for col in FEATURE_COLS))
    found_keys = set(map(tuple, combined[list(FEATURE_COLS)].itertuples(index=False, name=None)))
    if found_keys != expected_keys:
        raise ValueError(
            f"Feature pattern mismatch: {len(expected_keys - found_keys)} "
            f"expected combinations missing; {len(found_keys - expected_keys)} "
            "unexpected combinations"
        )

    for field in ("directional_stability_improve", "directional_stability_harm", "tree_used_ratio"):
        if not combined[field].between(0, 1).all():
            raise ValueError(f"{field} must contain fractions between 0 and 1")

    if not ((combined["directional_stability_improve"] +
             combined["directional_stability_harm"] - 1).abs() < 0.00001).all():
        raise ValueError("Improve and harm probabilities do not sum to 1")

    return combined


def lookup_prediction(
    combined: pd.DataFrame,
    choice_indices: Mapping[str, int],
) -> tuple[Prediction, dict[str, int], pd.Series]:
    """Find the unique matching data row, without interpolation or estimation."""
    features = feature_vector(choice_indices)
    matching = pd.Series(True, index=combined.index)
    for col, bit in features.items():
        matching &= combined[col].eq(bit)
    rows = combined.loc[matching]
    if len(rows) != 1:
        raise LookupError(f"Expected one matching row; found {len(rows)}")
    row = rows.iloc[0]
    prediction = Prediction(
        harm_probability=float(row["directional_stability_harm"]),
        delay_seconds=float(row["predicted_change_in_train_delay"]),
        improvement_probability=float(row["directional_stability_improve"]),
        retained_trees_ratio=float(row["tree_used_ratio"]),
    )
    return prediction, features, row
