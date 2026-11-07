"""Smoke test all 2,160 UI combinations against the source CSV records."""
from itertools import product
from pathlib import Path

from predictor import ENCODINGS, FEATURE_COLS, feature_vector, load_data, lookup_prediction


def main():
    folder = Path(__file__).resolve().parent
    data = load_data(folder / "webapp.csv", folder / "webapp2.csv")
    checked = 0
    for combination in product(*(range(len(patterns)) for _, patterns in ENCODINGS.values())):
        selected = dict(zip(ENCODINGS.keys(), combination))
        prediction, binary, source = lookup_prediction(data, selected)
        assert binary == feature_vector(selected)
        assert all(int(source[k]) == v for k, v in binary.items())
        assert prediction.harm_probability == float(source["directional_stability_harm"])
        assert prediction.improvement_probability == float(source["directional_stability_improve"])
        assert prediction.delay_seconds == float(source["predicted_change_in_train_delay"])
        assert prediction.retained_trees_ratio == float(source["tree_used_ratio"])
        checked += 1
    print(f"SUCCESS: all {checked:,} combinations returned the exact stored results.")
    default = {key: 0 for key in ENCODINGS}
    pred, features, _ = lookup_prediction(data, default)
    print("First-option selections:", features)
    print("First-option results:", pred)


if __name__ == "__main__":
    main()
