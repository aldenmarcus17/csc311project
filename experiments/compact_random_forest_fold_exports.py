"""Convert verbose fold-model JSON files to compressed safe NumPy archives."""
import json
from pathlib import Path

import numpy as np

from experiments.common import ROOT


def compact_export(directory):
    source = directory / "model.json"
    if not source.exists():
        return
    payload = json.loads(source.read_text(encoding="utf-8"))
    trees = payload.pop("trees")
    offsets = [0]
    parts = {
        "children_left": [],
        "children_right": [],
        "feature": [],
        "threshold": [],
        "probability": [],
    }
    for tree in trees:
        count = len(tree["feature"])
        offsets.append(offsets[-1] + count)
        parts["children_left"].append(np.asarray(tree["children_left"], dtype=np.int32))
        parts["children_right"].append(np.asarray(tree["children_right"], dtype=np.int32))
        parts["feature"].append(np.asarray(tree["feature"], dtype=np.int16))
        parts["threshold"].append(np.asarray(tree["threshold"], dtype=np.float64))
        parts["probability"].append(np.asarray(tree["probability"], dtype=np.float64))
    np.savez_compressed(
        directory / "forest_arrays.npz",
        offsets=np.asarray(offsets, dtype=np.int64),
        **{name: np.concatenate(values) for name, values in parts.items()},
    )
    payload.update(
        format="concatenated_tree_arrays_npz",
        tree_count=len(trees),
        compact_conversion="lossless post-run conversion of verified JSON fold export",
    )
    (directory / "model_metadata.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    source.unlink()


def main():
    root = ROOT / "results/random_forest/fold_models"
    for fold in range(5):
        compact_export(root / f"fold_{fold}")
    print("Compacted random-forest fold exports.")


if __name__ == "__main__":
    main()
