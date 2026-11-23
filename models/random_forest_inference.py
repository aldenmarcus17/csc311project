"""Portable random-forest inference using only NumPy and shared preprocessing."""
import json
from pathlib import Path

import numpy as np

from preprocessing import load_state, read_survey, transform_features


def _validate_model(state, model):
    if model.get("model_family") != "random_forest" or model.get("scale_numeric") is not False:
        raise ValueError("Expected an unscaled random-forest export.")
    if model.get("class_order") != state.get("class_order"):
        raise ValueError("Model and preprocessing class orders differ.")
    if model.get("feature_order") != state.get("feature_order"):
        raise ValueError("Model and preprocessing feature orders differ.")
    if not model.get("trees"):
        raise ValueError("Random-forest export contains no trees.")


def _tree_probabilities(features, tree, class_count):
    children_left = np.asarray(tree["children_left"], dtype=np.int64)
    children_right = np.asarray(tree["children_right"], dtype=np.int64)
    split_features = np.asarray(tree["feature"], dtype=np.int64)
    thresholds = np.asarray(tree["threshold"], dtype=float)
    probabilities = np.asarray(tree["probability"], dtype=float)
    node_count = len(split_features)
    if not (
        children_left.shape == children_right.shape == thresholds.shape == (node_count,)
        and probabilities.shape == (node_count, class_count)
    ):
        raise ValueError("Malformed exported tree arrays.")

    nodes = np.zeros(len(features), dtype=np.int64)
    for _ in range(node_count):
        active = split_features[nodes] >= 0
        if not active.any():
            result = probabilities[nodes]
            if not np.isfinite(result).all() or not np.allclose(result.sum(axis=1), 1.0):
                raise ValueError("Malformed leaf probabilities.")
            return result
        rows = np.flatnonzero(active)
        current = nodes[rows]
        columns = split_features[current]
        go_left = features[rows, columns] <= thresholds[current]
        nodes[rows] = np.where(go_left, children_left[current], children_right[current])
        if (nodes[rows] < 0).any() or (nodes[rows] >= node_count).any():
            raise ValueError("Malformed exported tree links.")
    raise ValueError("Exported tree traversal did not reach leaves.")


def predict_proba(raw_frame, state, model):
    """Return probabilities in the saved Baseball/Basketball/Soccer order."""
    _validate_model(state, model)
    features = transform_features(raw_frame, state, scale_numeric=False).to_numpy(dtype=float)
    total = np.zeros((len(features), len(state["class_order"])), dtype=float)
    for tree in model["trees"]:
        total += _tree_probabilities(features, tree, len(state["class_order"]))
    return total / len(model["trees"])


def predict_csv(csv_filename, artifact_directory):
    """Load a saved preprocessing/model bundle and return predicted labels."""
    directory = Path(artifact_directory)
    state = load_state(directory / "preprocessing_state.json")
    model = json.loads((directory / "model.json").read_text(encoding="utf-8"))
    frame = read_survey(csv_filename, state=state)
    probabilities = predict_proba(frame, state, model)
    return np.asarray(state["class_order"])[probabilities.argmax(axis=1)]
