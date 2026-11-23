"""Library-restricted inference for the compressed final random forest."""
import json
from pathlib import Path

import numpy as np

from preprocessing import load_state, read_survey, transform_features


def _tree_probabilities(features, left, right, split_features, thresholds, probabilities):
    nodes = np.zeros(len(features), dtype=np.int64)
    for _ in range(len(split_features)):
        active = split_features[nodes] >= 0
        if not active.any():
            return probabilities[nodes]
        rows = np.flatnonzero(active)
        current = nodes[rows]
        columns = split_features[current]
        go_left = features[rows, columns] <= thresholds[current]
        nodes[rows] = np.where(go_left, left[current], right[current])
    raise ValueError("Compressed tree traversal did not reach leaves.")


def predict_proba(raw_frame, state, metadata, archive_path):
    if metadata.get("model_family") != "random_forest":
        raise ValueError("Expected random-forest metadata.")
    if metadata.get("class_order") != state.get("class_order"):
        raise ValueError("Model and preprocessing class orders differ.")
    if metadata.get("feature_order") != state.get("feature_order"):
        raise ValueError("Model and preprocessing feature orders differ.")
    features = transform_features(raw_frame, state, scale_numeric=False).to_numpy(dtype=float)
    total = np.zeros((len(features), len(state["class_order"])), dtype=float)
    with np.load(archive_path, allow_pickle=False) as arrays:
        offsets = arrays["offsets"]
        if len(offsets) != metadata["tree_count"] + 1:
            raise ValueError("Tree count and compressed offsets differ.")
        for index in range(metadata["tree_count"]):
            start, stop = int(offsets[index]), int(offsets[index + 1])
            total += _tree_probabilities(
                features,
                arrays["children_left"][start:stop],
                arrays["children_right"][start:stop],
                arrays["feature"][start:stop],
                arrays["threshold"][start:stop],
                arrays["probability"][start:stop],
            )
    probabilities = total / metadata["tree_count"]
    temperature = float(metadata.get("temperature", 1.0))
    if temperature <= 0 or not np.isfinite(temperature):
        raise ValueError("Temperature must be positive and finite.")
    log_probabilities = np.log(np.clip(probabilities, 1e-15, 1.0)) / temperature
    log_probabilities -= log_probabilities.max(axis=1, keepdims=True)
    calibrated = np.exp(log_probabilities)
    return calibrated / calibrated.sum(axis=1, keepdims=True)


def load_bundle(directory):
    directory = Path(directory)
    state = load_state(directory / "preprocessing_state.json")
    metadata = json.loads(
        (directory / "model_metadata.json").read_text(encoding="utf-8")
    )
    return state, metadata, directory / "forest_arrays.npz"


def predict_csv(csv_filename, directory):
    state, metadata, archive = load_bundle(directory)
    frame = read_survey(csv_filename, state=state)
    probabilities = predict_proba(frame, state, metadata, archive)
    return np.asarray(state["class_order"])[probabilities.argmax(axis=1)]
