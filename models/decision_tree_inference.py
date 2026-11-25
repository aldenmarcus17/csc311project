"""Predict with an exported single tree using the repository's tree traversal."""
import numpy as np

from models.random_forest_inference import _tree_probabilities
from preprocessing import transform_features


def predict_proba(raw_frame, state, model):
    if model["model_family"] != "decision_tree":
        raise ValueError("Expected a decision tree.")
    if any(model[key] != state[key] for key in ("feature_order", "class_order")):
        raise ValueError("Model and preprocessing orders differ.")
    # scikit-learn casts inputs to float32 before comparing them to stored thresholds.
    features = transform_features(raw_frame, state, scale_numeric=False).to_numpy(dtype=np.float32)
    return _tree_probabilities(features, model["tree"], len(state["class_order"]))
