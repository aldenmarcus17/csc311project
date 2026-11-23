"""Portable softmax inference. Imports only standard library, NumPy and pandas.

Used to verify exported outer-fold reference models; this is not pred.py and
does not select a final model or a final full-development-data configuration.
"""
import json
from pathlib import Path

import numpy as np

from preprocessing import load_state, read_survey, transform_features


def predict_proba(frame, state, parameters):
    if parameters.get("version") != 1 or parameters.get("model_family") != "softmax":
        raise ValueError("Unsupported exported model.")
    if not parameters.get("scale_numeric"):
        raise ValueError("Softmax reference model requires the saved numeric scaling.")
    if (parameters["feature_order"] != state["feature_order"] or
            parameters["class_order"] != state["class_order"]):
        raise ValueError("Model and preprocessing feature/class order differ.")
    coefficients = np.asarray(parameters["coef"], dtype=float)
    intercept = np.asarray(parameters["intercept"], dtype=float)
    expected = (len(state["class_order"]), len(state["feature_order"]))
    if coefficients.shape != expected or intercept.shape != (expected[0],):
        raise ValueError("Exported parameter shape is incorrect.")
    if not np.isfinite(coefficients).all() or not np.isfinite(intercept).all():
        raise ValueError("Exported parameters must be finite.")
    features = transform_features(frame, state, scale_numeric=True).to_numpy()
    if not len(features):
        return np.empty((0, expected[0]), dtype=float)
    scores = features @ coefficients.T + intercept
    if not np.isfinite(scores).all():
        raise ValueError("Softmax scores are nonfinite.")
    scores -= scores.max(axis=1, keepdims=True)
    numerator = np.exp(scores)
    return numerator / numerator.sum(axis=1, keepdims=True)


def predict_csv(csv_filename, artifact_directory):
    directory = Path(artifact_directory)
    state = load_state(directory / "preprocessing_state.json")
    parameters = json.loads((directory / "model.json").read_text(encoding="utf-8"))
    frame = read_survey(csv_filename, state=state)
    probabilities = predict_proba(frame, state, parameters)
    return np.asarray(parameters["class_order"])[probabilities.argmax(axis=1)]
