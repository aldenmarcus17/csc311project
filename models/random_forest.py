"""Random-forest development adapter and portable model export."""
from itertools import product

import numpy as np
from sklearn.ensemble import RandomForestClassifier


N_ESTIMATORS = [50, 100, 200]
MAX_DEPTH = [5, 10, 20, None]
MIN_SAMPLES_LEAF = [1, 2, 5]
MAX_FEATURES = ["sqrt", 0.5]


def initial_candidates():
    """Return the predeclared Cartesian-product search grid."""
    return [
        {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
        }
        for n_estimators, max_depth, min_samples_leaf, max_features in product(
            N_ESTIMATORS, MAX_DEPTH, MIN_SAMPLES_LEAF, MAX_FEATURES
        )
    ]


def make_estimator(parameters, seed):
    """Construct a fresh, deterministic estimator for one recorded fit."""
    return RandomForestClassifier(
        n_estimators=parameters["n_estimators"],
        criterion="gini",
        max_depth=parameters["max_depth"],
        min_samples_leaf=parameters["min_samples_leaf"],
        max_features=parameters["max_features"],
        bootstrap=True,
        class_weight=None,
        random_state=seed,
        n_jobs=1,
    )


def tie_key(parameters):
    """Prefer shallower, smoother, smaller forests and then fewer candidate features."""
    depth = parameters["max_depth"]
    depth_complexity = float("inf") if depth is None else depth
    feature_complexity = 0 if parameters["max_features"] == "sqrt" else 1
    return (
        depth_complexity,
        -parameters["min_samples_leaf"],
        parameters["n_estimators"],
        feature_complexity,
    )


def export_model(estimator, state, parameters):
    """Export tree arrays and leaf probabilities for NumPy-only inference."""
    class_positions = [list(estimator.classes_).index(label) for label in state["class_order"]]
    trees = []
    for fitted_tree in estimator.estimators_:
        tree = fitted_tree.tree_
        values = np.asarray(tree.value[:, 0, :], dtype=float)[:, class_positions]
        totals = values.sum(axis=1, keepdims=True)
        probabilities = np.divide(values, totals, out=np.zeros_like(values), where=totals != 0)
        trees.append(
            {
                "children_left": tree.children_left.astype(int).tolist(),
                "children_right": tree.children_right.astype(int).tolist(),
                "feature": tree.feature.astype(int).tolist(),
                "threshold": tree.threshold.astype(float).tolist(),
                "probability": probabilities.tolist(),
            }
        )
    return {
        "version": 1,
        "model_family": "random_forest",
        "scale_numeric": False,
        "purpose": "outer_cv_reference_not_final_submission",
        "class_order": state["class_order"],
        "feature_order": state["feature_order"],
        "training_parameters": {
            **parameters,
            "criterion": "gini",
            "bootstrap": True,
            "class_weight": None,
        },
        "trees": trees,
    }
