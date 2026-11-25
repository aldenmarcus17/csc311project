"""Single Gini tree for the shared respondent-grouped development experiment."""
from itertools import product

from sklearn.tree import DecisionTreeClassifier


DEPTHS = (3, 5, 10, 20, None)
LEAF_SIZES = (1, 2, 5, 10)


def grid(depths, leaf_sizes):
    return [{"max_depth": d, "min_samples_leaf": n} for d, n in product(depths, leaf_sizes)]


def make_estimator(parameters, seed):
    return DecisionTreeClassifier(criterion="gini", splitter="best", random_state=seed, **parameters)


def tie_key(parameters):
    depth = parameters["max_depth"]
    return (float("inf") if depth is None else depth, -parameters["min_samples_leaf"])


def expand_candidates(initial_best, initial_summaries):
    """Test one step beyond an expandable boundary, using only the initial inner winner."""
    depths = DEPTHS + (2,) if initial_best["max_depth"] == min(d for d in DEPTHS if d) else DEPTHS
    leaves = LEAF_SIZES + (20,) if initial_best["min_samples_leaf"] == max(LEAF_SIZES) else LEAF_SIZES
    return [p for p in grid(depths, leaves)
            if p["max_depth"] not in DEPTHS or p["min_samples_leaf"] not in LEAF_SIZES]


def export_model(estimator, state, parameters):
    tree = estimator.tree_
    order = list(estimator.classes_)
    values = tree.value[:, 0, :][:, [order.index(label) for label in state["class_order"]]]
    return {
        "model_family": "decision_tree",
        "class_order": state["class_order"],
        "feature_order": state["feature_order"],
        "training_parameters": parameters,
        "tree": {
            "children_left": tree.children_left.tolist(),
            "children_right": tree.children_right.tolist(),
            "feature": tree.feature.tolist(),
            "threshold": tree.threshold.tolist(),
            "probability": (values / values.sum(axis=1, keepdims=True)).tolist(),
        },
    }
