"""L2-regularized multinomial logistic regression development adapter."""
from sklearn.linear_model import LogisticRegression


INITIAL_C = [0.01, 0.1, 1.0, 10.0, 100.0]


def make_estimator(parameters, seed):
    # With three classes, lbfgs fits the multinomial objective in sklearn 1.7.
    return LogisticRegression(C=parameters["C"], penalty="l2", solver="lbfgs",
                              tol=1e-6, max_iter=1000, fit_intercept=True,
                              class_weight=None, random_state=seed)


def tie_key(parameters):
    """Smaller C is stronger regularization; smaller keys win accuracy ties."""
    return parameters["C"]


def expand_candidates(initial_best, initial_summaries):
    """One expansion step only, based solely on the INITIAL inner-CV winner."""
    value = initial_best["C"]
    if value == min(INITIAL_C):
        return [{"C": 0.001}]
    if value == max(INITIAL_C):
        return [{"C": 1000.0}]
    return []


def export_model(estimator, state, parameters):
    classes = state["class_order"]
    positions = [list(estimator.classes_).index(label) for label in classes]
    return {
        "version": 1, "model_family": "softmax", "scale_numeric": True,
        "purpose": "outer_cv_reference_not_final_submission",
        "class_order": classes, "feature_order": state["feature_order"],
        "coef": estimator.coef_[positions].tolist(),
        "intercept": estimator.intercept_[positions].tolist(),
        "training_parameters": {**parameters, "solver": "lbfgs", "penalty": "l2",
                                "tol": 1e-6, "max_iter": estimator.max_iter},
    }
