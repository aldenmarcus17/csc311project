"""Run the softmax workstream: python -m experiments.run_softmax"""
import argparse
import json
from pathlib import Path

from experiments.common import ModelSpec, ROOT, run_nested_cv
from models.softmax import INITIAL_C, expand_candidates, export_model, make_estimator, tie_key
from models.softmax_inference import predict_proba


def softmax_spec():
    return ModelSpec(
        name="softmax", scale_numeric=True,
        initial_candidates=[{"C": value} for value in INITIAL_C],
        make_estimator=make_estimator, tie_key=tie_key, expand_candidates=expand_candidates,
        export_model=export_model, predict_exported=predict_proba,
        description={"objective": "multinomial cross-entropy with L2 penalty", "solver": "lbfgs",
                     "tol": 1e-6, "initial_max_iter": 1000, "retry_max_iter": 5000,
                     "class_weight": None, "fit_intercept": True,
                     "tie_key": "smaller C", "expansion": "one extra C=0.001 or C=1000 at an initial-grid boundary"},
        source_files=("models/softmax.py", "models/softmax_inference.py", "experiments/run_softmax.py"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "csc311_project_training_202609.csv")
    parser.add_argument("--protocol", type=Path, default=ROOT / "development/protocol.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/softmax")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    summary = run_nested_cv(softmax_spec(), source=args.source, protocol_path=args.protocol,
                            output=args.output, overwrite=args.overwrite)
    print(json.dumps({"accuracy": summary["oof_metrics"]["accuracy"],
                      "macro_f1": summary["oof_metrics"]["macro_f1"],
                      "fit_attempts": summary["fit_attempts"], "final_model_selected": False}, indent=2))


if __name__ == "__main__":
    main()
