"""Run the random-forest workstream: python -m experiments.run_random_forest"""
import argparse
import json
from pathlib import Path

from experiments.common import ModelSpec, ROOT, run_nested_cv
from models.random_forest import (
    export_model,
    initial_candidates,
    make_estimator,
    tie_key,
)
from models.random_forest_inference import predict_proba


def random_forest_spec():
    return ModelSpec(
        name="random_forest",
        scale_numeric=False,
        initial_candidates=initial_candidates(),
        make_estimator=make_estimator,
        tie_key=tie_key,
        export_model=export_model,
        predict_exported=predict_proba,
        description={
            "criterion": "gini",
            "bootstrap": True,
            "class_weight": None,
            "n_jobs": 1,
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, 20, None],
            "min_samples_leaf": [1, 2, 5],
            "max_features": ["sqrt", 0.5],
            "tie_key": "shallower depth, larger leaf, fewer trees, then sqrt features",
            "expansion": "none",
        },
        source_files=(
            "models/random_forest.py",
            "models/random_forest_inference.py",
            "experiments/run_random_forest.py",
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "csc311_project_training_202609.csv")
    parser.add_argument("--protocol", type=Path, default=ROOT / "development/protocol.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/random_forest")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    summary = run_nested_cv(
        random_forest_spec(),
        source=args.source,
        protocol_path=args.protocol,
        output=args.output,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "accuracy": summary["oof_metrics"]["accuracy"],
                "macro_f1": summary["oof_metrics"]["macro_f1"],
                "fit_attempts": summary["fit_attempts"],
                "final_model_selected": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
