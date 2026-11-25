"""Evaluate the decision tree on the team's frozen nested respondent folds."""
import argparse
import json
from pathlib import Path

from experiments.common import ModelSpec, ROOT, run_nested_cv
from models import decision_tree as tree
from models.decision_tree_inference import predict_proba


def decision_tree_spec():
    return ModelSpec(
        name="decision_tree",
        scale_numeric=False,
        initial_candidates=tree.grid(tree.DEPTHS, tree.LEAF_SIZES),
        make_estimator=tree.make_estimator,
        tie_key=tree.tie_key,
        expand_candidates=tree.expand_candidates,
        export_model=tree.export_model,
        predict_exported=predict_proba,
        description={
            "criterion": "gini",
            "splitter": "best",
            "tie_key": "shallower depth, then larger minimum leaf size",
            "expansion": "Once only: add depth 2 if depth 3 wins, and leaf size 20 if 10 wins. "
                         "Cross any new value with the other axis; unbounded depth and leaf size 1 are limits.",
        },
        source_files=("models/decision_tree.py", "models/decision_tree_inference.py",
                      "models/random_forest_inference.py", "experiments/run_decision_tree.py"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "csc311_project_training_202609.csv")
    parser.add_argument("--protocol", type=Path, default=ROOT / "development/protocol.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/decision_tree")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    summary = run_nested_cv(decision_tree_spec(), source=args.source, protocol_path=args.protocol,
                            output=args.output, overwrite=args.overwrite)
    print(json.dumps(summary["oof_metrics"], indent=2))


if __name__ == "__main__":
    main()
