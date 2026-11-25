"""Write LaTeX fragments from a completed development run."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]


def build(results, destination):
    summary = json.loads((results / "summary.json").read_text())
    manifest = json.loads((results / "run_manifest.json").read_text())
    if manifest["status"] != "complete" or summary["model_family"] != "decision_tree":
        raise ValueError("Expected a completed decision-tree run.")
    with (results / "oof_predictions.csv").open(newline="") as preds:
        rows = list(csv.DictReader(preds))
    for name in ("random_forest", "softmax"):
        with (ROOT / "results" / name / "oof_predictions.csv").open(newline="") as preds:
            teammate = list(csv.DictReader(preds))
        keys = ("row_index", "id", "true_label", "outer_fold")
        if [[r[k] for k in keys] for r in rows] != [[r[k] for k in keys] for r in teammate]:
            raise ValueError(f"Respondent, label or fold mismatch with {name}.")
    correct = sum(r["true_label"] == r["predicted_label"] for r in rows)
    metrics = summary["oof_metrics"]
    if abs(correct / len(rows) - metrics["accuracy"]) > 1e-12:
        raise ValueError("Summary and saved predictions disagree.")
    search = json.loads((results / "inner_search_summary.json").read_text())
    depth_scores = {d: mean(r["mean_validation_accuracy"] for r in search if r["eligible"]
                            and r["parameters"] == {"max_depth": d, "min_samples_leaf": 1})
                    for d in (3, 10, None)}
    version = manifest["sklearn"]
    caveat = "" if version == "1.7.2" else (
        rf"\emph{{Preliminary run with scikit-learn {version}. The team comparison uses version 1.7.2.}} " + "\n")
    methodology = r"""\paragraph{Decision tree.}
A single classification tree can represent interactions among survey answers. For example,
the same colour flag can lead to different predictions depending on whether the respondent
also selected an indoor venue. We used scikit-learn's best-split Gini tree with seed 67
and all 73 unscaled predictors. Each split maximizes the reduction in weighted Gini impurity.
We varied maximum depth (3, 5, 10, 20 or unlimited) and minimum samples per leaf
(1, 2, 5 or 10). Depth limits the number of successive decisions, while a larger minimum
leaf size requires more training evidence per prediction.

We selected each outer fold's setting by mean accuracy on its three inner respondent folds.
Ties within $10^{-12}$ favoured shallower trees, then larger leaves. The one-step expansion
rule added depth 2 if depth 3 won, or leaf size 20 if size 10 won, crossing new values with
the other parameter. A leaf size of 1 and unlimited depth are natural endpoints.
"""
    matrix = metrics["confusion_matrix"]
    classes = summary["class_order"]
    pairs = [(matrix[i][j] + matrix[j][i], i, j) for i in range(len(classes)) for j in range(i)]
    count, i, j = max(pairs)
    precision = ", ".join(f"{100 * metrics['per_class'][c]['precision']:.2f}\\%" for c in classes)
    scores = summary["outer_fold_accuracies"]
    results_text = caveat + rf"""\paragraph{{Decision-tree results.}}
The tree correctly classified {correct} of {len(rows)} outer-fold responses
({metrics['accuracy']*100:.2f}\% accuracy, macro-F1 {metrics['macro_f1']:.3f}).
Accuracy ranged from {min(scores)*100:.2f}\% to {max(scores)*100:.2f}\%
across the five folds. Mean outer training accuracy was {summary['mean_training_accuracy']*100:.2f}\%,
a gap of {(summary['mean_training_accuracy'] - mean(scores))*100:.2f} percentage points.
Holding the minimum leaf size at one, mean inner validation accuracy across the five
searches was {100*depth_scores[3]:.2f}\% at depth 3, {100*depth_scores[10]:.2f}\% at
depth 10 and {100*depth_scores[None]:.2f}\% at unlimited depth.
The search used {summary['initial_search_fits']} initial inner fits, {summary['expansion_search_fits']} expansion
fits and {summary['outer_refits']} outer refits.

Precision for {', '.join(classes[:-1])} and {classes[-1]} was {precision}, respectively.
Confusion between {classes[j]} and {classes[i]} accounted for {count} of {len(rows)-correct} errors:
{matrix[j][i]} {classes[j]} responses were classified as {classes[i]}, and {matrix[i][j]} went
the other way. Preliminary inspection of responses informed the shared keyword dictionary,
so these development scores may be optimistic for new respondents.
"""
    destination.mkdir(exist_ok=True, parents=True)
    for name, value in (("methodology", methodology), ("results", results_text)):
        (destination / f"decision_tree_{name}.tex").write_text(value, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results/decision_tree")
    parser.add_argument("--output", type=Path, default=ROOT / "report")
    options = parser.parse_args()
    build(options.results, options.output)
