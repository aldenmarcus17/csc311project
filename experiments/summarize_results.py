"""Create common evaluation plots and error diagnostics from saved outer predictions.

Run after evaluation: python -m experiments.summarize_results --results results/softmax
These diagnostics do not trigger feature changes, tuning or model selection.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.common import ROOT, classification_metrics, validate_inputs, write_json
from preprocessing import CLASSES, FEATURE_COLUMNS, text


def summarize(directory, source=None):
    directory = Path(directory)
    source = Path(source) if source else ROOT / "csc311_project_training_202609.csv"
    frame, protocol, _ = validate_inputs(source, ROOT / "development/protocol.json")
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete" or manifest["source_sha256_normalized"] != protocol["source_sha256"]:
        raise ValueError("Cannot summarize incomplete or incompatible evaluation results.")
    oof = pd.read_csv(directory / "oof_predictions.csv")
    if oof["row_index"].tolist() != list(range(len(frame))):
        raise ValueError("Outer predictions must cover every input row once, in original order.")
    if oof["true_label"].tolist() != frame["label"].tolist() or oof["id"].tolist() != frame["id"].tolist():
        raise ValueError("Outer prediction identities or labels do not match the source records.")
    probabilities = oof[[f"probability_{label}" for label in CLASSES]].to_numpy()
    measured = classification_metrics(oof["true_label"], probabilities)
    if abs(measured["accuracy"] - summary["oof_metrics"]["accuracy"]) > 1e-12:
        raise ValueError("Saved predictions and summary accuracy disagree.")
    correct = oof["true_label"] == oof["predicted_label"]
    confidence = probabilities.max(axis=1)
    true_positions = np.array([CLASSES.index(label) for label in oof["true_label"]])
    blank_counts = frame[FEATURE_COLUMNS].apply(lambda col: col.map(lambda v: int(text(v) == ""))).sum(axis=1)
    diagnostics = oof.assign(confidence=confidence, probability_true_class=probabilities[np.arange(len(frame)), true_positions],
                             blank_answers=blank_counts.to_numpy())
    errors = diagnostics.loc[~correct].copy()
    for column in FEATURE_COLUMNS:
        errors[f"raw_{column}"] = frame.loc[errors["row_index"], column].to_numpy()
    errors.sort_values(["confidence", "row_index"], ascending=[False, True]).to_csv(directory / "errors.csv", index=False)
    matrix = np.asarray(measured["confusion_matrix"])
    pd.DataFrame(matrix, index=pd.Index(CLASSES, name="true_label"), columns=CLASSES).to_csv(directory / "confusion_matrix.csv")
    pairs = [{"true_label": actual, "predicted_label": predicted, "count": int(matrix[i, j])}
             for i, actual in enumerate(CLASSES) for j, predicted in enumerate(CLASSES) if i != j and matrix[i, j]]
    pairs.sort(key=lambda p: (-p["count"], p["true_label"], p["predicted_label"]))
    examples = errors.sort_values(["confidence", "row_index"], ascending=[False, True]).head(3)
    error_summary = {
        "model_family": summary["model_family"], "errors": int((~correct).sum()), "rows": len(frame),
        "common_confusions": pairs,
        "per_class_recall": {label: measured["per_class"][label]["recall"] for label in CLASSES},
        "errors_with_confidence_at_least_0_9": int(((~correct) & (confidence >= .9)).sum()),
        "rows_with_blank_answers": int((blank_counts > 0).sum()),
        "error_rate_with_blank_answers": float((~correct)[blank_counts > 0].mean()) if (blank_counts > 0).any() else None,
        "error_rate_without_blank_answers": float((~correct)[blank_counts == 0].mean()) if (blank_counts == 0).any() else None,
        "high_confidence_error_examples": examples[["row_index", "id", "true_label", "predicted_label", "confidence", "probability_true_class"]].to_dict(orient="records"),
        "interpretation": "Outer-fold diagnostic evidence only. Associations are not causal; "
                          "these errors do not trigger feature changes or retuning. Historical feature-design exposure remains.",
    }
    write_json(directory / "error_summary.json", error_summary)

    figures = directory / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    title = summary["model_family"].replace("_", " ").capitalize()
    fig, axis = plt.subplots(figsize=(6, 5))
    axis.imshow(matrix, cmap="Blues", vmin=0, vmax=matrix.max())
    axis.set(xticks=range(3), yticks=range(3), xticklabels=CLASSES, yticklabels=CLASSES,
             xlabel="Predicted sport", ylabel="Actual sport", title=f"{title}: outer-fold predictions")
    for i in range(3):
        for j in range(3):
            axis.text(j, i, f"{matrix[i,j]}\n{100 * matrix[i,j] / matrix[i].sum():.1f}%", ha="center", va="center",
                      color="white" if matrix[i, j] > matrix.max() / 2 else "#152b40")
    fig.tight_layout()
    fig.savefig(figures / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    folds = json.loads((directory / "outer_folds.json").read_text(encoding="utf-8"))
    fig, axis = plt.subplots(figsize=(7, 4))
    positions = np.arange(len(folds))
    axis.bar(positions - .18, [f["training_metrics"]["accuracy"] * 100 for f in folds], width=.36, label="Outer training", color="#7796ad")
    axis.bar(positions + .18, [f["validation_metrics"]["accuracy"] * 100 for f in folds], width=.36, label="Outer validation", color="#356a9a")
    axis.set(xticks=positions, xticklabels=[str(f["outer_fold"]) for f in folds], xlabel="Recorded outer fold", ylabel="Accuracy (%)",
             ylim=(0, 105), title=f"{title}: training and outer validation")
    axis.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(figures / "outer_accuracy.png", dpi=180)
    plt.close(fig)

    search = json.loads((directory / "inner_search_summary.json").read_text(encoding="utf-8"))
    if summary["model_family"] == "softmax":
        fig, axes = plt.subplots(1, 5, figsize=(14, 3.4), sharey=True)
        for outer, axis in enumerate(axes):
            rows = sorted([r for r in search if r["outer_fold"] == outer and r["eligible"]], key=lambda r: r["parameters"]["C"])
            axis.semilogx([r["parameters"]["C"] for r in rows], [r["mean_train_accuracy"] * 100 for r in rows], "o--", label="Inner training", color="#7796ad")
            axis.semilogx([r["parameters"]["C"] for r in rows], [r["mean_validation_accuracy"] * 100 for r in rows], "o-", label="Inner validation", color="#356a9a")
            axis.axvline(folds[outer]["parameters"]["C"], color="#b86f28", linestyle=":", label="Selected C")
            axis.set(title=f"Outer fold {outer}", xlabel="C", ylim=(0, 105))
        axes[0].set_ylabel("Mean inner-fold accuracy (%)")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
        fig.tight_layout(rect=(0, 0, 1, .9))
        fig.savefig(figures / "inner_search_accuracy.png", dpi=180)
        plt.close(fig)
        # A compact validation-only plot keeps labels readable in the report.
        fig, axis = plt.subplots(figsize=(7, 2.6))
        colors = ["#356a9a", "#b86f28", "#42876b", "#8a6599", "#b35167"]
        for outer, color in enumerate(colors):
            rows = sorted([r for r in search if r["outer_fold"] == outer and r["eligible"]], key=lambda r: r["parameters"]["C"])
            axis.semilogx([r["parameters"]["C"] for r in rows], [r["mean_validation_accuracy"] * 100 for r in rows],
                          "o-", label=f"Fold {outer}", color=color, linewidth=1.4, markersize=4)
            selected = next(r for r in rows if r["parameters"] == folds[outer]["parameters"])
            axis.scatter([selected["parameters"]["C"]], [selected["mean_validation_accuracy"] * 100],
                         marker="*", s=90, color=color, edgecolors="black", linewidths=.4, zorder=4)
        axis.set(xlabel="C (inverse regularization strength)", ylabel="Inner validation\naccuracy (%)", ylim=(84, 95))
        axis.legend(loc="upper center", ncol=5, frameon=False, fontsize=9)
        fig.tight_layout()
        fig.savefig(figures / "inner_validation_accuracy.png", dpi=180)
        plt.close(fig)
    return error_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.results, args.source), indent=2))


if __name__ == "__main__":
    main()
