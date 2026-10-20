"""Create exploratory plots from cleaned train and validation data.

Usage: python plot_distributions.py [cleaned_data_dir] [output_dir]

The test-cleaned file is intentionally not read.
"""
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd


INPUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "cleaned_data"
OUTPUT_DIR = sys.argv[2] if len(sys.argv) > 2 else os.path.join(INPUT_DIR, "plots")


def write_summary(data, path):
    """Write a feature and distribution summary for train plus validation."""
    feature_types = [
        ("id", "Identifier", "Respondent identifier; not a predictive feature"),
        ("ease", "Numeric", "Ease score, clipped to 0-10 and median-imputed"),
        ("strategy, injury, brilliance", "Ordinal", "Likert scores encoded from 1 to 5"),
        ("where_*, season_*, music_*", "Binary multi-hot", "One column per selected category"),
        ("food_*, colour_*, feel_*, song_*", "Binary keyword flags",
         "One column per matched keyword or category"),
        ("song_named", "Binary", "Whether a song was named"),
        ("pay_log", "Numeric", "log(1 + parsed pay), with missing values median-imputed"),
        ("label", "Categorical target", "Sport class to predict"),
    ]
    lines = [
        "# Processed feature summary",
        "",
        "This summary uses the combined training and validation data only "
        f"({len(data)} rows). The test set was not read or used.",
        "",
        "## Feature types",
        "",
        "| Feature(s) | Type | Description |",
        "|---|---|---|",
    ]
    lines.extend(f"| {features} | {kind} | {description} |"
                 for features, kind, description in feature_types)

    lines.extend(["", "## Class balance", "",
                  "| Class | Count | Percentage |",
                  "|---|---:|---:|"])
    class_counts = data["label"].value_counts().sort_index()
    lines.extend(
        f"| {label} | {count} | {100 * count / len(data):.1f}% |"
        for label, count in class_counts.items()
    )

    lines.extend(["", "## Numeric distributions", "",
                  "| Feature | Count | Mean | Std. dev. | Median | Minimum | Maximum |",
                  "|---|---:|---:|---:|---:|---:|---:|"])
    for column in ["ease", "pay_log"]:
        values = pd.to_numeric(data[column], errors="coerce").dropna()
        lines.append(
            f"| {column} | {len(values)} | {values.mean():.2f} | "
            f"{values.std():.2f} | {values.median():.2f} | "
            f"{values.min():.2f} | {values.max():.2f} |"
        )

    lines.extend(["", "## Likert distributions", "",
                  "Counts for the encoded ordinal scores (1 = strongly disagree, "
                  "5 = strongly agree).", ""])
    for column in ["strategy", "injury", "brilliance"]:
        counts = data[column].value_counts().reindex(range(1, 6), fill_value=0)
        distribution = ", ".join(f"{score}: {counts[score]}" for score in range(1, 6))
        lines.append(f"- **{column}** — {distribution}")

    lines.extend(["", "## Binary feature activation", "",
                  "| Feature group | Columns | Rows with at least one active feature |",
                  "|---|---:|---:|"])
    groups = {
        "where": [c for c in data if c.startswith("where_")],
        "season": [c for c in data if c.startswith("season_")],
        "music": [c for c in data if c.startswith("music_")],
        "food": [c for c in data if c.startswith("food_")],
        "colour": [c for c in data if c.startswith("colour_")],
        "feel": [c for c in data if c.startswith("feel_")],
        "song": [c for c in data if c.startswith("song_")],
    }
    for name, columns in groups.items():
        active = int((data[columns].sum(axis=1) > 0).sum())
        lines.append(f"| {name} | {len(columns)} | {active} ({100 * active / len(data):.1f}%) |")

    with open(path, "w", encoding="utf-8") as summary:
        summary.write("\n".join(lines) + "\n")


def save_class_conditioned_plots(data, output_dir):
    """Plot every processed feature's distribution or activation by class."""
    classes = sorted(data["label"].dropna().unique())
    colors = ["#4c78a8", "#f58518", "#54a24b"][:len(classes)]

    numeric = ["ease", "pay_log"]
    fig, axes = plt.subplots(1, len(numeric), figsize=(11, 4))
    for axis, column in zip(axes, numeric):
        values = [data.loc[data["label"] == label, column].dropna() for label in classes]
        axis.boxplot(values, tick_labels=classes)
        axis.set_title(f"{column} by class")
        axis.set_xlabel("Class")
        axis.set_ylabel("Value")
    fig.suptitle("Numeric feature distributions by class")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "numeric_distributions_by_class.png"), dpi=150)
    plt.close(fig)

    likert = ["strategy", "injury", "brilliance"]
    fig, axes = plt.subplots(1, len(likert), figsize=(14, 4), sharey=True)
    for axis, column in zip(axes, likert):
        counts = data.groupby(["label", column]).size().unstack(fill_value=0)
        percentages = counts.div(counts.sum(axis=1), axis=0).mul(100)
        percentages.reindex(columns=range(1, 6), fill_value=0).plot.bar(
            ax=axis, color=colors, rot=0
        )
        axis.set_title(column)
        axis.set_xlabel("Class")
        axis.set_ylabel("Percentage of class")
        axis.legend(title="Score", ncol=5)
    fig.suptitle("Likert feature distributions by class")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "likert_distributions_by_class.png"), dpi=150)
    plt.close(fig)

    groups = {
        "where": [c for c in data if c.startswith("where_")],
        "season": [c for c in data if c.startswith("season_")],
        "music": [c for c in data if c.startswith("music_")],
        "food": [c for c in data if c.startswith("food_")],
        "colour": [c for c in data if c.startswith("colour_")],
        "feel": [c for c in data if c.startswith("feel_")],
        "song": [c for c in data if c.startswith("song_")],
    }
    for group, columns in groups.items():
        rates = data.groupby("label")[columns].mean().mul(100).T
        axis = rates.plot.bar(figsize=(max(10, len(columns) * 0.45), 5), color=colors)
        axis.set_title(f"{group} feature activation by class")
        axis.set_xlabel("Processed feature")
        axis.set_ylabel("Percentage of class with feature active")
        axis.tick_params(axis="x", rotation=70)
        axis.legend(title="Class")
        axis.figure.tight_layout()
        axis.figure.savefig(
            os.path.join(output_dir, f"{group}_distributions_by_class.png"), dpi=150
        )
        plt.close(axis.figure)


def write_class_feature_summary(data, path):
    """Write class balance and every processed feature's class-conditioned rate."""
    classes = sorted(data["label"].dropna().unique())
    feature_columns = [
        column for column in data.columns if column not in {"id", "label"}
    ]
    lines = [
        "# Feature distributions by class",
        "",
        f"This report uses {len(data)} train+validation rows only. The test set was not read.",
        "",
        "## Class balance",
        "",
        "| Class | Rows | Percentage |",
        "|---|---:|---:|",
    ]
    counts = data["label"].value_counts().sort_index()
    lines.extend(
        f"| {label} | {count} | {100 * count / len(data):.1f}% |"
        for label, count in counts.items()
    )
    lines.extend(["", "## Feature distributions by class", "",
                  "For numeric and ordinal features, values are means. For binary "
                  "features, values are the percentage of rows in each class with "
                  "the feature active.", "",
                  "| Feature | Type | " + " | ".join(f"{label}" for label in classes) + " |",
                  "|---|---|" + "---:|" * len(classes)])
    for column in feature_columns:
        if column in {"ease", "pay_log"}:
            kind = "numeric"
        elif column in {"strategy", "injury", "brilliance"}:
            kind = "ordinal mean"
        else:
            kind = "binary activation"
        values = []
        for label in classes:
            series = pd.to_numeric(data.loc[data["label"] == label, column], errors="coerce")
            if kind == "binary activation":
                values.append(f"{100 * series.mean():.1f}%")
            else:
                values.append(f"{series.mean():.2f}")
        lines.append(f"| {column} | {kind} | " + " | ".join(values) + " |")
    with open(path, "w", encoding="utf-8") as summary:
        summary.write("\n".join(lines) + "\n")


def main():
    train = pd.read_csv(os.path.join(INPUT_DIR, "train_clean.csv"))
    val = pd.read_csv(os.path.join(INPUT_DIR, "val_clean.csv"))
    data = pd.concat([train, val], ignore_index=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    labels = data["label"].value_counts().sort_index()
    ax = labels.plot.bar(color=["#4c78a8", "#f58518", "#54a24b"], rot=0)
    ax.set_title("Class balance (train + validation)")
    ax.set_xlabel("Class")
    ax.set_ylabel("Rows")
    for index, count in enumerate(labels):
        ax.text(index, count, str(count), ha="center", va="bottom")
    ax.figure.tight_layout()
    ax.figure.savefig(os.path.join(OUTPUT_DIR, "class_balance.png"), dpi=150)
    plt.close(ax.figure)

    numeric = data[["ease", "pay_log"]].rename(
        columns={"ease": "Ease", "pay_log": "Log pay"}
    )
    axes = numeric.hist(bins=20, figsize=(10, 4), color="#4c78a8")
    axes[0, 0].set_xlabel("Ease score")
    axes[0, 0].set_ylabel("Rows")
    axes[0, 1].set_xlabel("log(1 + pay)")
    axes[0, 1].set_ylabel("Rows")
    axes[0, 0].figure.suptitle("Numeric feature distributions (train + validation)")
    axes[0, 0].figure.tight_layout()
    axes[0, 0].figure.savefig(
        os.path.join(OUTPUT_DIR, "numeric_distributions.png"), dpi=150
    )
    plt.close(axes[0, 0].figure)

    likert_cols = ["strategy", "injury", "brilliance"]
    likert = data[likert_cols].melt(var_name="Question", value_name="Score")
    ax = likert.groupby(["Question", "Score"]).size().unstack(fill_value=0).plot.bar(
        figsize=(9, 4), color="#72b7b2"
    )
    ax.set_title("Likert feature distributions (train + validation)")
    ax.set_xlabel("Question")
    ax.set_ylabel("Rows")
    ax.legend(title="Score", ncol=5)
    ax.figure.tight_layout()
    ax.figure.savefig(os.path.join(OUTPUT_DIR, "likert_distributions.png"), dpi=150)
    plt.close(ax.figure)

    groups = {
        "Food": [c for c in data if c.startswith("food_")],
        "Colour": [c for c in data if c.startswith("colour_")],
        "Feel": [c for c in data if c.startswith("feel_")],
        "Song": [c for c in data if c.startswith("song_")],
    }
    activation = pd.Series({
        name: 100 * (data[columns].sum(axis=1) > 0).mean()
        for name, columns in groups.items()
    })
    ax = activation.sort_values().plot.barh(figsize=(8, 4), color="#e45756")
    ax.set_title("Rows with at least one activated feature (train + validation)")
    ax.set_xlabel("Percentage of rows")
    ax.set_xlim(0, 100)
    ax.figure.tight_layout()
    ax.figure.savefig(os.path.join(OUTPUT_DIR, "feature_activation.png"), dpi=150)
    plt.close(ax.figure)
    write_summary(data, os.path.join(OUTPUT_DIR, "processed_feature_summary.md"))
    save_class_conditioned_plots(data, OUTPUT_DIR)
    write_class_feature_summary(
        data, os.path.join(OUTPUT_DIR, "feature_distributions_by_class.md")
    )


if __name__ == "__main__":
    main()
