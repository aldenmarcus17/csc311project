"""Reproduce the Member 1 development protocol and training-only exploration.

Run: python prepare_member1.py
Historical proposal scripts, cleaned_data outputs, and proposal.pdf are untouched.
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from development_protocol import (build_protocol, exposure_audit, outer_splits,
                                  sha256, write_protocol)
from preprocessing import (CLASSES, FEATURE_COLUMNS, fit_preprocessor, parse_pay,
                           read_survey, save_state, text, transform_features)

ROOT = Path(__file__).resolve().parent
TYPES = {"ease": "bounded numeric", "strategy": "ordinal Likert", "injury": "ordinal Likert",
         "brilliance": "ordinal Likert", "where": "multi-select", "seasons": "multi-select",
         "music": "multi-select", "food": "free text", "colour": "free text",
         "song": "free text", "feel": "free text", "pay": "free-text monetary amount"}


def write_json(path, content):
    path.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def describe_training(train, state, directory):
    directory.mkdir(parents=True, exist_ok=True)
    figures = directory / "figures"
    figures.mkdir(exist_ok=True)
    features = transform_features(train, state)
    scaled = transform_features(train, state, scale_numeric=True)
    train[["id", "label"]].to_csv(directory / "eda_row_metadata.csv", index=False)
    features.to_csv(directory / "eda_features_tree.csv", index=False)
    scaled.to_csv(directory / "eda_features_softmax.csv", index=False)
    save_state(state, directory / "eda_preprocessing_state.json")

    audit = []
    for col in FEATURE_COLUMNS:
        normalized = train[col].map(text)
        blank = normalized.eq("")
        audit.append({"feature": col, "type": TYPES[col], "rows": len(train),
                      "blank_count": int(blank.sum()), "blank_percent": float(blank.mean() * 100),
                      "distinct_nonblank": int(normalized[~blank].nunique()),
                      "median_text_length": float(normalized.str.len().median())})
    pd.DataFrame(audit).to_csv(directory / "raw_feature_audit.csv", index=False)
    counts = train["label"].value_counts().reindex(CLASSES)
    counts.rename_axis("class").to_frame("rows").to_csv(directory / "class_balance.csv")

    pay = train["pay"].map(parse_pay)
    pay_values = pay.map(lambda value: value[0])
    statuses = pay.map(lambda value: value[1])
    statuses.value_counts().rename_axis("status").to_frame("rows").to_csv(directory / "pay_parse_counts.csv")
    pd.DataFrame({"id": train["id"], "label": train["label"], "raw": train["pay"],
                  "parsed_pay": pay_values, "status": statuses}).to_csv(directory / "training_pay_audit.csv", index=False)
    ease_raw = pd.to_numeric(train["ease"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    numeric = features[["ease", "strategy", "injury", "brilliance", "pay_log"]].copy()
    numeric["pay_parsed_before_imputation"] = pay_values
    numeric.describe(percentiles=[.25, .5, .75, .95]).T.to_csv(directory / "numeric_summary.csv")
    numeric_by_class = features.assign(label=train["label"]).groupby("label")[["ease", "strategy", "injury", "brilliance", "pay_log"]].agg(["mean", "std"])
    numeric_by_class.to_csv(directory / "numeric_by_class.csv")
    likert_counts = []
    unknown_likert = {}
    for col in state["definitions"]["likert_columns"]:
        normalized = train[col].map(lambda v: text(v).casefold())
        invalid = ~normalized.isin(state["definitions"]["likert"])
        unknown_likert[col] = {"blank": int(normalized.eq("").sum()),
                               "unrecognized_nonblank": int((invalid & normalized.ne("")).sum())}
        for label in CLASSES:
            for score in range(1, 6):
                likert_counts.append({"feature": col, "class": label, "score": score,
                                      "count": int(((features[col] == score) & (train["label"] == label)).sum())})
    pd.DataFrame(likert_counts).to_csv(directory / "likert_by_class.csv", index=False)
    features.assign(label=train["label"]).groupby("label").mean().T.to_csv(directory / "feature_means_by_class.csv")

    unknown_options = {}
    for col, options in state["definitions"]["multiselect"].items():
        permitted = {o.casefold() for o in options}
        values = [token for value in train[col] for token in
                  {text(t).casefold() for t in text(value).split(",")} if token and token not in permitted]
        unknown_options[col] = pd.Series(values, dtype=str).value_counts().to_dict()
    coverage = {}
    for col in ["food", "colour", "feel", "song"]:
        columns = list(state["definitions"]["keywords"][col])
        if col != "song":
            columns = [f"{col}_{name}" for name in columns]
        matched = features[columns].sum(axis=1) > 0
        nonblank = train[col].map(text).ne("")
        coverage[col] = {"matched_rows": int(matched.sum()),
                         "percent_all_training_rows": float(100 * matched.mean()),
                         "percent_nonblank_training_rows": float(100 * matched[nonblank].mean()) if nonblank.any() else 0.0,
                         "blank_rows": int((~nonblank).sum()),
                         "unmatched_nonblank_top10": train.loc[~matched & nonblank, col].map(text).value_counts().head(10).to_dict()}
        train[col].map(lambda v: len(text(v))).describe().to_frame("characters").to_csv(directory / f"text_length_{col}.csv")

    outliers = {}
    for name, series in [("ease_raw", ease_raw), ("pay_parsed", pay_values)]:
        observed = series.dropna()
        q1, q3 = observed.quantile([.25, .75])
        iqr = q3 - q1
        outliers[name] = {"minimum": float(observed.min()), "maximum": float(observed.max()),
                          "iqr_outlier_count": int(((observed < q1 - 1.5 * iqr) | (observed > q3 + 1.5 * iqr)).sum())}
    summary = {
        "scope": "EDA training only: respondents with outer_fold != 0; development data, not a clean historical holdout protocol",
        "rows": len(train), "respondents": int(train["id"].nunique()), "features": features.shape[1],
        "binary_features": features.shape[1] - 5, "class_counts": counts.to_dict(),
        "raw_feature_audit": audit, "imputation": state["imputation"],
        "numeric": {col: {"mean": float(features[col].mean()), "std": float(features[col].std(ddof=1)),
                          "minimum": float(features[col].min()), "maximum": float(features[col].max())}
                    for col in ["ease", "strategy", "injury", "brilliance", "pay_log"]},
        "pay_parse_counts": statuses.value_counts().to_dict(),
        "pay_imputed_rows": int(pay_values.isna().sum()),
        "ease_invalid_or_blank_rows": int(ease_raw.isna().sum()),
        "ease_outside_0_10_rows": int(((ease_raw < 0) | (ease_raw > 10)).sum()),
        "likert_unknown": unknown_likert, "unknown_multiselect_options": unknown_options,
        "keyword_coverage": coverage, "outliers": outliers,
        "top_binary_class_ranges": (features.assign(label=train["label"]).groupby("label").mean()
                                   .drop(columns=["ease", "strategy", "injury", "brilliance", "pay_log"])
                                   .agg(lambda s: s.max() - s.min()).sort_values(ascending=False).head(8).to_dict()),
        "limitations": ["Prior manual dictionary design has unknown respondent exposure.",
                        "Validation responses were explored for the proposal.",
                        "The reported EDA excludes outer fold 0 now; this does not erase historical exposure.",
                        "No accuracy or generalization claim follows from these descriptive feature statistics."],
    }
    write_json(directory / "summary.json", summary)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = ["#356a9a", "#b86f28", "#42876b"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for axis, col, label in zip(axes, ["ease", "brilliance", "pay_log"],
                                ["Ease (0-10)", "Individual brilliance (1-5)", "log(1 + payment)"]):
        values = [features.loc[train["label"] == sport, col] for sport in CLASSES]
        boxes = axis.boxplot(values, tick_labels=CLASSES, patch_artist=True, widths=.6)
        for patch, color in zip(boxes["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(.7)
        axis.set_title(label)
        axis.tick_params(axis="x", labelrotation=15)
    fig.suptitle(f"EDA training respondents only: {len(train)} responses, {train['id'].nunique()} people", fontsize=11)
    fig.tight_layout()
    fig.savefig(figures / "numeric_by_class.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 4, figsize=(13, 10))
    for axis, col in zip(axes.flat, FEATURE_COLUMNS):
        if col == "ease":
            axis.hist(features[col], bins=np.arange(-.5, 11.5, 1), color=colors[0])
            axis.set_xlabel("Score")
        elif col in state["definitions"]["likert_columns"]:
            features[col].value_counts().reindex(range(1, 6), fill_value=0).plot.bar(ax=axis, color=colors[0], rot=0)
            axis.set_xlabel("Encoded score")
        elif col == "pay":
            axis.hist(features["pay_log"], bins=16, color=colors[0])
            axis.set_xlabel("log(1 + payment)")
        elif col in state["definitions"]["multiselect"]:
            prefix = "season_" if col == "seasons" else f"{col}_"
            cols = [c for c in features if c.startswith(prefix)]
            values = features[cols].mean().sort_values()
            axis.barh([c.removeprefix(prefix).replace("_", " ") for c in values.index], 100 * values, color=colors[0])
            axis.set_xlabel("Rows selecting option (%)")
            axis.tick_params(axis="y", labelsize=7)
        else:
            axis.hist(train[col].map(lambda v: len(text(v))), bins=16, color=colors[0])
            axis.set_xlabel("Answer length (characters)")
        axis.set_title(col)
    fig.suptitle("All input feature distributions - EDA training respondents only")
    fig.tight_layout()
    fig.savefig(figures / "all_feature_distributions.png", dpi=160)
    plt.close(fig)

    binary = features.assign(label=train["label"]).groupby("label").mean()
    chosen = list(summary["top_binary_class_ranges"])
    fig, axis = plt.subplots(figsize=(10, 4.4))
    binary.loc[CLASSES, chosen].T.mul(100).plot.barh(ax=axis, color=colors, width=.8)
    axis.set_xlabel("Training responses with active flag (%)")
    axis.set_ylabel("")
    axis.set_xlim(0, 105)
    axis.set_title("Largest class differences among fixed binary features", pad=37)
    axis.legend(loc="lower center", bbox_to_anchor=(.5, 1.01), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(figures / "binary_features_by_class.png", dpi=180)
    plt.close(fig)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "csc311_project_training_202609.csv")
    args = parser.parse_args()
    frame = read_survey(args.source, require_label=True)
    protocol = build_protocol(frame, args.source)
    directory = ROOT / "development"
    write_protocol(protocol, directory)
    audit = exposure_audit(ROOT, protocol)
    write_json(directory / "exposure_audit.json", audit)
    fold, training, validation = next(outer_splits(frame, protocol))
    train = frame.iloc[training].copy()
    state = fit_preprocessor(train)
    summary = describe_training(train, state, directory / "eda")
    write_json(directory / "feature_specification.json", {
        "version": 1, "feature_order": state["feature_order"], "class_order": state["class_order"],
        "excluded_predictors": ["id", "label"], "binary_features": summary["binary_features"],
        "scaled_for_softmax": list(state["scaling"]), "unscaled_for_trees": True,
        "fit_rule": "Fit a fresh state for every inner/outer training partition; EDA state is not a CV or final deployment state.",
        "definitions_sha256": sha256(ROOT / "feature_definitions.json"),
        "payment_rule": "Conservative CAD-compatible single amount/range parser v1; ambiguous, negative, nonfinite, foreign and unparsed values are training-median imputed.",
    })
    paths = [ROOT / n for n in ["preprocessing.py", "development_protocol.py", "prepare_member1.py", "feature_definitions.json"]]
    write_json(directory / "run_manifest.json", {
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__, "source_sha256": sha256(args.source),
        "scripts_sha256": {p.name: sha256(p) for p in paths},
        "protocol_sha256": sha256(directory / "protocol.json"),
        "eda_training_respondents": len(training) // 3,
        "eda_training_rows": len(training), "excluded_from_new_eda_rows": len(validation),
        "feature_matrix_sha256": sha256(directory / "eda/eda_features_tree.csv"),
    })
    print(json.dumps({"development_respondents": len(protocol["respondents"]),
                      "verified_clean_test_respondents": 0, "eda_training_rows": summary["rows"],
                      "features": summary["features"], "exposure_counts": audit["counts"]}, indent=2))


if __name__ == "__main__":
    main()
