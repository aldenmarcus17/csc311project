"""Shared, respondent-grouped nested evaluation for independently owned models.

All models consume the same recorded protocol and write the same result schema.
Only inner validation accuracy chooses hyperparameters. Outer labels are used
after the search, for reporting; they never enter candidate selection.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time
from typing import Callable
import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss,
                             precision_recall_fscore_support)
from threadpoolctl import threadpool_limits

from development_protocol import (build_protocol, inner_splits, load_protocol,
                                  outer_splits, sha256)
from preprocessing import (CLASSES, fit_preprocessor, load_state, read_survey,
                           save_state, transform_features)

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    scale_numeric: bool
    initial_candidates: list[dict]
    make_estimator: Callable
    tie_key: Callable
    description: dict
    expand_candidates: Callable | None = None
    export_model: Callable | None = None
    predict_exported: Callable | None = None
    source_files: tuple[str, ...] = ()


def write_json(path, content):
    Path(path).write_text(json.dumps(content, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def candidate_id(parameters):
    return json.dumps(parameters, sort_keys=True, separators=(",", ":"), allow_nan=False)


def align_probabilities(estimator, features):
    observed = list(estimator.classes_)
    if len(observed) != len(CLASSES) or set(observed) != set(CLASSES):
        raise ValueError("Estimator must fit all three recorded classes.")
    probabilities = np.asarray(estimator.predict_proba(features), dtype=float)
    if probabilities.shape != (len(features), len(CLASSES)):
        raise ValueError("Estimator probability shape is incorrect.")
    return probabilities[:, [observed.index(label) for label in CLASSES]]


def classification_metrics(labels, probabilities):
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities, dtype=float)
    if len(labels) == 0 or probabilities.shape != (len(labels), len(CLASSES)):
        raise ValueError("Nonempty labels and one three-class probability row per label are required.")
    if not set(labels).issubset(CLASSES):
        raise ValueError("Unexpected target label.")
    if (not np.isfinite(probabilities).all() or (probabilities < 0).any() or
            (probabilities > 1).any() or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-10)):
        raise ValueError("Probabilities must be finite, nonnegative and sum to one.")
    predictions = np.asarray(CLASSES)[probabilities.argmax(axis=1)]
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=CLASSES, zero_division=0)
    return {
        "rows": len(labels), "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, labels=CLASSES, average="macro", zero_division=0)),
        "log_loss": float(log_loss(labels, probabilities, labels=CLASSES)),
        "per_class": {label: {"precision": float(precision[i]), "recall": float(recall[i]),
                              "f1": float(f1[i]), "support": int(support[i])}
                      for i, label in enumerate(CLASSES)},
        "confusion_matrix": confusion_matrix(labels, predictions, labels=CLASSES).tolist(),
    }


def flat_metrics(metrics, prefix):
    result = {f"{prefix}_{name}": metrics[name] for name in ["accuracy", "macro_f1", "log_loss"]}
    for label, values in metrics["per_class"].items():
        for name, value in values.items():
            result[f"{prefix}_{name}_{label}"] = value
    return result


def choose_candidate(summaries, tie_key):
    valid = [s for s in summaries if s["eligible"]]
    if not valid:
        raise RuntimeError("No candidate completed every required inner fold successfully.")
    best_accuracy = max(s["mean_validation_accuracy"] for s in valid)
    tied = [s for s in valid if abs(s["mean_validation_accuracy"] - best_accuracy) <= 1e-12]
    return min(tied, key=lambda s: (tie_key(s["parameters"]), candidate_id(s["parameters"])))


def prepare_partition(frame, training_positions, validation_positions, *, scale_numeric):
    train = frame.iloc[training_positions]
    validation = frame.iloc[validation_positions]
    if set(training_positions) & set(validation_positions):
        raise ValueError("Training and validation rows overlap.")
    if set(train["id"]) & set(validation["id"]):
        raise ValueError("Training and validation respondents overlap.")
    if train.empty or validation.empty:
        raise ValueError("Training and validation partitions must both be nonempty.")
    state = fit_preprocessor(train)
    return {
        "state": state,
        "X_train": transform_features(train, state, scale_numeric=scale_numeric).to_numpy(),
        "X_validation": transform_features(validation, state, scale_numeric=scale_numeric).to_numpy(),
        "y_train": train["label"].to_numpy(), "y_validation": validation["label"].to_numpy(),
        "training_positions": np.asarray(training_positions), "validation_positions": np.asarray(validation_positions),
        "training_respondents": int(train["id"].nunique()),
        "validation_respondents": int(validation["id"].nunique()),
    }


def fit_recorded(spec, parameters, partition, contract, context, fit_log):
    estimator = spec.make_estimator(copy.deepcopy(parameters), contract["seed"])
    started = time.perf_counter()
    all_warnings = []
    for attempt in range(2):
        record = {**context, "parameters": parameters, "attempt": attempt + 1,
                  "max_iter": estimator.get_params().get("max_iter")}
        try:
            with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=contract["thread_limit"]):
                warnings.simplefilter("always")
                estimator.fit(partition["X_train"], partition["y_train"])
            messages = [f"{w.category.__name__}: {w.message}" for w in caught]
            all_warnings.extend(messages)
            convergence_warning = any(issubclass(w.category, ConvergenceWarning) for w in caught)
            record.update(status="nonconverged" if convergence_warning else "ok", warnings=messages,
                          iterations=np.asarray(getattr(estimator, "n_iter_", [])).astype(int).tolist())
            fit_log.append(record)
            if not convergence_warning:
                return estimator, {"fit_status": "ok", "fit_seconds": time.perf_counter() - started,
                                   "fit_attempts": attempt + 1, "warnings_json": json.dumps(all_warnings),
                                   "iterations_json": json.dumps(record["iterations"])}
            if attempt == 0 and "max_iter" in estimator.get_params() and estimator.max_iter < 5000:
                estimator.set_params(max_iter=5000)
                continue
            return None, {"fit_status": "nonconverged", "fit_seconds": time.perf_counter() - started,
                          "fit_attempts": attempt + 1, "warnings_json": json.dumps(all_warnings)}
        except Exception as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
            fit_log.append(record)
            return None, {"fit_status": "failed", "fit_seconds": time.perf_counter() - started,
                          "fit_attempts": attempt + 1, "error": record["error"],
                          "warnings_json": json.dumps(all_warnings)}


def evaluate_candidates(spec, candidates, partitions, contract, outer, stage, rows, fit_log):
    summaries = []
    for parameters in candidates:
        scores = []
        for inner, partition in partitions:
            context = {"model": spec.name, "outer_fold": outer, "inner_fold": inner,
                       "phase": "inner_search", "stage": stage}
            estimator, details = fit_recorded(spec, parameters, partition, contract, context, fit_log)
            row = {"outer_fold": outer, "inner_fold": inner, "stage": stage,
                   "candidate_id": candidate_id(parameters), "parameters_json": json.dumps(parameters, sort_keys=True),
                   "training_rows": len(partition["y_train"]), "validation_rows": len(partition["y_validation"]),
                   "training_respondents": partition["training_respondents"],
                   "validation_respondents": partition["validation_respondents"], **details}
            if estimator is not None:
                try:
                    train_metrics = classification_metrics(partition["y_train"], align_probabilities(estimator, partition["X_train"]))
                    val_metrics = classification_metrics(partition["y_validation"], align_probabilities(estimator, partition["X_validation"]))
                    row.update(flat_metrics(train_metrics, "train"))
                    row.update(flat_metrics(val_metrics, "validation"))
                    scores.append(row)
                except Exception as error:
                    row.update(fit_status="invalid_predictions", error=f"{type(error).__name__}: {error}")
            rows.append(row)
        complete = len(scores) == len(partitions)
        summaries.append({"outer_fold": outer, "stage": stage, "parameters": parameters,
                          "candidate_id": candidate_id(parameters), "eligible": complete,
                          "completed_inner_folds": len(scores),
                          "mean_validation_accuracy": float(np.mean([r["validation_accuracy"] for r in scores])) if complete else None,
                          "std_validation_accuracy": float(np.std([r["validation_accuracy"] for r in scores], ddof=1)) if complete else None,
                          "mean_validation_macro_f1": float(np.mean([r["validation_macro_f1"] for r in scores])) if complete else None,
                          "mean_train_accuracy": float(np.mean([r["train_accuracy"] for r in scores])) if complete else None})
    return summaries


def search_inner(spec, partitions, contract, outer, rows, fit_log):
    candidates = copy.deepcopy(spec.initial_candidates)
    if len({candidate_id(p) for p in candidates}) != len(candidates):
        raise ValueError("Initial search candidates contain duplicates.")
    summaries = evaluate_candidates(spec, candidates, partitions, contract, outer, "initial", rows, fit_log)
    initial_winner = choose_candidate(summaries, spec.tie_key)
    extra = spec.expand_candidates(copy.deepcopy(initial_winner["parameters"]), copy.deepcopy(summaries)) if spec.expand_candidates else []
    if len({candidate_id(p) for p in candidates + extra}) != len(candidates + extra):
        raise ValueError("Expanded search candidates contain duplicates.")
    if extra:
        summaries.extend(evaluate_candidates(spec, extra, partitions, contract, outer, "expansion", rows, fit_log))
    winner = choose_candidate(summaries, spec.tie_key)
    return winner, summaries, {"initial_winner": initial_winner["parameters"], "expanded_candidates": extra}


def validate_inputs(source, protocol_path):
    frame = read_survey(source, require_label=True)
    protocol = load_protocol(protocol_path)
    if build_protocol(frame, source) != protocol:
        raise ValueError("Dataset or respondent assignments differ from the frozen development protocol.")
    feature_spec = json.loads((ROOT / "development/feature_specification.json").read_text(encoding="utf-8"))
    if sha256(ROOT / "feature_definitions.json") != feature_spec["definitions_sha256"]:
        raise ValueError("Feature definitions changed; do not silently compare incompatible experiments.")
    if feature_spec["class_order"] != CLASSES:
        raise ValueError("Feature specification uses an unexpected class order.")
    return frame, protocol, feature_spec


def save_search(directory, search_rows, summaries, fit_log):
    if search_rows:
        pd.DataFrame(search_rows).to_csv(directory / "inner_search.csv", index=False)
    write_json(directory / "inner_search_summary.json", summaries)
    write_json(directory / "fit_log.json", fit_log)


def _git_head():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_nested_cv(spec, *, source, protocol_path, output, overwrite=False):
    source, protocol_path, output = Path(source), Path(protocol_path), Path(output)
    if (spec.export_model is None) != (spec.predict_exported is None):
        raise ValueError("Supply both export_model and predict_exported, or neither.")
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError("Output directory is not empty. Use a new output path or explicitly pass --overwrite.")
    frame, protocol, feature_spec = validate_inputs(source, protocol_path)
    contract_path = ROOT / "experiments/contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract["class_order"] != CLASSES or contract["outer_folds"] != protocol["outer_folds"] or contract["inner_folds"] != protocol["inner_folds"]:
        raise ValueError("Evaluation contract and recorded protocol differ.")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    code_paths = ["preprocessing.py", "development_protocol.py", "feature_definitions.json",
                  "experiments/common.py", "experiments/contract.json", *spec.source_files]
    manifest = {
        "version": 1, "status": "running", "model_family": spec.name,
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "base_commit": _git_head(),
        "source_sha256_normalized": protocol["source_sha256"], "source_sha256_bytes": sha256(source),
        "protocol_sha256": sha256(protocol_path), "feature_specification_sha256": sha256(ROOT / "development/feature_specification.json"),
        "contract_sha256": sha256(contract_path), "class_order": CLASSES,
        "feature_order": feature_spec["feature_order"], "scale_numeric": spec.scale_numeric,
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "sklearn": sklearn.__version__, "initial_candidates": spec.initial_candidates,
        "model_description": spec.description,
        "code_sha256": {name: sha256(ROOT / name) for name in code_paths},
        "final_model_selected": False, "full_data_refit_performed": False,
        "limitation": contract["limitation"],
    }
    write_json(output / "run_manifest.json", manifest)
    search_rows, all_summaries, fit_log, folds, predictions = [], [], [], [], []
    try:
        for outer, training_rows, validation_rows in outer_splits(frame, protocol):
            print(f"{spec.name}: outer fold {outer + 1}/{protocol['outer_folds']} - inner search", flush=True)
            partitions = [(inner, prepare_partition(frame, train, val, scale_numeric=spec.scale_numeric))
                          for inner, train, val in inner_splits(frame, protocol, outer)]
            winner, summaries, expansion = search_inner(spec, partitions, contract, outer, search_rows, fit_log)
            all_summaries.extend(summaries)
            save_search(output, search_rows, all_summaries, fit_log)
            partition = prepare_partition(frame, training_rows, validation_rows, scale_numeric=spec.scale_numeric)
            if partition["state"]["feature_order"] != feature_spec["feature_order"]:
                raise ValueError("Preprocessing feature order differs from the shared feature specification.")
            estimator, details = fit_recorded(spec, winner["parameters"], partition, contract,
                                               {"model": spec.name, "outer_fold": outer, "phase": "outer_refit"}, fit_log)
            if estimator is None:
                raise RuntimeError(f"Outer fold {outer} refit failed: {details}")
            validation_probabilities = align_probabilities(estimator, partition["X_validation"])
            training_metrics = classification_metrics(partition["y_train"], align_probabilities(estimator, partition["X_train"]))
            validation_metrics = classification_metrics(partition["y_validation"], validation_probabilities)
            artifact_directory = output / "fold_models" / f"fold_{outer}"
            artifact_directory.mkdir(parents=True, exist_ok=True)
            save_state(partition["state"], artifact_directory / "preprocessing_state.json")
            parity = {"checked": False}
            if spec.export_model:
                payload = spec.export_model(estimator, partition["state"], winner["parameters"])
                write_json(artifact_directory / "model.json", payload)
                loaded = json.loads((artifact_directory / "model.json").read_text(encoding="utf-8"))
                saved_state = load_state(artifact_directory / "preprocessing_state.json")
                portable = spec.predict_exported(frame.iloc[validation_rows].drop(columns=["id", "label"]), saved_state, loaded)
                max_difference = float(np.max(np.abs(portable - validation_probabilities)))
                same_labels = bool(np.array_equal(portable.argmax(axis=1), validation_probabilities.argmax(axis=1)))
                if not same_labels or not np.allclose(portable, validation_probabilities, atol=1e-12, rtol=1e-12):
                    raise ValueError("Portable export differs from the reference estimator.")
                parity = {"checked": True, "identical_labels": same_labels, "max_absolute_probability_difference": max_difference,
                          "rows": len(validation_rows)}
            fold_result = {"outer_fold": outer, "parameters": winner["parameters"], **expansion,
                           "inner_selection_accuracy": winner["mean_validation_accuracy"],
                           "training_rows": len(training_rows), "validation_rows": len(validation_rows),
                           "training_respondents": partition["training_respondents"],
                           "validation_respondents": partition["validation_respondents"],
                           "training_metrics": training_metrics, "validation_metrics": validation_metrics,
                           "export_parity": parity, **details}
            folds.append(fold_result)
            prediction = pd.DataFrame({"row_index": validation_rows, "id": frame.iloc[validation_rows]["id"].to_numpy(),
                                       "outer_fold": outer, "true_label": partition["y_validation"],
                                       "predicted_label": np.asarray(CLASSES)[validation_probabilities.argmax(axis=1)]})
            for i, label in enumerate(CLASSES):
                prediction[f"probability_{label}"] = validation_probabilities[:, i]
            prediction.to_csv(output / f"outer_fold_{outer}_predictions.csv", index=False)
            predictions.append(prediction)
            write_json(output / "outer_folds.json", folds)
            save_search(output, search_rows, all_summaries, fit_log)
            print(f"{spec.name}: outer fold {outer + 1} complete; selected {winner['parameters']}", flush=True)

        oof = pd.concat(predictions).sort_values("row_index").reset_index(drop=True)
        if oof["row_index"].tolist() != list(range(len(frame))):
            raise ValueError("Outer predictions do not cover every original record exactly once.")
        oof.to_csv(output / "oof_predictions.csv", index=False)
        probabilities = oof[[f"probability_{label}" for label in CLASSES]].to_numpy()
        aggregate = classification_metrics(oof["true_label"].to_numpy(), probabilities)
        fold_scores = [f["validation_metrics"]["accuracy"] for f in folds]
        summary = {
            "version": 1, "model_family": spec.name, "evaluation": "recorded nested respondent-grouped development CV",
            "class_order": CLASSES, "respondents": int(frame["id"].nunique()), "rows": len(frame),
            "oof_metrics": aggregate, "mean_outer_accuracy": float(np.mean(fold_scores)),
            "sample_std_outer_accuracy": float(np.std(fold_scores, ddof=1)),
            "mean_outer_macro_f1": float(np.mean([f["validation_metrics"]["macro_f1"] for f in folds])),
            "mean_training_accuracy": float(np.mean([f["training_metrics"]["accuracy"] for f in folds])),
            "outer_fold_accuracies": fold_scores, "selected_parameters_by_fold": [f["parameters"] for f in folds],
            "initial_search_fits": sum(r["stage"] == "initial" for r in search_rows),
            "expansion_search_fits": sum(r["stage"] == "expansion" for r in search_rows),
            "fit_attempts": len(fit_log), "outer_refits": len(folds),
            "unresolved_inner_failures": sum(r["fit_status"] != "ok" for r in search_rows),
            "convergence_retries": sum(r["attempt"] > 1 for r in fit_log),
            "export_parity": [f["export_parity"] for f in folds],
            "final_model_selected": False, "full_data_refit_performed": False,
            "limitation": contract["limitation"],
        }
        write_json(output / "summary.json", summary)
        pd.DataFrame([{"outer_fold": f["outer_fold"], "parameters_json": json.dumps(f["parameters"]),
                       **flat_metrics(f["training_metrics"], "train"),
                       **flat_metrics(f["validation_metrics"], "validation")} for f in folds]).to_csv(output / "outer_metrics.csv", index=False)
        save_search(output, search_rows, all_summaries, fit_log)
        manifest.update(status="complete", finished_at_utc=datetime.now(timezone.utc).isoformat(),
                        elapsed_seconds=time.perf_counter() - started,
                        oof_predictions_sha256=sha256(output / "oof_predictions.csv"), summary_sha256=sha256(output / "summary.json"))
        write_json(output / "run_manifest.json", manifest)
        return summary
    except Exception as error:
        save_search(output, search_rows, all_summaries, fit_log)
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}", elapsed_seconds=time.perf_counter() - started)
        write_json(output / "run_manifest.json", manifest)
        raise
