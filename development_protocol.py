"""Recorded respondent-grouped development partitions; no pristine local test set."""
from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 67
OUTER_FOLDS = 5
INNER_FOLDS = 3


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_respondents(frame):
    if "id" not in frame or "label" not in frame:
        raise ValueError("Development data requires id and label columns.")
    numeric = pd.to_numeric(frame["id"], errors="coerce")
    if numeric.isna().any() or not np.isfinite(numeric).all() or (numeric % 1 != 0).any():
        raise ValueError("Development respondent IDs must be finite integers.")
    groups = frame.assign(id=numeric.astype(int)).groupby("id")["label"]
    expected = {"Baseball", "Basketball", "Soccer"}
    if any(len(values) != 3 or set(values) != expected for _, values in groups):
        raise ValueError("Expected exactly one response per sport for each development respondent.")
    return sorted(numeric.astype(int).unique().tolist())


def _assign(ids, folds, seed):
    shuffled = sorted(ids)
    random.Random(seed).shuffle(shuffled)
    return {int(person): position % folds for position, person in enumerate(shuffled)}


def build_protocol(frame, source):
    ids = validate_respondents(frame)
    if len(ids) < OUTER_FOLDS + INNER_FOLDS:
        raise ValueError("Too few respondents for the specified development folds.")
    outer = _assign(ids, OUTER_FOLDS, SEED)
    inner = {fold: _assign([p for p in ids if outer[p] != fold], INNER_FOLDS,
                          SEED + 1000 + fold) for fold in range(OUTER_FOLDS)}
    return {
        "version": 1, "seed": SEED,
        "source_sha256": hashlib.sha256(Path(source).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
        "source_hash_policy": "SHA256 of CSV bytes after CRLF-to-LF normalization for cross-platform checkouts",
        "status": "development_only_no_verified_unexamined_local_test",
        "inspection_history": "unknown; confirmed by group member on 2026-11-23",
        "limitation": "Historical raw-data inspection and dictionary design may involve any respondent. "
                      "Grouped evaluation does not remove historical feature-selection exposure.",
        "outer_folds": OUTER_FOLDS, "inner_folds": INNER_FOLDS,
        "assignment_algorithm": "Python random.Random(seed) shuffles sorted integer IDs; "
                                "fold = shuffled position modulo fold count. Inner seed = 1067 + outer fold.",
        "eda_training_rule": "outer_fold != 0; 232 respondents / 696 rows for the supplied dataset",
        "evaluation_rule": "Nested grouped development CV: tune within each outer training partition "
                           "using its recorded three inner folds; evaluate only the selected setting "
                           "on that outer validation fold. Retrospective evidence, not an untouched test estimate.",
        "respondents": [{"id": person, "outer_fold": outer[person],
                         "inner_fold_by_outer": {
                             str(fold): inner[fold].get(person) for fold in range(OUTER_FOLDS)}}
                        for person in ids],
    }


def write_protocol(protocol, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "protocol.json"
    if target.exists() and json.loads(target.read_text(encoding="utf-8")) != protocol:
        raise ValueError("Frozen protocol differs. Review source/history before changing assignments.")
    target.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    with (directory / "respondent_folds.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "status", "outer_fold", "eda_role"] +
                        [f"inner_fold_for_outer_{fold}" for fold in range(OUTER_FOLDS)])
        for item in protocol["respondents"]:
            writer.writerow([item["id"], "development", item["outer_fold"],
                             "development_validation" if item["outer_fold"] == 0 else "eda_training"] +
                            [item["inner_fold_by_outer"][str(f)] for f in range(OUTER_FOLDS)])


def load_protocol(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _row_folds(frame, protocol, outer=None):
    items = protocol["respondents"]
    mapping = {r["id"]: r["outer_fold"] if outer is None else
               r["inner_fold_by_outer"][str(outer)] for r in items}
    ids = pd.to_numeric(frame["id"], errors="raise")
    if not set(ids).issubset(mapping):
        raise ValueError("Respondent missing from recorded protocol.")
    return ids.map(mapping).to_numpy(dtype=float)


def outer_splits(frame, protocol):
    """Yield (fold, training row positions, validation row positions) for iloc."""
    folds = _row_folds(frame, protocol)
    for fold in range(protocol["outer_folds"]):
        yield fold, np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)


def inner_splits(frame, protocol, outer):
    """Positions refer to the provided frame; excluded outer rows appear in neither side."""
    folds = _row_folds(frame, protocol, outer)
    included = np.isfinite(folds)
    for fold in range(protocol["inner_folds"]):
        yield fold, np.flatnonzero(included & (folds != fold)), np.flatnonzero(folds == fold)


def exposure_audit(root, protocol):
    root = Path(root)
    history = root / "development/history"
    initial_path = history / "initial_split_manifest.json"
    proposal_path = history / "proposal_methodology_split_manifest.json"
    initial = json.loads(initial_path.read_text(encoding="utf-8"))
    proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
    submitted = {}
    sources = []
    for name in ["train", "val", "test"]:
        path = root / f"cleaned_data/{name}_clean.csv"
        # Read only identifiers. No historical held-out responses are profiled.
        with path.open(encoding="utf-8-sig", newline="") as stream:
            submitted[name] = {int(row["id"]) for row in csv.DictReader(stream)}
        sources.append({"path": str(path.relative_to(root)), "sha256": sha256(path)})
    earlier = set(initial["groups"]["train"])
    submitted_exposed = submitted["train"] | submitted["val"]
    known = earlier | submitted_exposed
    ids = {r["id"] for r in protocol["respondents"]}
    if set.union(*submitted.values()) != ids or sum(map(len, submitted.values())) != len(ids):
        raise ValueError("Historical split does not partition the recorded respondent IDs.")
    return {
        "version": 1, "inspection_history": "unknown", "verified_clean_local_test_ids": [],
        "decision": "All supplied respondents are development data. No untouched local test claim.",
        "initial_explored_ids": sorted(earlier),
        "submitted_groups": {name: sorted(values) for name, values in submitted.items()},
        "known_exposed_ids": sorted(known),
        "exposure_unknown_ids": sorted(ids - known),
        "submitted_test_previously_explored_ids": sorted(submitted["test"] & earlier),
        "methodology_test_explored_in_submission_ids": sorted(set(proposal["groups"]["test"]) & submitted_exposed),
        "counts": {"respondents": len(ids), "known_exposed": len(known),
                   "outside_recorded_exploration_but_exposure_unknown": len(ids - known),
                   "submitted_train_val_test": {k: len(v) for k, v in submitted.items()},
                   "methodology_train_validation_test": {k: len(v) for k, v in proposal["groups"].items()}},
        "sources": sources + [{"path": str(p.relative_to(root)), "sha256": sha256(p)}
                               for p in [initial_path, proposal_path]],
        "follow_up": "Proceed with all 290 respondents as development data under the recorded "
                     "grouped CV protocol, as decided by the group. Do not contact course staff about "
                     "test separation. Briefly disclose preliminary keyword-design exposure, "
                     "fold-specific preprocessing, and potentially optimistic evaluation estimates.",
    }
