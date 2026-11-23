"""Shared survey preprocessing. Runtime dependencies: NumPy and pandas only.

Fit a new state on EACH training partition. Transform validation and prediction
rows with that state. This module never fits statistics during transformation.
"""
from __future__ import annotations

import copy
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_COLUMNS = ["ease", "strategy", "injury", "brilliance", "where", "seasons",
                   "music", "food", "colour", "song", "feel", "pay"]
SCALED_COLUMNS = ["ease", "strategy", "injury", "brilliance", "pay_log"]
CLASSES = ["Baseball", "Basketball", "Soccer"]
MISSING_TEXT = {"", "nan", "null", "<na>"}


def load_definitions(path=None):
    path = Path(path) if path else Path(__file__).with_name("feature_definitions.json")
    return json.loads(path.read_text(encoding="utf-8"))


def text(value):
    """Normalize Unicode and whitespace; retain meaningful 'none' and 'N/A'."""
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value))).strip()


def canonicalize(frame, definitions, require_label=False):
    aliases = {text(k).casefold(): v for k, v in definitions["raw_columns"].items()}
    aliases.update({k: k for k in FEATURE_COLUMNS + ["id", "label"]})
    renamed = frame.rename(columns={c: aliases.get(text(c).casefold(), c) for c in frame})
    if renamed.columns.duplicated().any():
        raise ValueError("Duplicate columns after mapping survey headers.")
    missing = set(FEATURE_COLUMNS) - set(renamed.columns)
    if require_label and "label" not in renamed:
        missing.add("label")
    if missing:
        raise ValueError(f"Required CSV columns missing: {sorted(missing)}")
    return renamed.copy()


def read_survey(filename, *, state=None, require_label=False):
    definitions = state["definitions"] if state is not None else load_definitions()
    # Preserve literal 'N/A'/'None' answers; blank cells are handled by each rule.
    return canonicalize(pd.read_csv(filename, keep_default_na=False), definitions,
                        require_label=require_label)


def parse_pay(value):
    """Return (nonnegative amount or NaN, audit category).

    Accept one number, one unambiguous ascending range, simple English numbers,
    and explicit refusals. Reject unrelated multiple amounts, negative/nonfinite
    amounts and explicit non-CAD currencies instead of averaging all digits.
    Bare dollars are provisionally treated as CAD; no exchange rates are guessed.
    """
    raw = text(value).casefold()
    if raw in MISSING_TEXT:
        return np.nan, "blank"
    if re.search(r"\b(?:usd|eur|gbp|yen|rupees?|inr|cny)\b|us\s*\$|[\u20ac\u00a3\u00a5]", raw):
        return np.nan, "foreign_currency"
    if re.search(r"\b(?:inf(?:inity)?|nan)\b", raw):
        return np.nan, "invalid_number"
    # A sign directly before an amount (possibly with '$') is not a range.
    if re.search(r"^(?:\$\s*)?[-\u2212]\s*\$?\s*(?:\d|\.\d)", raw):
        return np.nan, "negative"
    if re.search(r"\d(?:\.\d+)?e[+-]?\d", raw):
        return np.nan, "invalid_number"
    refusal = re.search(
        r"\b(?:none|nothing|unwilling|not (?:really |too )?willing|no willing|"
        r"wouldn.?t|would not|won.?t|no i don|do not watch|not too much)\b|need to pay me",
        raw)
    number = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+"
    matches = list(re.finditer(rf"(?<![\w.])({number})\s*(k\b)?", raw))
    values = [float(m.group(1).replace(",", "")) * (1000 if m.group(2) else 1)
              for m in matches]
    if any(not np.isfinite(v) for v in values):
        return np.nan, "invalid_number"
    if len(values) == 2:
        between = raw[matches[0].end():matches[1].start()]
        if (re.fullmatch(r"\s*\$?\s*(?:-|\u2013|\u2014|to)\s*\$?\s*", between)
                and values[0] <= values[1] and not refusal):
            return float(np.mean(values)), "range"
        return np.nan, "ambiguous_multiple"
    if len(values) > 2 or (values and refusal):
        return np.nan, "ambiguous_multiple"
    if values:
        if re.search(r"(?:^|[\s(])[-\u2212]\s*\$?\s*(?:\d|\.\d)|\$\s*[-\u2212]\s*(?:\d|\.\d)", raw):
            return np.nan, "negative"
        # '1,50' and '1.2.3' must not be silently parsed as ordinary amounts.
        if re.search(r"\d,\d{1,2}(?:\D|$)|\d\.\d+\.", raw):
            return np.nan, "invalid_number"
        return values[0], "numeric"
    words = {"zero": 0, "ten": 10, "twenty": 20, "thirty": 30, "forty": 40,
             "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
             "ninety": 90, "one hundred": 100, "a hundred": 100,
             "two hundred": 200, "five hundred": 500, "one thousand": 1000}
    cleaned = re.sub(r"\s+(?:cad|dollars?|bucks)\.?$", "", raw).strip()
    if cleaned in words:
        return float(words[cleaned]), "written_number"
    if re.match(r"^zero(?:\b|[.!])", raw):
        return 0.0, "written_number"
    if refusal:
        return 0.0, "refusal"
    return np.nan, "unparsed_text"


def _slug(value):
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def _base_features(frame, state):
    definitions = state["definitions"]
    frame = canonicalize(frame, definitions)
    values = {}
    ease = pd.to_numeric(frame["ease"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    values["ease"] = ease.clip(0, 10).fillna(state["imputation"]["ease"])
    for col in definitions["likert_columns"]:
        values[col] = frame[col].map(lambda v: definitions["likert"].get(text(v).casefold(), 3))
    for col, options in definitions["multiselect"].items():
        selected = frame[col].map(lambda v: {text(x).casefold() for x in text(v).split(",")})
        prefix = "season" if col == "seasons" else col
        for option in options:
            values[f"{prefix}_{_slug(option)}"] = selected.map(lambda s, o=option.casefold(): int(o in s))
    for col in ["food", "colour", "song"]:
        normalized = frame[col].map(lambda v: text(v).casefold())
        for name, pattern in definitions["keywords"][col].items():
            name = name if col == "song" else f"{col}_{name}"
            values[name] = normalized.map(lambda v, p=pattern: int(bool(re.search(p, v))))
    no_song = re.compile(definitions["no_song_pattern"], re.I)
    values["song_named"] = frame["song"].map(
        lambda v: int(text(v).casefold() not in MISSING_TEXT and not no_song.search(text(v))))
    pay = frame["pay"].map(lambda v: parse_pay(v)[0]).fillna(state["imputation"]["pay"])
    values["pay_log"] = np.log1p(pay)
    normalized = frame["feel"].map(lambda v: text(v).casefold())
    for name, pattern in definitions["keywords"]["feel"].items():
        values[f"feel_{name}"] = normalized.map(lambda v, p=pattern: int(bool(re.search(p, v))))
    return pd.DataFrame(values, index=frame.index, dtype=float)


def fit_preprocessor(train, definitions=None):
    definitions = copy.deepcopy(definitions if definitions is not None else load_definitions())
    train = canonicalize(train, definitions)
    if train.empty:
        raise ValueError("Cannot fit preprocessing on zero training rows.")
    ease = pd.to_numeric(train["ease"], errors="coerce").replace([np.inf, -np.inf], np.nan).clip(0, 10)
    pay = train["pay"].map(lambda v: parse_pay(v)[0])
    state = {
        "version": 1, "definitions": definitions, "class_order": list(CLASSES),
        "fit_rows": len(train),
        "imputation": {"ease": float(ease.median()) if ease.notna().any() else 5.0,
                       "pay": float(pay.median()) if pay.notna().any() else 0.0},
        "all_missing_fallback": {"ease": not bool(ease.notna().any()),
                                 "pay": not bool(pay.notna().any())},
    }
    features = _base_features(train, state)
    state["feature_order"] = list(features.columns)
    state["scaling"] = {
        col: {"mean": float(features[col].mean()),
              "std": float(features[col].std(ddof=0)) or 1.0}
        for col in SCALED_COLUMNS
    }
    return state


def transform_features(frame, state, *, scale_numeric=False):
    if state.get("version") != 1:
        raise ValueError("Unsupported preprocessing state version.")
    features = _base_features(frame, state)
    if list(features.columns) != state["feature_order"]:
        raise ValueError("Saved feature order does not match saved definitions.")
    if scale_numeric:
        for col, stats in state["scaling"].items():
            features[col] = (features[col] - stats["mean"]) / stats["std"]
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError("Preprocessing produced nonfinite features.")
    return features


def save_state(state, filename):
    Path(filename).write_text(json.dumps(state, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_state(filename):
    return json.loads(Path(filename).read_text(encoding="utf-8"))
