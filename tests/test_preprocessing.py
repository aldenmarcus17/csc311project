"""Behavior checks for training/inference consistency and respondent isolation."""
import copy
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing import (FEATURE_COLUMNS, fit_preprocessor, load_definitions,
                           load_state, parse_pay, read_survey, save_state,
                           transform_features)
from development_protocol import (build_protocol, inner_splits, load_protocol,
                                  outer_splits, validate_respondents, write_protocol)

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    frame = pd.DataFrame({col: ["", "", ""] for col in FEATURE_COLUMNS})
    frame["ease"] = [2, 6, ""]
    frame["strategy"] = ["Agree", "Strongly disagree", ""]
    frame["pay"] = ["100", "300", ""]
    frame["food"] = ["hot dog", "pizza", ""]
    frame["where"] = ["Indoor facility", "  INDOOR facility , Outdoor public space ", ""]
    frame["song"] = ["Waka Waka", "", "N/A"]
    return frame


class PaymentTests(unittest.TestCase):
    def test_valid_amounts_and_ranges(self):
        for raw, expected in [("$1,000", 1000), ("CAD 150", 150), ("1k", 1000),
                              ("300-500", 400), ("400 - 700", 550),
                              ("$500 - $750", 625), ("300$-500$", 400),
                              ("100 to 200 CAD", 150), (".50", .5),
                              ("fifty dollars", 50), ("Zero. No interest", 0),
                              ("nothing", 0), ("not willing", 0)]:
            with self.subTest(raw=raw):
                self.assertEqual(parse_pay(raw)[0], expected)

    def test_ambiguous_or_invalid_amounts_are_not_averaged(self):
        for raw in ["", None, "N/A", "$200, $500 for a great game", "2 tickets for $100",
                    "5000 to 1000", "-100", "$-100", "pay -100", "-.50", "$-.50", "100 USD", "1e3",
                    "1.2.3", "1,50", "infinity", "nothing unless $100"]:
            with self.subTest(raw=raw):
                self.assertTrue(np.isnan(parse_pay(raw)[0]))


class TransformationTests(unittest.TestCase):
    def setUp(self):
        self.train = fixture()
        self.state = fit_preprocessor(self.train)

    def test_held_out_rows_never_refit_state(self):
        before = json.dumps(self.state, sort_keys=True)
        held = fixture().iloc[[0]].copy()
        held["ease"] = ""
        held["pay"] = ""
        transformed = transform_features(held, self.state)
        self.assertEqual(transformed.iloc[0]["ease"], 4)
        self.assertAlmostEqual(transformed.iloc[0]["pay_log"], np.log1p(200))
        held["pay"] = 1e9
        transform_features(held, self.state, scale_numeric=True)
        self.assertEqual(json.dumps(self.state, sort_keys=True), before)

    def test_saved_state_definitions_drive_transformation(self):
        customized = copy.deepcopy(self.state)
        customized["definitions"]["keywords"]["food"]["hotdog"] = "pizza"
        transformed = transform_features(self.train, customized)
        self.assertEqual(transformed["food_hotdog"].tolist(), [0, 1, 0])

    def test_unlabeled_rows_and_changed_ids_have_identical_features(self):
        expected = transform_features(self.train, self.state)
        labeled = self.train.assign(id=[1, 9, 100], label=["Soccer", "Baseball", "Basketball"])
        pd.testing.assert_frame_equal(expected, transform_features(labeled, self.state))
        self.assertNotIn("id", expected)
        self.assertNotIn("label", expected)
        self.assertEqual(expected.shape[1], 73)

    def test_reordered_input_columns_and_rows_preserve_mapping(self):
        expected = transform_features(self.train, self.state)
        actual = transform_features(self.train.iloc[::-1, ::-1], self.state)
        pd.testing.assert_frame_equal(actual, expected.iloc[::-1])

    def test_schema_and_optional_columns(self):
        definitions = load_definitions()
        raw = self.train.rename(columns={v: k for k, v in definitions["raw_columns"].items()})
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "unlabeled.csv"
            raw.to_csv(path, index=False)
            parsed = read_survey(path, state=self.state)
        pd.testing.assert_frame_equal(transform_features(parsed, self.state), transform_features(self.train, self.state))
        with self.assertRaises(ValueError):
            transform_features(self.train.drop(columns="food"), self.state)

    def test_all_missing_and_constant_fallbacks_are_finite(self):
        missing = pd.DataFrame({col: [None, ""] for col in FEATURE_COLUMNS})
        state = fit_preprocessor(missing)
        transformed = transform_features(missing, state, scale_numeric=True)
        self.assertTrue(np.isfinite(transformed.to_numpy()).all())
        self.assertEqual(state["imputation"], {"ease": 5.0, "pay": 0.0})
        self.assertEqual(transformed["song_named"].tolist(), [0, 0])
        self.assertEqual(transform_features(self.train.iloc[:0], self.state).shape, (0, 73))

    def test_scaling_uses_population_training_statistics_only(self):
        transformed = transform_features(self.train, self.state, scale_numeric=True)
        for col in self.state["scaling"]:
            self.assertAlmostEqual(transformed[col].mean(), 0)
            if transformed[col].std(ddof=0) > 0:
                self.assertAlmostEqual(transformed[col].std(ddof=0), 1)
        self.assertEqual(transformed["where_indoor_facility"].tolist(), [1, 1, 0])

    def test_state_roundtrip_is_exact(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "state.json"
            save_state(self.state, path)
            loaded = load_state(path)
        pd.testing.assert_frame_equal(transform_features(self.train, self.state),
                                      transform_features(self.train, loaded))

    def test_inference_bundle_does_not_need_external_definitions(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            shutil.copyfile(ROOT / "preprocessing.py", directory / "preprocessing.py")
            save_state(self.state, directory / "state.json")
            self.train.to_csv(directory / "unlabeled.csv", index=False)
            result = subprocess.run(
                [sys.executable, "-c", "from preprocessing import load_state, read_survey, transform_features; "
                 "s=load_state('state.json'); d=read_survey('unlabeled.csv',state=s); "
                 "assert transform_features(d,s).shape == (3,73)"],
                cwd=directory, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_csv_preserves_literal_no_association_answers(self):
        frame = self.train.copy()
        frame["food"] = ["N/A", "None", ""]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "answers.csv"
            frame.to_csv(path, index=False)
            loaded = read_survey(path, state=self.state)
        self.assertEqual(loaded["food"].tolist(), ["N/A", "None", ""])
        self.assertEqual(transform_features(loaded, self.state)["food_none"].tolist(), [1, 1, 0])


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = read_survey(ROOT / "csc311_project_training_202609.csv", require_label=True)
        cls.protocol = load_protocol(ROOT / "development/protocol.json")

    def test_all_rows_covered_once_as_outer_validation(self):
        validation_rows = []
        for _, train, val in outer_splits(self.frame, self.protocol):
            self.assertEqual(len(set(self.frame.iloc[val]["id"])), 58)
            self.assertFalse(set(self.frame.iloc[train]["id"]) & set(self.frame.iloc[val]["id"]))
            self.assertEqual(len(train), 696)
            validation_rows.extend(val)
        self.assertEqual(sorted(validation_rows), list(range(len(self.frame))))

    def test_inner_splits_exclude_outer_validation(self):
        for outer, outer_train, outer_val in outer_splits(self.frame, self.protocol):
            covered = []
            for _, train, val in inner_splits(self.frame, self.protocol, outer):
                self.assertFalse(set(train) & set(outer_val))
                self.assertFalse(set(val) & set(outer_val))
                self.assertFalse(set(self.frame.iloc[train]["id"]) & set(self.frame.iloc[val]["id"]))
                self.assertEqual(set(train) | set(val), set(outer_train))
                covered.extend(val)
            self.assertEqual(sorted(covered), sorted(outer_train))

    def test_reproducible_and_frozen_assignments(self):
        expected = build_protocol(self.frame, ROOT / "csc311_project_training_202609.csv")
        self.assertEqual(expected, self.protocol)
        with tempfile.TemporaryDirectory() as temp:
            write_protocol(expected, temp)
            changed = copy.deepcopy(expected)
            changed["seed"] += 1
            with self.assertRaises(ValueError):
                write_protocol(changed, temp)

    def test_invalid_group_structure_rejected(self):
        with self.assertRaises(ValueError):
            validate_respondents(self.frame.iloc[:-1])
        bad = self.frame.copy()
        bad["id"] = bad["id"].astype(float)
        bad.loc[0, "id"] = 1.5
        with self.assertRaises(ValueError):
            validate_respondents(bad)

    def test_protocol_checksum_is_portable_across_line_endings(self):
        data = (ROOT / "csc311_project_training_202609.csv").read_bytes().replace(b"\r\n", b"\n")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "source.csv"
            path.write_bytes(data)
            unix = build_protocol(self.frame, path)
            path.write_bytes(data.replace(b"\n", b"\r\n"))
            windows = build_protocol(self.frame, path)
        self.assertEqual(unix, windows)
        self.assertEqual(unix, self.protocol)


if __name__ == "__main__":
    unittest.main()
