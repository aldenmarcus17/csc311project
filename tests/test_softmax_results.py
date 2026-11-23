"""Integrity checks on the completed softmax experiment and portable fold exports."""
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from models.softmax_inference import predict_proba
from preprocessing import CLASSES, load_state, read_survey

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/softmax"


@unittest.skipUnless((RESULTS / "summary.json").exists(), "Softmax results not present")
class CompletedRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = json.loads((RESULTS / "summary.json").read_text())
        cls.manifest = json.loads((RESULTS / "run_manifest.json").read_text())
        cls.oof = pd.read_csv(RESULTS / "oof_predictions.csv")
        cls.frame = read_survey(ROOT / "csc311_project_training_202609.csv", require_label=True)

    def test_all_respondents_and_records_predicted_once(self):
        self.assertEqual(self.oof["row_index"].tolist(), list(range(870)))
        self.assertEqual(self.oof["id"].nunique(), 290)
        self.assertTrue((self.oof.groupby("id")["outer_fold"].nunique() == 1).all())
        self.assertEqual(self.oof.groupby("outer_fold").size().tolist(), [174] * 5)

    def test_reported_metrics_agree_with_independent_counts(self):
        correct = self.oof["true_label"] == self.oof["predicted_label"]
        self.assertAlmostEqual(correct.mean(), self.summary["oof_metrics"]["accuracy"])
        matrix = [[int(((self.oof.true_label == actual) & (self.oof.predicted_label == predicted)).sum())
                   for predicted in CLASSES] for actual in CLASSES]
        self.assertEqual(matrix, self.summary["oof_metrics"]["confusion_matrix"])
        self.assertEqual(self.manifest["status"], "complete")
        self.assertFalse(self.summary["final_model_selected"])
        self.assertFalse(self.summary["full_data_refit_performed"])

    def test_selected_parameters_use_inner_scores_only(self):
        summaries = json.loads((RESULTS / "inner_search_summary.json").read_text())
        folds = json.loads((RESULTS / "outer_folds.json").read_text())
        for fold in folds:
            choices = [r for r in summaries if r["outer_fold"] == fold["outer_fold"] and r["eligible"]]
            maximum = max(r["mean_validation_accuracy"] for r in choices)
            candidates = [r for r in choices if abs(r["mean_validation_accuracy"] - maximum) <= 1e-12]
            expected = min(r["parameters"]["C"] for r in candidates)
            self.assertEqual(fold["parameters"]["C"], expected)

    def test_all_reference_exports_reproduce_saved_outer_predictions(self):
        for fold in range(5):
            rows = self.oof[self.oof.outer_fold == fold]
            directory = RESULTS / "fold_models" / f"fold_{fold}"
            state = load_state(directory / "preprocessing_state.json")
            model = json.loads((directory / "model.json").read_text())
            frame = self.frame.iloc[rows["row_index"]].drop(columns=["id", "label"])
            actual = predict_proba(frame, state, model)
            expected = rows[[f"probability_{label}" for label in CLASSES]].to_numpy()
            np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
            self.assertEqual(np.asarray(CLASSES)[actual.argmax(axis=1)].tolist(), rows.predicted_label.tolist())
            self.assertEqual(state["fit_rows"], 696)

    def test_result_and_source_hashes_match_provenance(self):
        for name, digest in self.manifest["code_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest, name)
        for filename, key in [("oof_predictions.csv", "oof_predictions_sha256"), ("summary.json", "summary_sha256")]:
            self.assertEqual(hashlib.sha256((RESULTS / filename).read_bytes()).hexdigest(), self.manifest[key])


if __name__ == "__main__":
    unittest.main()
