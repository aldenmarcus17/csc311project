"""Checks for comparable metrics, inner-only tuning, isolation and export parity."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning

from experiments.common import (ModelSpec, align_probabilities, choose_candidate,
                                classification_metrics, fit_recorded, prepare_partition,
                                search_inner, validate_inputs)
from models.softmax import expand_candidates, export_model, make_estimator
from models.softmax_inference import predict_proba
from preprocessing import CLASSES, FEATURE_COLUMNS, fit_preprocessor, save_state, transform_features

ROOT = Path(__file__).resolve().parents[1]


def synthetic_survey():
    rows = []
    for person in range(12):
        for index, label in enumerate(CLASSES):
            row = {col: "" for col in FEATURE_COLUMNS}
            row.update(id=person + 1, label=label, ease=[2, 6, 8][index],
                       strategy="Agree", pay=[50, 100, 200][index],
                       food=["hot dog", "pizza", "water"][index],
                       song=["Take me out to the ballgame", "Drake", "Waka Waka"][index])
            rows.append(row)
    return pd.DataFrame(rows)


class MetricsTests(unittest.TestCase):
    def test_known_confusion_matrix_and_zero_prediction_class(self):
        y = np.array(["Baseball", "Basketball", "Soccer"])
        p = np.array([[.9, .05, .05], [.6, .3, .1], [.1, .8, .1]])
        metrics = classification_metrics(y, p)
        self.assertEqual(metrics["confusion_matrix"], [[1, 0, 0], [1, 0, 0], [0, 1, 0]])
        self.assertAlmostEqual(metrics["accuracy"], 1 / 3)
        self.assertAlmostEqual(metrics["macro_f1"], 2 / 9)
        self.assertEqual(metrics["per_class"]["Soccer"]["precision"], 0)

    def test_probability_validation(self):
        for probabilities in [np.array([[.3, .3, .3]]), np.array([[np.nan, .5, .5]]),
                              np.array([[-.1, .6, .5]])]:
            with self.assertRaises(ValueError):
                classification_metrics(["Baseball"], probabilities)

    def test_estimator_class_order_is_aligned(self):
        class Reversed:
            classes_ = np.array(CLASSES[::-1])

            def predict_proba(self, X):
                return np.tile([.6, .3, .1], (len(X), 1))
        np.testing.assert_allclose(align_probabilities(Reversed(), np.ones((2, 1))),
                                   [[.1, .3, .6], [.1, .3, .6]])


class SelectionTests(unittest.TestCase):
    def test_accuracy_then_declared_complexity_with_failed_candidate_excluded(self):
        summaries = [
            {"eligible": True, "parameters": {"C": 1}, "mean_validation_accuracy": .8},
            {"eligible": True, "parameters": {"C": .1}, "mean_validation_accuracy": .8},
            {"eligible": False, "parameters": {"C": .01}, "mean_validation_accuracy": None},
        ]
        self.assertEqual(choose_candidate(summaries, lambda p: p["C"])["parameters"], {"C": .1})
        summaries[0]["mean_validation_accuracy"] = .81
        self.assertEqual(choose_candidate(summaries, lambda p: p["C"])["parameters"], {"C": 1})

    def test_predeclared_boundary_expansion(self):
        self.assertEqual(expand_candidates({"C": .01}, []), [{"C": .001}])
        self.assertEqual(expand_candidates({"C": 100}, []), [{"C": 1000.0}])
        self.assertEqual(expand_candidates({"C": 1}, []), [])

    def test_search_cannot_use_excluded_outer_respondent_labels(self):
        frame = synthetic_survey()
        def run(data):
            # Respondents 10-12 are excluded from every inner training/validation partition.
            partitions = [(fold, prepare_partition(data,
                np.array([i for i in range(27) if i // 9 != fold]),
                np.array([i for i in range(27) if i // 9 == fold]), scale_numeric=True))
                for fold in range(3)]
            spec = ModelSpec("dummy", True, [{"C": 1}, {"C": .1}],
                             lambda p, seed: DummyClassifier(strategy="prior"), lambda p: p["C"], {})
            rows, fit_log = [], []
            winner, _, _ = search_inner(spec, partitions, {"seed": 67, "thread_limit": 1}, 0, rows, fit_log)
            return winner["parameters"], [r["validation_accuracy"] for r in rows]
        original = run(frame)
        changed = frame.copy()
        changed.loc[27:, "label"] = "Baseball"
        self.assertEqual(run(changed), original)


class IsolationAndExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = synthetic_survey()
        cls.partition = prepare_partition(cls.frame, np.arange(27), np.arange(27, 36), scale_numeric=True)
        cls.estimator = make_estimator({"C": 1.0}, 67)
        cls.estimator.fit(cls.partition["X_train"], cls.partition["y_train"])
        cls.payload = export_model(cls.estimator, cls.partition["state"], {"C": 1.0})

    def test_partition_state_ignores_validation_statistics(self):
        changed = self.frame.copy()
        changed.loc[27:, "pay"] = 1e9
        second = prepare_partition(changed, np.arange(27), np.arange(27, 36), scale_numeric=True)
        self.assertEqual(second["state"], self.partition["state"])
        np.testing.assert_array_equal(second["X_train"], self.partition["X_train"])
        with self.assertRaises(ValueError):
            prepare_partition(self.frame, np.arange(5), np.arange(5, 8), scale_numeric=True)

    def test_json_export_probabilities_match_reference(self):
        saved_state = json.loads(json.dumps(self.partition["state"]))
        saved_model = json.loads(json.dumps(self.payload))
        actual = predict_proba(self.frame.iloc[27:].drop(columns=["id", "label"]), saved_state, saved_model)
        reference = align_probabilities(self.estimator, self.partition["X_validation"])
        np.testing.assert_allclose(actual, reference, atol=1e-12, rtol=1e-12)
        np.testing.assert_array_equal(actual.argmax(axis=1), reference.argmax(axis=1))

    def test_export_shape_and_order_mismatch_are_rejected(self):
        for change in ["class_order", "feature_order", "coef"]:
            bad = copy.deepcopy(self.payload)
            bad[change] = bad[change][::-1] if change != "coef" else bad[change][:-1]
            with self.assertRaises(ValueError):
                predict_proba(self.frame.iloc[27:], self.partition["state"], bad)

    def test_isolated_inference_without_sklearn_or_definitions_file(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            shutil.copy(ROOT / "preprocessing.py", path / "preprocessing.py")
            shutil.copy(ROOT / "models/softmax_inference.py", path / "softmax_inference.py")
            save_state(self.partition["state"], path / "preprocessing_state.json")
            (path / "model.json").write_text(json.dumps(self.payload))
            self.frame.iloc[27:].drop(columns=["id", "label"]).to_csv(path / "unlabeled.csv", index=False)
            result = subprocess.run([sys.executable, "-c",
                "import sys; sys.modules['sklearn']=None; "
                "from softmax_inference import predict_csv; "
                "p=predict_csv('unlabeled.csv','.'); "
                "assert p.tolist()==['Baseball','Basketball','Soccer']*3"],
                cwd=path, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_dataset_and_protocol_integrity(self):
        validate_inputs(ROOT / "csc311_project_training_202609.csv", ROOT / "development/protocol.json")

    def test_convergence_warning_retries_same_setting_once(self):
        import warnings
        class NeedsMoreIterations:
            def __init__(self):
                self.max_iter = 1000
            def get_params(self):
                return {"max_iter": self.max_iter}
            def set_params(self, **params):
                self.max_iter = params["max_iter"]
                return self
            def fit(self, X, y):
                if self.max_iter == 1000:
                    warnings.warn("needs retry", ConvergenceWarning)
                return self
        spec = ModelSpec("retry", True, [{}], lambda p, seed: NeedsMoreIterations(), lambda p: 0, {})
        log = []
        estimator, details = fit_recorded(spec, {}, self.partition, {"seed": 67, "thread_limit": 1}, {}, log)
        self.assertIsNotNone(estimator)
        self.assertEqual(estimator.max_iter, 5000)
        self.assertEqual(details["fit_attempts"], 2)
        self.assertEqual([entry["status"] for entry in log], ["nonconverged", "ok"])


if __name__ == "__main__":
    unittest.main()
