# Shared model development

Each member owns one model family: softmax regression, decision tree or random
forest. All three use the same evaluation machinery. Do not pick the final family
until all three have been tuned and evaluated.

## Frozen comparison rules

- All 290 respondents / 870 rows are development data. There is no verified
  untouched local test group. Retain the brief historical feature-design disclosure.
- Use `development/protocol.json` and `development/respondent_folds.csv` exactly.
  Five outer folds contain 58 respondents / 174 rows each. For any outer fold,
  training contains 232 respondents / 696 rows and validation contains 58 / 174.
- Within that outer training partition, use its three recorded inner folds:
  78/77/77 validation respondents, with 154/155/155 training respondents. Each
  respondent's three answers stay together. These inner fold numbers are specific
  to that outer split; do not create new random row splits.
- Use the fixed 73 predictors in `development/feature_specification.json`. ID and
  label are not predictors. Keep class order Baseball, Basketball, Soccer.
- Refit preprocessing separately for each distinct training partition, then
  transform its validation data with that state. Caching one training partition's
  state across hyperparameter candidates is safe. Never reuse the EDA matrices or
  EDA preprocessing state for CV.
- Softmax scales ease, three Likert values and log-payment using training mean and
  population SD. Trees and forests use the same features without scaling.
- Select hyperparameters using mean INNER validation accuracy; exact numerical
  ties (within 1e-12) use the model's declared simpler-setting key. Macro-F1 and
  per-class metrics are diagnostics, not extra selection criteria.
- Record outer accuracy, macro-F1, class precision/recall/F1, confusion matrix and
  log loss. Use the same labels and `zero_division=0` convention for every family.
- Seeds are 67. The common runner limits numerical-library threads to one; forest
  adapters should also explicitly set `n_jobs=1` for comparable resource use.
- Predeclare ranges, tie rules and any range-expansion rule before looking at
  scores. Outer-fold errors must not trigger feature changes or retuning.

`experiments/contract.json` records these rules. The runner checks data, protocol
and fixed feature-definition checksums before fitting. Full provenance is written
to each family's `run_manifest.json`, including hashes of the adapter source files.
Match the tested scikit-learn version in `requirements-development.txt`.

## Softmax workstream

From the repository root:

```sh
python -m unittest discover -s tests -v
python -m experiments.run_softmax
python -m experiments.summarize_results --results results/softmax
python report/build_softmax_section.py
```

The default output directory must be empty. To reproduce an existing run, prefer
a separate output directory:

```sh
python -m experiments.run_softmax --output results/softmax_reproduction
```

`--overwrite` explicitly rewrites known output files in the requested directory;
use it only when replacing that run is intended. It does not recursively delete
the directory. Preserve failed-run logs until the failure is understood.

The softmax initial grid is C = 0.01, 0.1, 1, 10, 100, with L2 regularization,
L-BFGS, tolerance 1e-6 and 1,000 maximum iterations. Smaller C wins accuracy ties.
An initial-grid boundary winner triggers ONE additional candidate: 0.001 below
the lower boundary or 1,000 above the upper boundary. This is selected using inner
evidence only. The search does not continue expanding after that step.

Any convergence warning retries the same fit once at 5,000 iterations. A failed,
nonfinite or still-nonconverged candidate is logged and cannot win the search
unless it completed all required inner folds successfully. If the outer refit
fails, the run fails rather than silently choosing a model using outer scores.

The base run has 75 inner fits plus five outer refits. Each boundary expansion
adds three inner fits; convergence retries add logged attempts. No all-data model
is fitted, and no prediction submission or final family is selected.

## Add a decision-tree or random-forest adapter

Keep model logic in your own module, for example `models/decision_tree.py` and a
CLI in `experiments/run_decision_tree.py`. Use `ModelSpec` and `run_nested_cv` from
`experiments/common.py`. Do not copy the split or scoring implementation.

The adapter must supply:

```python
ModelSpec(
    name="decision_tree",  # or random_forest
    scale_numeric=False,
    initial_candidates=candidates,  # list of parameter dictionaries
    make_estimator=make_estimator,  # (parameters, seed) -> fresh estimator
    tie_key=tie_key,                # smaller tuple means simpler
    description={...},             # training and search settings for provenance
    source_files=("models/decision_tree.py", "experiments/run_decision_tree.py"),
)
```

The estimator must implement `fit`, `predict_proba`, `classes_` and `get_params`.
scikit-learn classifiers provide these. Construct a fresh estimator for every fit;
do not return a shared fitted instance. `make_estimator` receives seed 67.

Use the proposal's initial search spaces:

| Family | Initial settings |
|---|---|
| Decision tree | Gini; max_depth = 3, 5, 10, 20, None; min_samples_leaf = 1, 2, 5, 10 |
| Random forest | Gini and bootstrap; n_estimators = 100, 200; max_depth = 5, 10, 20, None; min_samples_leaf = 1, 2, 5; max_features = sqrt, 0.5 |

Create candidates using `itertools.product` or `sklearn.model_selection.ParameterGrid`.
State a deterministic tie key in advance, such as shallower depth then larger
minimum leaf size; forests can additionally prefer fewer trees. Map depth `None`
to a larger complexity value in the key. If max_features also ties, define its
ordering explicitly. If expanding ranges, supply `expand_candidates` and document
the stopping rule before evaluation. It receives ONLY the initial inner winner
and inner-search summaries; it must not inspect outer predictions.

Call:

```python
run_nested_cv(spec,
    source=ROOT / "csc311_project_training_202609.csv",
    protocol_path=ROOT / "development/protocol.json",
    output=ROOT / "results/decision_tree")
```

Then use the common diagnostics command with your results directory. The runner
can evaluate an adapter without export callbacks. To verify deployment later,
supply BOTH `export_model(estimator, state, parameters)` returning JSON-compatible
data and `predict_exported(raw_frame, saved_state, payload)` returning probabilities
in the fixed class order. The latter must use only permitted inference dependencies.
Missing export callbacks are recorded as `checked: false`, not as a passed check.

## Required output from every member

| Artifact | Purpose |
|---|---|
| `inner_search.csv` | One record per candidate/inner fold, including failures, training/validation metrics and runtime |
| `inner_search_summary.json` | Mean inner scores and search eligibility for each candidate |
| `fit_log.json` | Fit attempts, convergence status, warnings and failures |
| `outer_folds.json`, `outer_metrics.csv` | Selected settings, training and validation results for each outer fold |
| `oof_predictions.csv` | Exactly 870 rows in original source order, each predicted once by an outer model |
| `summary.json` | Aggregate and fold-level metrics, variability, fit counts and limitations |
| `run_manifest.json` | Dataset/protocol/code hashes, versions and execution status |
| `errors.csv`, `error_summary.json` | Common confusions and representative validation errors |
| `figures/` | Labeled tuning and evaluation figures |
| Model-specific report text | Training setup, ranges, results and weaknesses |

The prediction schema is `row_index` (zero based), `id`, `outer_fold`, `true_label`,
`predicted_label`, `probability_Baseball`, `probability_Basketball`, and
`probability_Soccer`. Preserve those names for direct comparison.

Fold exports in `fold_models/fold_0` through `fold_4` are reference models trained
on 232 respondents each. They are not a chosen final model or an inference ensemble.
The softmax NumPy implementation verifies all outer predictions against these
JSON exports. It does not implement the final `pred.py` entry point.

## Combine after all three models finish

Compare the three tuned families on the same outer folds, with accuracy primary
and macro-F1/class precision and variability as diagnostics. Do not describe any
family as the final winner until all three results exist. Once the family is
chosen, select its final hyperparameters using grouped development validation,
fit fresh preprocessing and that model on all 290 respondents, and export the
final assets. That full-data fit is separate from the reported outer-fold scores.

Complete the restricted-import prediction script and verify numerical parity,
row order, runtime and size. Integrate the report sections and single-number
performance estimate with the historical exposure limitation. Do not claim that
nested CV reverses preliminary feature-design exposure or eliminates all optimism.

## Method references

- scikit-learn 1.7 LogisticRegression: https://scikit-learn.org/1.7/modules/generated/sklearn.linear_model.LogisticRegression.html
- Nested validation rationale: https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html
