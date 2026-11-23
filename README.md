# CSC311 sport classification project

## Current development work

The submitted proposal is preserved in `proposal.pdf`. The historical `clean.py`,
`plot_distributions.py`, and `cleaned_data/` outputs are unchanged. They document
proposal exploration over training plus validation, not the new exploration scope.

The Member 1 handoff contains reusable preprocessing, respondent assignments,
an exposure audit, training-only exploration, tests, and a Data Exploration draft.
It does not train, tune, select, or evaluate a classifier, or implement `pred.py`.

## Reproduce and verify

Use Python 3.10 or later and install `requirements-development.txt` in a development
environment. The tested environment and input/script checksums are recorded in
`development/run_manifest.json`.

```sh
python prepare_member1.py
python -m unittest discover -s tests -v
python report/build_data_section.py
```

Run these commands from the repository directory. `prepare_member1.py` also accepts
`--source PATH`; existing assignments are protected against a changed dataset or
protocol. CSV checksums in the protocol normalize CRLF to LF so Windows and Unix
checkouts agree. Do not delete the frozen protocol to bypass a mismatch.

The report builder optionally supports `--pdf` with reportlab installed. Its
standalone review PDF is in `output/pdf/data_exploration_review.pdf`; it previews
the section, not the final report's typesetting. The LaTeX fragment is
`report/data_exploration.tex`. Compile the full report from the repository root
so the figure path resolves, and check the two-page section limit after integration.

## Exposure decision and evaluation protocol

Inspection during original raw-data dictionary design is unknown, as confirmed
on November 23, 2026. The earlier local training exploration and submitted
train-plus-validation exploration together cover 270 of 290 respondents. The
other 20 have unknown exposure; they are not a verified clean test group.

All 290 supplied respondents are therefore development data. There is **no
certified untouched local test set**. See `development/exposure_audit.json` for
exact IDs, counts, evidence checksums, and split discrepancies. Historical
manifests are preserved in `development/history/`. The audit reads only IDs from
historical cleaned split files.

The new protocol uses five respondent-grouped outer development folds, each
containing 58 respondents / 174 rows. Within each 232-respondent outer training
partition, three grouped inner folds are recorded for tuning. `random.Random(67)`
shuffles sorted integer IDs; positions are assigned modulo five. Inner seeds are
1067 through 1071. Every person's three responses stay together. These assignments
are in `development/protocol.json` and `development/respondent_folds.csv`.

New exploration uses only `outer_fold != 0`: 232 respondents, 696 rows and 232
examples per sport. Fold 0 is excluded from the new descriptive plots and fitted
exploration state. This is a forward exploration boundary, not a claim that those
58 respondents were historically untouched. No model score has been measured.

For Member 2, use nested grouped development evaluation: tune each family on
the three inner folds, refit its selected configuration on that outer training
partition, and predict its outer validation fold. Compare all three families on
the same assignments. Fit preprocessing independently in every training partition.
Do not choose hyperparameters using outer validation scores. After comparing the
families, report that selecting a winner and historical feature design can still
make the reported accuracy optimistic; the fixed manual dictionaries were not
developed independently of these respondents. Nested CV cannot repair that history.

The group has decided to proceed with this recorded development protocol without
asking course staff about test separation. The final report will briefly and
neutrally disclose that preliminary exploration informed the keyword definitions
before an untouched local test group could be verified, that preprocessing is
fitted separately within each training fold, and that earlier feature design may
make evaluation estimates optimistic. This decision does not change the recorded
respondent assignments or certify any group as historically untouched.

## Shared preprocessing interface

`preprocessing.py` imports only the standard library, NumPy and pandas. It has no
training or command-line side effects when imported.

```python
from preprocessing import read_survey, fit_preprocessor, transform_features
from development_protocol import load_protocol, outer_splits, inner_splits

df = read_survey("csc311_project_training_202609.csv", require_label=True)
protocol = load_protocol("development/protocol.json")

for outer, training_rows, validation_rows in outer_splits(df, protocol):
    # Tune on inner_splits(df, protocol, outer) first. Those positions also
    # refer to df, and outer validation rows appear in neither inner side.
    state = fit_preprocessor(df.iloc[training_rows])
    X_train = transform_features(df.iloc[training_rows], state, scale_numeric=True)
    X_validation = transform_features(df.iloc[validation_rows], state, scale_numeric=True)
    y_train = df.iloc[training_rows]["label"].to_numpy()
    y_validation = df.iloc[validation_rows]["label"].to_numpy()
    # Fit the chosen softmax configuration on X_train/y_train, then evaluate.
    # For decision trees and forests, use scale_numeric=False on BOTH sides.
```

`outer_splits` and `inner_splits` yield `(fold_number, training_positions,
validation_positions)` for `.iloc`, not DataFrame index labels. Fit fresh states
inside the inner loop too; the example above illustrates the outer refit only.
The complete feature and class ordering is in `development/feature_specification.json`.

For Member 3, after model selection, Member 2 fits a final deployment state on
the agreed full development training data and exports that state with the model:

```python
from preprocessing import load_state, read_survey, transform_features

state = load_state("preprocessing_state.json")
rows = read_survey(csv_filename, state=state)  # label and id are optional
X = transform_features(rows, state, scale_numeric=True).to_numpy()
# Apply exported softmax weights; trees/forests require scale_numeric=False.
```

The inference bundle needs `preprocessing.py`, the final preprocessing JSON, the
model assets, and `pred.py`. Saved state contains the option lists, regexes, header
aliases, medians, means, scales and feature order; inference does not need the
external `feature_definitions.json`. It must not fit a new state or depend on
`development/`, matplotlib, historical outputs or the training CSV.

**Never use `development/eda/eda_preprocessing_state.json` or its precomputed
feature matrices as input to cross-validation or as the final deployment state.**
They were fitted on the 696 exploration rows, including respondents that belong
to other outer validation folds.

## Feature definitions and handling rules

- 73 predictors: ease, strategy, injury, brilliance, 68 binary flags and log-payment.
  The returned matrix excludes ID and label, preserves row order/index, and uses a
  fixed saved column order. No respondent-level sport-order constraints are used.
- Original survey headers or documented short aliases are accepted in any order.
  Missing required question columns and duplicate mapped columns raise an error;
  missing individual answers are handled below. Optional extra columns are ignored.
- Unicode and whitespace are normalized; matching is case insensitive. Literal
  `N/A` and `None` are retained as answers, rather than automatically converted
  into missing values by pandas. Only the recorded regex rules decide their flags.
- Ease: nonnumeric/nonfinite values become missing, valid numbers are clipped to
  0-10, and missing values use the training median. All-missing training fallback: 5.
- Likert: strongly disagree through strongly agree map to 1-5. Missing/unknown
  entries use 3. No unknown nonblank Likert answers occur in the exploration subset.
- Multi-select: venue 7 columns, season 4, music 10. Known comma-separated options
  set independent flags; missing or unknown options set no known flag. No unknown
  nonblank options occur in the exploration subset.
- Historical fixed dictionaries: food 24 flags, colour 10, feelings 9, song 3 plus
  `song_named`. Their provenance is preserved in `feature_definitions.json` and
  their exposure limitation remains explicit. No new keywords were added.
- Payment: accept a single nonnegative amount (including comma thousands and k),
  an unambiguous ascending range, limited simple written numbers or an explicit
  refusal. Reject unrelated multiple amounts, negatives, nonfinite amounts,
  explicit foreign currencies and unparsed text; use the training median. No
  exchange rate is guessed. Bare dollar amounts are provisionally CAD. All-missing
  training fallback: 0. Transform with `log1p` and retain valid large amounts.
- Differences from proposal parsing: no averaging every digit in a narrative;
  whitespace/case and empty-song handling are consistent; literal `N/A` is preserved.
  The training payment audit records each changed interpretation. Conflicting
  refusal-plus-amount narratives are deliberately treated as ambiguous.
- Softmax only: standardize ease, the three Likert values and log-payment using
  training means and population SD (`ddof=0`). Constant columns use scale 1. Binary
  flags remain 0/1. Trees use the same base representation without scaling.

Regex indicators are imperfect: unmatched terms, false matches and language
variation remain limitations. A low song-category coverage is not a reason to
invent labels or inspect validation answers to expand the dictionary.

## Handoff artifacts and remaining work

| Location | Purpose |
|---|---|
| `development/exposure_audit.json` | Exposure history, exact IDs and historical split reconciliation |
| `development/protocol.json`, `respondent_folds.csv` | Frozen development and nested-fold assignments |
| `development/feature_specification.json` | Predictive feature/class order and scaling contract |
| `development/eda/summary.json` | Training-only findings and limitations |
| `development/eda/*.csv` | Raw-feature, numeric, categorical, text and payment audit evidence |
| `development/eda/figures/` | Reviewed training-only distribution and class-pattern plots |
| `report/data_exploration.tex` | Data Exploration draft for group review and integration |
| `tests/test_preprocessing.py` | Parser, saved-state, inference and respondent-isolation checks |

Member 2 still owns model training/tuning, metrics, error analysis and the final
performance estimate. Member 3 still owns `pred.py`, model inference parity,
runtime/size verification and full-report packaging. Every member should review
the section wording and contribute their own contributions/learning text.

Report the historical exposure limitation without claiming that a new split
removes it or guarantees full credit for the test-separation rubric item.
No commits, pushes, merges or submissions are part of this local handoff.
