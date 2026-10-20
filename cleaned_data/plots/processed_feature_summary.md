# Processed feature summary

This summary uses the combined training and validation data only (738 rows). The test set was not read or used.

## Feature types

| Feature(s) | Type | Description |
|---|---|---|
| id | Identifier | Respondent identifier; not a predictive feature |
| ease | Numeric | Ease score, clipped to 0-10 and median-imputed |
| strategy, injury, brilliance | Ordinal | Likert scores encoded from 1 to 5 |
| where_*, season_*, music_* | Binary multi-hot | One column per selected category |
| food_*, colour_*, feel_*, song_* | Binary keyword flags | One column per matched keyword or category |
| song_named | Binary | Whether a song was named |
| pay_log | Numeric | log(1 + parsed pay), with missing values median-imputed |
| label | Categorical target | Sport class to predict |

## Class balance

| Class | Count | Percentage |
|---|---:|---:|
| Baseball | 246 | 33.3% |
| Basketball | 246 | 33.3% |
| Soccer | 246 | 33.3% |

## Numeric distributions

| Feature | Count | Mean | Std. dev. | Median | Minimum | Maximum |
|---|---:|---:|---:|---:|---:|---:|
| ease | 738 | 5.46 | 2.31 | 5.00 | 0.00 | 10.00 |
| pay_log | 738 | 4.31 | 1.86 | 4.62 | 0.00 | 9.21 |

## Likert distributions

Counts for the encoded ordinal scores (1 = strongly disagree, 5 = strongly agree).

- **strategy** — 1: 8, 2: 72, 3: 153, 4: 358, 5: 147
- **injury** — 1: 15, 2: 117, 3: 192, 4: 292, 5: 122
- **brilliance** — 1: 35, 2: 143, 3: 174, 4: 265, 5: 121

## Binary feature activation

| Feature group | Columns | Rows with at least one active feature |
|---|---:|---:|
| where | 7 | 731 (99.1%) |
| season | 4 | 731 (99.1%) |
| music | 10 | 731 (99.1%) |
| food | 24 | 695 (94.2%) |
| colour | 10 | 712 (96.5%) |
| feel | 9 | 664 (90.0%) |
| song | 4 | 577 (78.2%) |
