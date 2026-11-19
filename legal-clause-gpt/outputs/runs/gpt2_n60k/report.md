# Evaluation: `gpt2_n60k`

gpt2, full fine-tuning on 59,080 clauses, max_len 384, trained in 50.7 min on Tesla T4.

## Headline (test split, 10,000 clauses, 100 classes)

| | accuracy | macro-F1 |
|---|---|---|
| validation | 87.0% | 79.3% |
| **test** | **87.3%** | **80.6%** |
| test, clauses never seen verbatim in training (9,718) | 87.0% | 80.3% |
| test, clause cut to fit max_len (500) | 88.8% | |
| test, clause fits uncut (9,500) | 87.2% | |

Macro-F1 weighs all 100 clause types equally, so it is pulled down by the rare ones; accuracy is dominated by the common ones.

## Most confused pairs (test)

| # | clause type A | clause type B | A predicted as B | B predicted as A | total |
|---|---|---|---|---|---|
| 1 | Tax Withholdings | Withholdings | 34 | 17 | 51 |
| 2 | Applicable Laws | Governing Laws | 41 | 0 | 41 |
| 3 | Defined Terms | Definitions | 20 | 15 | 35 |
| 4 | No Waivers | Waivers | 17 | 14 | 31 |
| 5 | Binding Effects | Successors | 12 | 17 | 29 |
| 6 | Authority | Authorizations | 10 | 15 | 25 |
| 7 | Representations | Warranties | 9 | 15 | 24 |
| 8 | Amendments | Modifications | 3 | 18 | 21 |
| 9 | Entire Agreements | Integration | 1 | 18 | 19 |
| 10 | Taxes | Withholdings | 16 | 2 | 18 |
| 11 | Indemnifications | Indemnity | 2 | 16 | 18 |
| 12 | Assignments | Successors | 5 | 12 | 17 |
| 13 | Survival | Warranties | 4 | 13 | 17 |
| 14 | Remedies | Specific Performance | 11 | 5 | 16 |
| 15 | Approvals | Consents | 6 | 8 | 14 |

These 15 pairs account for 29.6% of all 1,269 test errors.

## Abstention: "needs human review" below a confidence threshold

Thresholds are chosen on validation as the lowest one that reaches the target accuracy, then applied unchanged to test.

| target accuracy | threshold (from val) | val coverage | val accuracy | **test coverage** | **test accuracy** | test clauses sent to review |
|---|---|---|---|---|---|---|
| 90.0% | 0.556 | 94.3% | 90.0% | **94.9%** | **90.0%** | 507 |
| 95.0% | 0.881 | 80.1% | 95.0% | **79.8%** | **94.6%** | 2,024 |
| 98.0% | 0.983 | 60.9% | 98.0% | **59.4%** | **97.4%** | 4,063 |
| 99.0% | 0.995 | 50.0% | 99.0% | **48.4%** | **98.4%** | 5,156 |

Fixed thresholds, for reference (test):

| threshold | coverage | accuracy |
|---|---|---|
| 0.5 | 97.0% | 89.2% |
| 0.7 | 89.0% | 92.2% |
| 0.9 | 78.2% | 95.0% |

## Rare classes: per-class accuracy vs training frequency (test)

Review threshold here: 0.881 (the 95%-accuracy threshold above).

| training examples per class | classes | mean per-class accuracy | mean F1 | share sent to review |
|---|---|---|---|---|
| 0-49 | 3 | 53.3% | 53.3% | 38.3% |
| 100-249 | 14 | 59.7% | 61.8% | 52.6% |
| 250-499 | 44 | 80.2% | 81.7% | 25.2% |
| 500-999 | 24 | 87.0% | 85.9% | 24.2% |
| 1000+ | 15 | 93.6% | 92.3% | 11.5% |

Spearman rank correlation with training frequency: per-class accuracy +0.48, review rate -0.38.

Ten worst clause types (test):

| clause type | train examples | test clauses | accuracy | F1 | most often predicted as |
|---|---|---|---|---|---|
| Assigns | 31 | 4 | 0.0% | 0.0% | Binding Effects |
| Applicable Laws | 348 | 53 | 5.7% | 10.5% | Governing Laws |
| Indemnity | 267 | 29 | 37.9% | 52.4% | Indemnifications |
| Venues | 130 | 20 | 40.0% | 50.0% | Governing Laws |
| Powers | 124 | 13 | 46.2% | 46.2% | Organizations |
| Tax Withholdings | 377 | 75 | 46.7% | 51.1% | Withholdings |
| Enforceability | 242 | 45 | 46.7% | 56.8% | Severability |
| Jurisdictions | 199 | 29 | 48.3% | 47.5% | Governing Laws |
| Miscellaneous | 472 | 79 | 51.9% | 59.0% | Entire Agreements |
| Modifications | 253 | 55 | 52.7% | 63.7% | Amendments |

Classes with fewer than 10 test clauses have very noisy per-class accuracy (3 such classes).

## Calibration: are 90%-confident predictions right about 90% of the time? (test)

| confidence bin | clauses | mean confidence | accuracy | gap |
|---|---|---|---|---|
| 0.1-0.2 | 5 | 18.3% | 0.0% | -18.3 pts |
| 0.2-0.3 | 30 | 26.0% | 23.3% | -2.7 pts |
| 0.3-0.4 | 74 | 35.5% | 16.2% | -19.3 pts |
| 0.4-0.5 | 189 | 45.6% | 31.7% | -13.9 pts |
| 0.5-0.6 | 380 | 55.1% | 52.6% | -2.5 pts |
| 0.6-0.7 | 418 | 64.9% | 58.6% | -6.3 pts |
| 0.7-0.8 | 450 | 75.2% | 65.1% | -10.1 pts |
| 0.8-0.9 | 633 | 85.5% | 76.9% | -8.5 pts |
| 0.9-1.0 | 7,821 | 98.7% | 95.0% | -3.7 pts |

Mean confidence 92.0% vs accuracy 87.3%: on average the model is **overconfident by 4.7 points**. ECE = 4.7%.

Temperature scaling (one number T fitted on validation; logits / T): T = 1.38, test ECE 4.7% -> 1.2%, test NLL 0.513 -> 0.465. Accuracy and the ranking of confidences are unchanged, so the coverage curve is identical; only the numbers the threshold is expressed in move.
