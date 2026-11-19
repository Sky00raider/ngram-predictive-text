# Evaluation: `distilgpt2_n60k`

distilgpt2, full fine-tuning on 59,080 clauses, max_len 384, trained in 33.5 min on Tesla T4.

## Headline (test split, 10,000 clauses, 100 classes)

| | accuracy | macro-F1 |
|---|---|---|
| validation | 85.4% | 77.1% |
| **test** | **85.5%** | **77.3%** |
| test, clauses never seen verbatim in training (9,718) | 85.3% | 77.1% |
| test, clause cut to fit max_len (500) | 86.8% | |
| test, clause fits uncut (9,500) | 85.4% | |

Macro-F1 weighs all 100 clause types equally, so it is pulled down by the rare ones; accuracy is dominated by the common ones.

## Most confused pairs (test)

| # | clause type A | clause type B | A predicted as B | B predicted as A | total |
|---|---|---|---|---|---|
| 1 | Tax Withholdings | Withholdings | 30 | 17 | 47 |
| 2 | Applicable Laws | Governing Laws | 43 | 0 | 43 |
| 3 | Defined Terms | Definitions | 21 | 19 | 40 |
| 4 | No Waivers | Waivers | 10 | 24 | 34 |
| 5 | Entire Agreements | Integration | 1 | 28 | 29 |
| 6 | Representations | Warranties | 4 | 24 | 28 |
| 7 | Authority | Authorizations | 19 | 8 | 27 |
| 8 | Amendments | Modifications | 4 | 21 | 25 |
| 9 | General | Payments | 24 | 0 | 24 |
| 10 | Binding Effects | Successors | 4 | 20 | 24 |
| 11 | Consent To Jurisdiction | Jurisdictions | 20 | 1 | 21 |
| 12 | Indemnifications | Indemnity | 11 | 9 | 20 |
| 13 | Remedies | Specific Performance | 17 | 2 | 19 |
| 14 | Assignments | Successors | 4 | 14 | 18 |
| 15 | Construction | Interpretations | 16 | 1 | 17 |

These 15 pairs account for 28.7% of all 1,449 test errors.

## Abstention: "needs human review" below a confidence threshold

Thresholds are chosen on validation as the lowest one that reaches the target accuracy, then applied unchanged to test.

| target accuracy | threshold (from val) | val coverage | val accuracy | **test coverage** | **test accuracy** | test clauses sent to review |
|---|---|---|---|---|---|---|
| 90.0% | 0.607 | 90.2% | 90.0% | **90.1%** | **90.1%** | 994 |
| 95.0% | 0.894 | 74.2% | 95.0% | **74.2%** | **94.8%** | 2,579 |
| 98.0% | 0.984 | 51.9% | 98.0% | **50.0%** | **98.0%** | 4,995 |
| 99.0% | 0.994 | 41.1% | 99.0% | **40.0%** | **98.8%** | 6,002 |

Fixed thresholds, for reference (test):

| threshold | coverage | accuracy |
|---|---|---|
| 0.5 | 95.2% | 88.0% |
| 0.7 | 86.0% | 91.6% |
| 0.9 | 73.5% | 94.9% |

## Rare classes: per-class accuracy vs training frequency (test)

Review threshold here: 0.894 (the 95%-accuracy threshold above).

| training examples per class | classes | mean per-class accuracy | mean F1 | share sent to review |
|---|---|---|---|---|
| 0-49 | 3 | 46.7% | 40.0% | 30.0% |
| 100-249 | 14 | 48.0% | 51.5% | 65.0% |
| 250-499 | 44 | 79.3% | 79.8% | 30.3% |
| 500-999 | 24 | 84.7% | 84.2% | 34.2% |
| 1000+ | 15 | 92.6% | 90.7% | 14.5% |

Spearman rank correlation with training frequency: per-class accuracy +0.48, review rate -0.37.

Ten worst clause types (test):

| clause type | train examples | test clauses | accuracy | F1 | most often predicted as |
|---|---|---|---|---|---|
| Assigns | 31 | 4 | 0.0% | 0.0% | Binding Effects |
| Applicable Laws | 348 | 53 | 1.9% | 3.7% | Governing Laws |
| Venues | 130 | 20 | 15.0% | 24.0% | Governing Laws |
| Consent To Jurisdiction | 173 | 33 | 27.3% | 37.5% | Jurisdictions |
| Effectiveness | 227 | 31 | 29.0% | 41.9% | Effective Dates |
| Miscellaneous | 472 | 79 | 34.2% | 44.6% | Entire Agreements |
| Submission To Jurisdiction | 232 | 29 | 34.5% | 48.8% | Jurisdictions |
| Qualifications | 47 | 5 | 40.0% | 40.0% | Organizations |
| Positions | 190 | 32 | 40.6% | 50.0% | Duties |
| General | 953 | 203 | 41.4% | 53.0% | Payments |

Classes with fewer than 10 test clauses have very noisy per-class accuracy (3 such classes).

## Calibration: are 90%-confident predictions right about 90% of the time? (test)

| confidence bin | clauses | mean confidence | accuracy | gap |
|---|---|---|---|---|
| 0.1-0.2 | 19 | 16.2% | 5.3% | -11.0 pts |
| 0.2-0.3 | 65 | 26.3% | 20.0% | -6.3 pts |
| 0.3-0.4 | 146 | 35.7% | 31.5% | -4.2 pts |
| 0.4-0.5 | 249 | 45.8% | 45.0% | -0.8 pts |
| 0.5-0.6 | 487 | 55.0% | 49.5% | -5.5 pts |
| 0.6-0.7 | 435 | 64.8% | 60.5% | -4.4 pts |
| 0.7-0.8 | 478 | 75.3% | 66.5% | -8.8 pts |
| 0.8-0.9 | 775 | 85.5% | 75.2% | -10.3 pts |
| 0.9-1.0 | 7,346 | 98.4% | 94.9% | -3.4 pts |

Mean confidence 89.8% vs accuracy 85.5%: on average the model is **overconfident by 4.3 points**. ECE = 4.3%.

Temperature scaling (one number T fitted on validation; logits / T): T = 1.23, test ECE 4.3% -> 1.3%, test NLL 0.550 -> 0.525. Accuracy and the ranking of confidences are unchanged, so the coverage curve is identical; only the numbers the threshold is expressed in move.
