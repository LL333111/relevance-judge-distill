# Report

## 1. Setup

**Data.** Amazon Shopping Queries (ESCI), US, Task 1. Each query has up to 40 candidate products labeled Exact, Substitute, Complement or Irrelevant, with nDCG gains 1.0, 0.1, 0.01 and 0. Splits are by query and frozen in `data/splits.json`.

| Split | Queries | Pairs | Use |
|---|---|---|---|
| H5k | 251 | 5,005 | The human-label budget |
| V | 1,000 | 19,913 | Model selection and gates |
| U | 17,390 | 349,734 | Unlabeled pool; its human labels are never loaded |
| T | 3,000 | 61,055 | Final evaluation, scored once per row |

**Student.** `Qwen/Qwen3-Reranker-0.6B`, scored by P("yes"). Fine-tuning uses LoRA (r = 16, all projection layers) and Margin-MSE over all candidate pairs within a query.

**Judge.** `Qwen/Qwen3-14B` with LoRA, trained with cross-entropy to emit one label token. Its score for a pair is the expected gain under the four label probabilities.

**Metric.** nDCG@10 with the ESCI gains; ties are broken by pair identifier. Differences are tested with a paired bootstrap over queries (10,000 resamples).

## 2. Results

### Main table (test set)

| Row | Configuration | nDCG@10 | Pairs/s | ms/query |
|---|---|---|---|---|
| 1 | Student, zero-shot | 0.7658 | 252 | 81 |
| 2 | Student + H5k human labels (baseline) | 0.7832 | 126 | 162 |
| 3 | Student + H5k human labels + U judge scores (method) | 0.7998 | 127 | 160 |
| 4 | Judge, fine-tuned on H5k | 0.7989 | 12 | 1,658 |

| Comparison | Difference | 95% CI |
|---|---|---|
| Row 3 − row 2 | +0.0166 | [+0.0125, +0.0208] |
| Row 2 − row 1 | +0.0174 | [+0.0130, +0.0219] |
| Row 4 − row 3 | −0.0009 | [−0.0059, +0.0041] |

- The judge's labels roughly double what the 5,000 human labels buy: +0.034 over zero-shot instead of +0.017.
- The student is statistically indistinguishable from its teacher while scoring 10× as many pairs per second.
- Rows 2 and 3 were measured with unmerged LoRA adapters. Merged, they have the same compute as row 1 (252 pairs/s, about 20× the judge); that figure was not measured for the tuned models.

### Model selection (validation set)

| Learning rate | Row 2 (baseline) | Row 3 (method) |
|---|---|---|
| 5e-5 | **0.7821** | **0.7972** |
| 1e-4 | 0.7815 | 0.7970 |
| 2e-4 | 0.7770 | 0.7960 |
| 4e-4 | 0.6789 | 0.7897 |

Row 2 ran 8 epochs with validation after each; it peaked at epoch 1 or 2 and then overfit. Row 3 ran one epoch over 354,739 pairs with four validations. Every row 3 configuration beats every row 2 configuration.

### Judge quality and gates

| Judge | Kappa zero-shot → tuned | Accuracy zero-shot → tuned | V nDCG@10 | Tuned judge − row 2 on V [95% CI] | Gate |
|---|---|---|---|---|---|
| Qwen3-8B | 0.236 → 0.351 | 0.48 → 0.58 | 0.7906 | +0.0085 [−0.0015, +0.0183] | Failed |
| Qwen3-14B | 0.262 → 0.340 | 0.52 → 0.58 | 0.7968 | +0.0147 [+0.0053, +0.0242] | Passed |

The gate was set before any run: a judge is distilled only if it beats the baseline on validation with a CI that excludes zero. The 8B judge did not, so it was replaced once by the 14B model, as the plan allowed. The 14B judge was best after its first epoch and declined afterwards.

Four-way accuracy of 0.58 is far below the 91% agreement between human annotators, yet the judge ranks better than the baseline. Ranking only needs the expected gain to order candidates correctly, not the hard label to be right.

## 3. Error analysis

**Where the judge disagrees with humans.** On T the judge's top label differs from the human label for 41% of pairs. Recall by class is E 0.76, S 0.46, C 0.29, I 0.48. The main confusion is Substitute read as Exact: 45% of human-Substitute pairs.

**Manual review of 20 sampled disagreements** (`results/judge_disagreements_sample.json`, seed 0):

| Category | Count | Example |
|---|---|---|
| Judge ignores a specific attribute in the query (model, brand, "without", "maternity") | 7 | "dog bowl stand without bowls" → stand with two bowls, judged Exact |
| Human label looks wrong; the judge's label is defensible | 5 | "lcp 380 laser sight" → laser for a different pistol, human label Exact |
| Substitute/Irrelevant boundary | 4 | "court of thorns and roses series" → a different fantasy series |
| Ambiguous or misspelled query | 4 | "e12 bolt" → E12 bulbs |

In 16 of the 20 the judge is more lenient than the human. The clearest weakness to fix is attribute and negation handling.

**Top-10 composition.** The share of top-10 positions held by Exact products rises from 0.534 (row 1) to 0.545 (row 2) to 0.554 (row 3), equal to the judge's 0.554; Irrelevant products fall from 0.117 to 0.107.

## 4. Deviations from the plan

- **Judge model.** Qwen3-8B failed the gate; Qwen3-14B was used (the plan's single permitted fallback).
- **50K-label ablation not run.** Its partial baseline runs were lost with a deleted GPU instance and the remaining budget went to the main table.
- **Baseline adapter retrained.** The selected baseline's weights were lost with the same instance. The configuration was rerun unchanged (validation 0.7829 vs. 0.7821 originally, same best epoch); the test score in row 2 comes from the rerun.
- **Pool labeled twice.** The first set of judge labels was stranded on stopped instances. The pool was relabeled with the same adapter; only the second set is used.
- **Order of test evaluation.** Rows 1, 2 and 4 were scored on T while row 3 was still training. Row 3 was selected on V alone by a fixed rule (highest validation nDCG@10).
- **Hardware.** Runs used A40, RTX PRO 6000, L40S and RTX 4090 cards as availability changed. All throughput figures come from one GPU type (L40S).

## 5. Limitations

1. One draw of the 5,000 labels and one seed; the spread across draws is unknown.
2. The baseline ignores the unlabeled pool. Self-training (the student pseudo-labels the pool itself) needs no judge and was not compared.
3. For both arms the best learning rate is the lowest of the four fixed in advance, so lower rates might do better. This applies equally to both.
4. Reranking of already-retrieved candidates, US English only.
5. Label noise (91% human agreement) caps every model, and ESCI's public availability since 2022 allows pretraining exposure.
6. The judge costs 7.9 GPU-hours on L40S to label 350K pairs (12 pairs/s); that cost is paid once, offline.
