# Experiments (CS 473 Part 1 report)

Code behind every number and figure in the report. Run everything **from the repository root**
(the folder that contains `approach1/`, `approach2/` and `submission/`).

## Setup

1. Put the updated training data in the repository root as `training_dataset_updated.csv`
   (or pass another path with `--csv`). It is not committed to the repo.
2. Install the libraries (same versions as the grader):
   ```bash
   pip install numpy==2.2.6 scikit-learn==1.6.1 scipy pandas joblib nltk matplotlib
   ```
   No sentence-transformers or model downloads are needed: Approach 2 runs on the MiniLM + BGE
   embeddings already saved in `approach2/weights_updated/approach2_knn.npz`.

## What each script does

| Script | Report section | Runtime (laptop) | Output in `experiments/results/` |
| --- | --- | --- | --- |
| `exp1_compare.py` | 3.1 Experiment 1: all three approaches on the shared folds | ~10 min | `exp1_summary.csv`, `exp1_per_question.csv` |
| `a1_ablation.py` | 3.1 Tuning: Approach 1 tuning path and ablations | ~10 min | `a1_tuning_path.csv`, `a1_ablations.csv` |
| `exp2_exp3.py --method random` | 3.2 Experiment 2: QWK vs. training-set size | ~50 min (10 splits) | `random.npz` |
| `exp2_exp3.py --method cluster --n N` | 3.3 Experiment 3: clustering | ~10 min per N (2 splits) | `cluster_N<N>.npz` |
| `make_figures.py` | figures for 3.2 and 3.3 | seconds | `exp2_results.csv`, `exp3_heatmap.csv`, `fig_exp2.png`, `fig_exp3.png` |

`common.py` holds the shared data loading, folds and metric. `approach3_old/` is the TF-IDF + Ridge
Approach 3 used in the submission (its training code is no longer in `approach3/`).

## Commands

```bash
python experiments/exp1_compare.py --csv training_dataset_updated.csv
python experiments/a1_ablation.py  --csv training_dataset_updated.csv

# Experiment 2 (all 10 splits)
python experiments/exp2_exp3.py --csv training_dataset_updated.csv --method random --splits 0-9

# Experiment 3 (heatmap: N = 2, 4, 8, 16 on splits 0-1)
for N in 2 4 8 16; do
  python experiments/exp2_exp3.py --csv training_dataset_updated.csv --method cluster --n $N \
      --splits 0-1 --sizes 0.1,0.2,0.3,0.5,0.7,0.9
done

python experiments/make_figures.py --csv training_dataset_updated.csv
```

`results/` already contains the `.npz` files from our runs, so `make_figures.py` works right away
without re-running the long experiments.

## Method notes

- **Folds:** `approach1/folds_updated.csv`, shared by all approaches. Inside each question, answers
  are sorted by grade and dealt round-robin into 5 folds, so every fold has the same grade mix.
- **Metric:** QWK per question on the pooled out-of-fold predictions, averaged across questions
  (as the leaderboard averages). `lb_min`/`lb_max` score each fold on its own, like a small test
  set, with undefined QWK counted as 0. Question 56 (all answers 5.0) is excluded.
- **Experiment 2:** ten 90/10 splits (T/P) stratified by grade; from T, 10%-90% of each question's
  answers are drawn with a grade-stratified sample; Approach 2 is retrained and predicts P.
- **Experiment 3:** K-Means++ per question on the Approach 2 embeddings; each cluster gets a share
  of the picks proportional to its size (at least one), filled with the answers nearest its centre.

## Expected results

| | Original Qs | New Qs | All |
| --- | --- | --- | --- |
| Approach 1 | 0.545 | 0.476 | 0.484 |
| Approach 2 | 0.640 | 0.481 | 0.498 |
| Approach 3 (TF-IDF + Ridge) | 0.660 | 0.404 | 0.433 |
| Experiment 2, 10% -> 90% of T (Approach 2) | 0.503 -> 0.639 | 0.269 -> 0.487 | 0.294 -> 0.504 |
