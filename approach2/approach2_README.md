# Approach 2 — Sentence-Embedding KNN (Updated Dataset)

## Current status

**Current verified baseline: overall equal-weight QWK = 0.4431 on the updated dataset.**

The current dataset contains **19,225 rows and 91 questions**. The model uses sentence embeddings and K-nearest-neighbor (KNN) prediction, with different settings for large and small questions.

The updated model has been trained and checked across all five folds.

| Metric | Updated result |
|---|---:|
| Dataset rows | 19,225 |
| Number of questions | 91 |
| Large questions | 10 |
| Small questions | 81 |
| Large-question QWK | 0.6309 |
| Small-question QWK | 0.4196 |
| Overall equal-weight QWK | 0.4431 |
| Folds checked | 5 |
| Interface test | Passed |

The checker successfully generated predictions for all five validation folds. Each fold passed the interface checks, and the unseen-question fallback returned a valid grade.

These are local cross-validation results, **not an official leaderboard score**.

### Important: old results versus updated results

The previous dataset contained 17,207 rows. Its recorded production baseline was mean five-fold CV QWK = 0.5748, with an earlier experiment reporting higher scores under different settings.

The updated dataset contains 19,225 rows and 91 questions. The verified updated run reports an overall equal-weight QWK of 0.4431.

**Do not directly compare these scores as if they were measured on identical data.** The datasets, question coverage, and possibly the evaluation methods differ. Re-evaluate both models on the same dataset, folds, questions, and scoring method before claiming that one is better.

## 1. Files and what they do

All paths below are relative to the `approach2/` directory.

| File or folder | Purpose |
|---|---|
| `approach2.py` | Grader-facing inference code. Loads saved artifacts and predicts essay scores. |
| `train_approach2.py` | Trains and evaluates configurations using cross-validation. Saves the selected configuration and artifacts. |
| `check_approach2.py` | Checks the inference interface and calculates QWK on validation folds. |
| `weights/` | Older saved model artifacts. Preserve these until they are no longer needed. |
| `weights_updated/` | Artifacts generated using the updated dataset and current training code. This is the current verified updated configuration. |
| `weights_updated/approach2_config.json` | Selected encoders, large/small question settings, allowed grades, and fallback settings. |
| `weights_updated/approach2_knn.npz` | Saved model arrays, including embeddings, labels, question IDs, and fitted transformations. |
| `folds.csv` | Original fold assignments for the older dataset, if retained. |
| `folds_updated.csv` | Fold assignments matching the updated 19,225-row dataset. |
| `results_approach2.csv` | Historical results from earlier experiments. Do not assume these are results for the updated dataset. |
| `results_approach2_experiments.csv` | Historical experimental comparisons. These are not automatically integrated into the inference model. |
| `experiments_approach2.py` | Experimental configurations and comparisons. Use it to investigate alternatives, not as proof that an experiment is production-ready. |
| `local_assets/assets.json` | Local paths to downloaded sentence-embedding models. These paths may differ on another computer. |
| `cache/` | Cached embeddings, if present. |
| `local_data/` | Local-only data, if present. |

Other files in the directory, such as `approach2_README.md`, `folds.csv`, and the existing results files, should be reviewed before replacing or deleting anything.

## 2. Updated dataset and fold assignments

The updated CSV is stored outside the repository at:

```text
~/Downloads/training_dataset_updated.csv
```

The matching fold assignments are:

```text
approach2/folds_updated.csv
```

The updated dataset has 19,225 rows and 91 questions. The fold file was regenerated for this dataset and checked against its row IDs:

- Dataset rows: 19,225
- Fold rows: 19,225
- Missing IDs: 0
- Extra IDs: 0

**Always use `folds_updated.csv` with the updated dataset.** Do not substitute the old `folds.csv` unless you are deliberately reproducing the older experiment.

The dataset is not included in the repository. A teammate must obtain it through the permitted course/project sharing method and place it at the expected path, or supply a different path using `--csv`.

## 3. Current updated model configuration

The selected configuration saved under `weights_updated/approach2_config.json` is:

- **Encoders:** MiniLM and BGE-small, concatenated
- **Large-question threshold:** 200 graded answers
- **Large questions:** whitening, 64 PCA dimensions, alpha = 0.5, NCA metric with weight = 0.5 and 16 dimensions, K = 20, temperature = 0.1, weighted median aggregation
- **Small questions:** centering enabled, K = 5, temperature = 0.1, weighted mean aggregation
- **Unseen-question fallback:** K = 15, temperature = 0.1

The large-question settings were selected through whitening and NCA searches. The small-question settings were selected through a KNN configuration search.

The training output reported:

| Training metric | QWK |
|---|---:|
| Large questions | 0.6310 |
| Small questions | 0.4196 |
| All questions, equal weight | 0.4431 |
| Handout-style per-fold mean ± standard deviation | 0.4822 ± 0.0348 |

The checker subsequently reported 0.6309, 0.4196, and 0.4431, respectively. The small difference in the large-question score is due to displayed precision.

The handout-style per-fold statistic is a separate summary. Do not confuse it with the pooled out-of-fold, equal-weight score.

## 4. Reproduce the updated results

From the `approach2/` directory, activate the virtual environment and use the updated dataset and folds.

```bash
source .venv/bin/activate

python3 check_approach2.py \
  --csv ~/Downloads/training_dataset_updated.csv \
  --folds folds_updated.csv \
  --weights weights_updated \
  --fold all
```

The verified output reported:

```text
fold 0: interface OK (3850 predictions)
fold 1: interface OK (3849 predictions)
fold 2: interface OK (3847 predictions)
fold 3: interface OK (3844 predictions)
fold 4: interface OK (3835 predictions)

large questions: 0.6309
small questions: 0.4196
ALL (equal weight): 0.4431
```

The exact output may include additional diagnostic lines, depending on the current version of `check_approach2.py`.

This command checks the saved configuration and inference pipeline. It does not retrain the model.

## 5. Retrain the updated model

Only retrain when you intend to compare or change configurations.

The updated training command is:

```bash
python3 train_approach2.py \
  --csv ~/Downloads/training_dataset_updated.csv \
  --folds folds_updated.csv \
  --out-dir weights_updated
```

Training searches for settings for the large and small questions, calculates cross-validation results, and saves artifacts to `weights_updated/`.

**Caution:** rerunning this command with the same output directory may replace its existing artifacts. Before retraining, back up the current verified `weights_updated/` directory or use a new output directory, such as `weights_experiment_1/`.

After training, check the resulting configuration:

```bash
python3 check_approach2.py \
  --csv ~/Downloads/training_dataset_updated.csv \
  --folds folds_updated.csv \
  --weights weights_updated \
  --fold all
```

Keep the old and experimental artifacts separate until the new results have been evaluated.

## 6. Per-question QWK diagnostics

The checker can print the QWK for each question, helping identify where performance is weakest.

The current diagnostic results include:

| Question | QWK |
|---|---:|
| 83 | -0.1173 |
| 41 | -0.0827 |
| 71 | -0.0807 |
| 77 | -0.0784 |
| 82 | -0.0736 |
| 64 | -0.0644 |
| 3 | 0.2284 |
| 2 | 0.5898 |
| 8 | 0.5991 |

Question 3 is especially worth investigating because it has 1,891 examples but a relatively low QWK of 0.2284. Several small questions have negative QWK, but each has only about 24–25 examples, so those individual scores can be unstable.

Question 56 has constant true grades in the checked data and therefore returns NaN for QWK. Constant-grade questions are excluded from meaningful QWK averages by the current scoring implementation.

The per-question results are diagnostic findings, not proof that a particular model change will improve the overall score.

When investigating weak questions, examine their training examples, grade distributions, predictions, and nearest-neighbor matches before changing model settings.

## 7. Safe optimization plan

The main opportunity is improving the small questions, whose current average QWK is 0.4196. However, the large questions also contain at least one question with a lot of training data and relatively weak performance.

Recommended workflow:

1. **Preserve the baseline.** Keep a copy of `weights_updated/`, `folds_updated.csv`, and the current results.
2. **Use the same dataset and folds.** Do not change fold assignments between competing experiments.
3. **Investigate weak questions.** Start with Questions 83, 41, 71, 77, 82, 64, and Question 3.
4. **Change one factor at a time.** Compare encoders, centering, transformations, K, temperature, and aggregation.
5. **Prevent data leakage.** Fit centering, whitening, PCA, and supervised metric transformations on training folds only, then apply them to the corresponding validation fold.
6. **Compare using the same scoring method.** Track the overall equal-weight QWK, large- and small-question QWK, per-question results, and fold variation.
7. **Promote only verified improvements.** A higher search score alone does not guarantee better performance in the final inference pipeline.
8. **Run both checks after changing the model.** Verify cross-validation results and, if available, the project's official local grader interface.

The current training script already searches multiple settings for both question groups. Avoid repeating an identical search without a clear reason; use the diagnostics to identify a specific hypothesis first.

## 8. Production inference versus experiments

The grader-facing model consists of:

```text
approach2.py
weights_updated/approach2_config.json
weights_updated/approach2_knn.npz
```

The checker is used for local validation:

```text
check_approach2.py
```

Training and experimental files are used to generate or evaluate model configurations:

```text
train_approach2.py
experiments_approach2.py
```

Editing `train_approach2.py` does not automatically change the saved inference model. Likewise, recording a better result in an experiment CSV does not automatically promote that configuration into the production inference pipeline.

For a new experiment, save artifacts in a separate directory and pass that directory using `--weights` when running the checker. Confirm that the grader-facing code can load the artifacts before treating the experiment as submission-ready.

## 9. Local setup on a teammate's Mac

From the repository root:

```bash
git switch approach2
git pull origin approach2
cd approach2
```

Create and activate a Python environment if necessary:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy pandas scikit-learn sentence-transformers torch
```

If the project has a pinned requirements file, use it instead to keep dependencies consistent.

The teammate must obtain the updated training CSV and configure their own local model assets. The paths inside `local_assets/assets.json` may be specific to the original computer. Do not copy absolute paths unchanged.

The model assets are the Sentence Transformers models:

- `sentence-transformers/all-MiniLM-L6-v2`
- `BAAI/bge-small-en-v1.5`

The project uses local model paths when the assets file is supplied. Confirm that those paths exist on the teammate's computer before running the checker or training.

## 10. Git and file-safety workflow

Before changing anything:

```bash
git status --short
git log -1 --oneline
```

Keep the existing working configuration and weights safe. Before committing:

```bash
git diff --stat
git status --short
git diff --cached --stat
```

Do not commit private local assets, absolute-path configuration, virtual environments, embedding caches, or the training CSV unless the project explicitly permits it.

Before submitting, make sure the intended inference code and compatible artifacts are included and that the local grader test passes.

## Handoff summary

The current updated baseline uses the 19,225-row dataset and `folds_updated.csv`. Its saved, checker-verified artifacts are in `weights_updated/`, and its overall equal-weight QWK is **0.4431**.

The older `weights/` directory and historical results belong to the earlier workflow and should not be silently substituted for the updated configuration.

The next task is to investigate the weak questions and run controlled experiments using the same updated dataset, folds, and evaluation method. Preserve the verified baseline until a replacement has been tested successfully.
