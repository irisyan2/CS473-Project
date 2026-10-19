# Approach 2 — Sentence-Embedding KNN

## Current status

**Current production baseline: mean 5-fold CV QWK = 0.5748.**

The current selected configuration is:

- **Encoders:** `minilm` + `bge_small`, concatenated
- **Centering:** per-question centering enabled
- **K:** `20`
- **Temperature:** `0.1`
- **Aggregation:** weighted vote
- **CV standard deviation:** `0.014381`
- **Fold-0 check:** mean QWK `0.5474`
- **Local grader interface test:** passed; returned all `17,207` predictions

The centering ablation showed a substantial improvement:

| Encoder configuration | Without centering | With centering |
|---|---:|---:|
| MiniLM | 0.5072 | 0.5501 |
| BGE-small | 0.5056 | 0.5724 |
| MiniLM + BGE-small | 0.5187 | 0.5748 |

These are cross-validation results, not an official leaderboard score.

### Experimental result — not yet the production model

`experiments_approach2.py` tested additional transformations. Its best recorded result was:

- Concatenated embeddings
- PCA whitening, 64 dimensions, `alpha=0.5`
- `K=30`, temperature `0.1`, weighted vote
- Mean CV QWK: `0.5760`, standard deviation `0.0132`

This result is only about `0.0012` above the current baseline and has **not** been integrated into and verified by the production inference pipeline. Treat it as an experiment, not the current submission configuration.

## What each file does

| File | Purpose |
|---|---|
| `approach2.py` | Grader-facing inference interface; loads saved artifacts and predicts scores |
| `train_approach2.py` | Runs cross-validation/configuration search and saves the selected model artifacts |
| `check_approach2.py` | Checks the model interface and evaluates a held-out fold |
| `weights/approach2_config.json` | Selected inference configuration and allowed score classes |
| `weights/approach2_knn.npz` | Saved embedding vectors, training labels, question IDs, and centering information |
| `experiments_approach2.py` | Additional experimental comparisons, including centering and whitening |
| `results_approach2_experiments.csv` | Recorded experiment results |
| `folds.csv` | Reusable fold assignments, if present locally |
| `cache/` | Cached encoder embeddings, if present locally |

The model weights and the grader-facing `approach2.py` are the production inference files. The experiment script and its CSV are for analysis; they do not automatically change the production model.

## Files and data needed on a new device

Clone the repository and check out the `approach2` branch. The repository should contain this folder and the saved `weights/` files.

The training CSV is a separate local file and is **not included in the repository**. Obtain it through the project/course's permitted sharing method and place it at:

`~/Downloads/training dataset.csv`

The local grader files are also separate from this folder. If available to your teammate, place them under `~/Downloads/local_environment/`, including `local_test.py` and `questions.json`.

### Model assets

`local_assets/assets.json` and the model snapshots are local-only assets and may not be committed. The paths in one person's `assets.json` will not work on another computer.

On a new device, the teammate must either:
1. download/cache the required Sentence Transformers models (`sentence-transformers/all-MiniLM-L6-v2` and `BAAI/bge-small-en-v1.5`) and configure that device's local assets file according to the project's local testing setup; or
2. use the code's Hugging Face model-loading path where supported, with internet access on the first run.

Do not copy absolute paths from another person's `assets.json` unchanged. Keep credentials, private paths, virtual environments, local assets, caches, and training data out of Git.

## Setup on macOS

From the repository root:

```bash
cd ~/Desktop/CS473/CS473-Project
git switch approach2
git pull origin approach2
cd approach2
```

Use Python 3.12 if possible and create a fresh environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy pandas scikit-learn sentence-transformers torch
```

If the project provides a pinned requirements file, prefer installing from that file instead so everyone's environment is reproducible. If a command or dependency differs from the course setup, check the relevant script's `--help` output before changing its arguments.

## Reproduce the current baseline

Ensure the training CSV is at `~/Downloads/training dataset.csv` and that this device's model assets are configured. Then run from the `approach2/` directory:

```bash
python train_approach2.py --csv "$HOME/Downloads/training dataset.csv"
python check_approach2.py --csv "$HOME/Downloads/training dataset.csv"
```

Training may take time the first time because it may need to generate encoder embeddings. Cached embeddings can make later experiments faster. The training script saves updated files under `weights/`; this can overwrite the current production artifacts, so preserve a copy or commit the baseline before experimenting.

For a fair comparison, reuse the existing `folds.csv` when available. Do not silently regenerate different folds between experiments.

## Run the local grader test

The grader test folder is a separate folder at the repository root. If it exists, run:

```bash
cd ~/Desktop/CS473/CS473-Project/approach2_test

python ~/Downloads/local_environment/local_test.py \
  --submission-dir "$PWD" \
  --csv "$HOME/Downloads/training dataset.csv" \
  --questions ~/Downloads/local_environment/questions.json \
  --assets ../approach2/local_assets/assets.json
```

A successful run should report the grader entry point, then `OK: 17207 predictions`. The `--assets` path above assumes the teammate has configured a valid local assets file at that location; update it for their own machine. A successful interface test confirms that predictions run and have the expected structure; it does not measure leaderboard QWK.

## Optimizing the model safely

1. **Keep the baseline.** Record the current CV mean QWK (`0.5748`), standard deviation (`0.014381`), fold assignments, config, and weights before changes.
2. **Change one thing at a time.** Compare encoder choices, transformations, K, temperature, and aggregation using the same folds.
3. **Avoid leakage.** Fit centering, PCA/whitening, or other learned transformations on each training fold only; apply those fitted transformations to its validation fold.
4. **Compare per-question scores.** An average can hide a question set that performs poorly.
5. **Do not promote an experiment just because its score is slightly higher.** The whitening result (`0.5760`) is very close to the baseline (`0.5748`) relative to fold variation. Verify it with the production inference path and grader before replacing the saved model.
6. **Re-run both checks** after changing inference or artifacts:
   - `python check_approach2.py --csv "$HOME/Downloads/training dataset.csv"`
   - the local grader test above.
7. **Keep experiment outputs distinct** from production artifacts so the final submission is easy to identify.

## Git workflow

Before optimizing:

```bash
cd ~/Desktop/CS473/CS473-Project
git status --short
git log -1 --oneline
```

Create a separate branch for experiments if desired:

```bash
git switch -c approach2-optimization
```

Commit only intended files. Do not add `.venv/`, local model assets, private configuration, cached embeddings, or the training dataset. Before pushing, inspect:

```bash
git diff --stat
git status --short
git diff --cached --stat
```

When taking over from someone else, first pull the latest `approach2` branch and confirm the production config and weights before starting experiments.
