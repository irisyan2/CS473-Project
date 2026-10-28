# CS 473 Project 1: Short-Answer Grade Prediction (Feynmanians)

https://github.com/irisyan2/CS473-Project/tree/approach1

Code and trained weights for our three approaches. Each approach predicts a grade for a student
answer from already-graded answers to the same question.

| Folder | Approach | Files |
| --- | --- | --- |
| `approach1/` | 1. KNN with TF-IDF | `approach1.py` (model), `train_approach1.py` (training), `cv_check.py` (5-fold check), `folds_updated.csv` |
| `approach2/` | 2. KNN with sentence embeddings (MiniLM + BGE) | `approach2.py` (model), `train_approach2.py` (tuning + training), `check_approach2.py` (fold check), `folds_updated.csv` |
| `approach3/` | 3. TF-IDF + Ridge regression | `approach3.py` (model), `train_approach3.py` (grid search + training) |
| `weights/` | trained weights for all three | see below |

## Weights

All weights are included (about 77 MB in total), so nothing needs to be downloaded or retrained.

| File | Approach | Size |
| --- | --- | --- |
| `weights/approach1.joblib` | 1 | 38 MB |
| `weights/approach2_config.json` | 2 (chosen settings) | 6 KB |
| `weights/approach2_knn.npz` | 2 (stored embeddings and grades) | 28 MB |
| `weights/approach3_bundle.joblib` | 3 | 7.5 MB |

All three were trained on `training_dataset_updated.csv` (91 questions, 19,225 graded answers).

## 1. Setup

```bash
conda create -n cs473 python=3.12 -y
conda activate cs473
pip install -r requirements.txt
```

Put the course data file `training_dataset_updated.csv` in this folder (next to this README).
It is not included in the zip.

## 2. Predict with the trained weights

Every approach has the same interface: `load_model(weights_dir, config, assets)` returns a
model, and `model.predict(rows, questions)` returns `[{"id": ..., "score": ...}, ...]`.
Run from this folder, replacing `N` with 1, 2 or 3:

```bash
python -c "
import sys; N = 1
sys.path.insert(0, f'approach{N}')
model = __import__(f'approach{N}').load_model('weights', {}, None)
rows = [{'id': 1, 'question': '1', 'text': 'Add a control group and repeat the trials.'}]
print(model.predict(rows, {'1': {'allowed_grades': [0, 1, 2, 3]}}))
"
```

Approach 2 downloads its two sentence-transformer models (all-MiniLM-L6-v2 and
bge-small-en-v1.5) from Hugging Face the first time it loads, so it needs internet once.

Approach 3 prints an `InconsistentVersionWarning` when it loads: its saved weights were written
with scikit-learn 1.8, and they load and predict correctly with 1.6.1. Retraining it (section 4)
writes a new file without the warning.

## 3. Check the scores (5-fold cross-validation)

```bash
# Approach 1: about 1 minute, prints ALL questions 0.484
cd approach1
python cv_check.py ../training_dataset_updated.csv
cd ..

# Approach 2: rebuilds the model on 4 folds and predicts the 5th, for all 5 folds
cd approach2
python check_approach2.py --csv ../training_dataset_updated.csv --folds folds_updated.csv --weights ../weights --fold all
cd ..
```

Approach 3's check is part of its training script (next section).

## 4. Retrain the weights (optional)

Each command overwrites that approach's files in `weights/`.

```bash
# Approach 1: about 10 seconds
cd approach1
python train_approach1.py ../training_dataset_updated.csv --out ../weights/approach1.joblib
cd ..

# Approach 2: downloads MiniLM + BGE, then tunes on the 5 folds; 20-40 minutes on a laptop CPU
cd approach2
python train_approach2.py --csv ../training_dataset_updated.csv --folds folds_updated.csv --out-dir ../weights
cd ..

# Approach 3: grid search over 3,240 settings, then fits the final model; about 5 minutes
cd approach3
python train_approach3.py --csv ../training_dataset_updated.csv --out ../weights
cd ..
```

Approach 3's training prints the settings it chose: `a_big = 1.0, a_small = 0.1, a_g = 1.0,
m = 10, s = 1.9, b = -0.05` (validation QWK 0.489 on its own internal splits). It also writes
`weights/folds_approach3.csv`, the splits it used.

## Expected results

Cross-validated QWK on the 5 shared folds (`folds_updated.csv`), computed per question and
averaged across questions. Question 56 is left out because all its answers have the same grade.

| Approach | Original Qs (1-10) | New Qs (11-91) | All |
| --- | --- | --- | --- |
| 1. KNN with TF-IDF | 0.545 | 0.476 | 0.484 |
| 2. KNN with embeddings | 0.640 | 0.481 | **0.498** |
| 3. TF-IDF + Ridge | 0.660 | 0.404 | 0.433 |
