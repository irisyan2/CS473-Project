"""Check Approach 1's score with 5-fold cross-validation, using the leaderboard metric
(QWK per question, then the plain average across questions).

Usage (from demo_submission, next to approach1.py and train_approach1.py):
    python cv_check.py training_dataset_updated.csv

It also writes:
    folds_updated.csv     - the 5-fold split (Id, EssaySet, fold); give this to your teammates
                            so all three approaches use the same folds
    cv_per_question.csv   - QWK for every question (for the report's success/error analysis)
"""
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from approach1 import load_model
from train_approach1 import build


def make_folds(df, seed=0, n_splits=5):
    """Per question: shuffle, sort by grade, deal rows round-robin into folds.
    Every question (even the 24-25-answer ones) gets ~equal rows per fold and the same grade mix."""
    rng = np.random.RandomState(seed)
    fold = np.empty(len(df), int)
    for q, idx in df.groupby("EssaySet").indices.items():
        idx = rng.permutation(idx)
        idx = idx[np.argsort(df["Score1"].values[idx], kind="stable")]
        fold[idx] = (np.arange(len(idx)) + rng.randint(n_splits)) % n_splits
    return fold


def allowed_grades(scores):
    """Grade scale: 0-5 in half-points for the new questions, 0..max for the original ones."""
    s = np.asarray(scores, float)
    if (s % 1 != 0).any() or s.max() > 3:
        return [i / 2 for i in range(11)]
    return [float(g) for g in range(int(s.max()) + 1)]


def qwk(y_true, y_pred, allowed):
    lab = {float(g): i for i, g in enumerate(sorted(allowed))}
    a = [lab[float(v)] for v in y_true]
    b = [lab[float(v)] for v in y_pred]
    if len(set(a)) == 1 and len(set(b)) == 1:
        return np.nan  # undefined when only one grade appears; left out of the average
    return cohen_kappa_score(a, b, labels=list(range(len(lab))), weights="quadratic")


def main(path):
    df = pd.read_csv(path)
    folds_file = Path("folds_updated.csv")
    if folds_file.exists():
        df = df.merge(pd.read_csv(folds_file)[["Id", "fold"]], on="Id")
    else:
        df["fold"] = make_folds(df)
        df[["Id", "EssaySet", "fold"]].to_csv(folds_file, index=False)
        print("wrote folds_updated.csv")

    scales = {q: allowed_grades(d["Score1"]) for q, d in df.groupby("EssaySet")}
    questions = {str(q): {"allowed_grades": a} for q, a in scales.items()}
    df["pred"] = np.nan

    for f in range(5):
        tr, te = df[df.fold != f], df[df.fold == f]
        with tempfile.TemporaryDirectory() as tmp:
            joblib.dump(build(tr), Path(tmp) / "approach1.joblib")
            model = load_model(tmp, {}, {})
            rows = [{"id": r.Id, "question": str(r.EssaySet), "text": r.EssayText}
                    for r in te.itertuples()]
            preds = model.predict(rows, questions)
        df.loc[te.index, "pred"] = [p["score"] for p in preds]
        print(f"fold {f + 1}/5 done", flush=True)

    # QWK per question on all out-of-fold predictions (a 24-25-answer question only has
    # ~5 answers per fold, too few for a stable per-fold QWK), then averaged across questions.
    per_q = pd.Series({q: qwk(d["Score1"], d["pred"], scales[q])
                       for q, d in df.groupby("EssaySet")}, name="qwk")
    per_q.to_csv("cv_per_question.csv", index_label="EssaySet")
    big, small = per_q[per_q.index <= 10], per_q[per_q.index > 10]

    print("\nApproach 1 cross-validated QWK (per question, averaged like the leaderboard)")
    print(f"  original questions 1-10 : {big.mean():.3f}")
    print(f"  new questions 11-91     : {small.mean():.3f}"
          + (f"  ({small.isna().sum()} skipped: only one grade)" if small.isna().any() else ""))
    print(f"  ALL questions           : {per_q.mean():.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "training_dataset_updated.csv")