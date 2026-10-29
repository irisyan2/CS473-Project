"""Shared helpers for the CS 473 Part 1 experiments.

Every script is run from the REPOSITORY ROOT, e.g.
    python experiments/exp1_compare.py --csv training_dataset_updated.csv

Metric (same as the leaderboard): QWK per question, then the plain mean across questions.
  * pooled  : each question's out-of-fold predictions are pooled before computing its QWK
              (a 25-answer question has only ~5 answers per fold - too few for a stable QWK)
  * lb-style: QWK per question on ONE fold's answers (like a small leaderboard test set);
              undefined QWK counts as 0. Reported as the range over the 5 folds.
Question 56 (every answer graded 5.0) has an undefined pooled QWK and is left out.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[1]          # repository root
EXP = ROOT / "experiments"
OUT = EXP / "results"
OUT.mkdir(exist_ok=True)
FOLDS_CSV = ROOT / "approach1" / "folds_updated.csv"  # shared 5-fold split (all approaches)
A2_NPZ = ROOT / "approach2" / "weights_updated" / "approach2_knn.npz"
A2_CFG = ROOT / "approach2" / "weights_updated" / "approach2_config.json"


def add_path(p):
    p = str(p)
    if p not in sys.path:
        sys.path.insert(0, p)


def load_data(csv):
    df = pd.read_csv(csv)
    df["Score1"] = df["Score1"].astype(float)
    return df


def allowed_grades(scores):
    """0-5 in half points for the new questions, 0..max whole points for the original ones."""
    s = np.asarray(scores, float)
    if (s % 1 != 0).any() or s.max() > 3:
        return [i / 2 for i in range(11)]
    return [float(g) for g in range(int(s.max()) + 1)]


def question_info(df):
    allowed = {q: allowed_grades(d.Score1) for q, d in df.groupby("EssaySet")}
    questions = {str(q): {"allowed_grades": a} for q, a in allowed.items()}
    return allowed, questions


def shared_folds(df):
    f = pd.read_csv(FOLDS_CSV)[["Id", "fold"]]
    fold = df[["Id"]].merge(f, on="Id", how="left")["fold"]
    assert fold.notna().all(), "folds_updated.csv does not cover every Id"
    return fold.to_numpy().astype(int)


def qwk(y_true, y_pred, allowed):
    lab = {float(g): i for i, g in enumerate(sorted(allowed))}
    a = [lab[float(v)] for v in y_true]
    b = [lab[float(v)] for v in y_pred]
    if len(set(a)) == 1 and len(set(b)) == 1:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cohen_kappa_score(a, b, labels=list(range(len(lab))), weights="quadratic")


def score(df, pred, fold=None, rows=None):
    """Returns dict(original, new, all[, lb_min, lb_max]) and the per-question pooled QWK."""
    allowed, _ = question_info(df)
    Q, Y = df.EssaySet.to_numpy(), df.Score1.to_numpy()
    rows = np.ones(len(df), bool) if rows is None else rows
    per = {}
    for q in np.unique(Q):
        m = rows & (Q == q) & ~np.isnan(pred)
        per[q] = qwk(Y[m], pred[m], allowed[q])
    per = pd.Series(per, name="qwk")
    out = dict(original=per[per.index <= 10].mean(), new=per[per.index > 10].mean(), all=per.mean())
    if fold is not None:
        lb = []
        for f in np.unique(fold):
            ks = [qwk(Y[(Q == q) & (fold == f)], pred[(Q == q) & (fold == f)], allowed[q]) for q in np.unique(Q)]
            lb.append(np.mean([0.0 if np.isnan(k) else k for k in ks]))
        out.update(lb_min=min(lb), lb_max=max(lb))
    return out, per


def stored_embeddings(df):
    """Approach 2's saved MiniLM+BGE vectors (rows in dataset order) -> no model download needed.
    Returns the concatenated vectors and a text -> row lookup."""
    z = np.load(A2_NPZ)
    assert (z["qid"] == df.EssaySet.astype(str).to_numpy()).all(), \
        "approach2_knn.npz does not match this CSV's row order"
    E = z["emb"].astype(np.float32)
    look = {}
    for i, t in enumerate(df.EssayText.astype(str).to_numpy()):
        look.setdefault(t, i)
    return E, look


def use_stored_embeddings(approach2_module, E, look):
    """Make approach2.py read stored vectors instead of loading sentence-transformers."""
    approach2_module.get_encoder = lambda name, assets=None: name
    approach2_module.embed = lambda encs, names, texts: approach2_module.l2(E[[look[str(t)] for t in texts]])


def fmt(d):
    return "  ".join(f"{k} {v:.3f}" for k, v in d.items())
