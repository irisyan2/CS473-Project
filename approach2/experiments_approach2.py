"""Local-only. Tries ways to improve cosine-KNN-on-embeddings, using cached embeddings
and the same folds.csv as train_approach2.py (fair comparison). Transforms are fit on the
TRAIN part of each fold only, per essay set.

  python experiments_approach2.py --csv "path/to/training dataset.csv"
Writes results_approach2_experiments.csv
"""
import argparse, itertools
from pathlib import Path
import numpy as np, pandas as pd
import approach2 as A
import train_approach2 as T

KS = [3, 5, 7, 10, 15, 20, 30]
TEMPS = [0.02, 0.05, 0.1]
AGGS = ["mean", "vote"]
TRANSFORMS = [("raw", 0, 0), ("center", 0, 0),
              ("white", 64, 0.5), ("white", 128, 0.5), ("white", 64, 1.0), ("white", 128, 1.0)]


def l2(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def fit_tf(Etr, kind, n, alpha):
    if kind == "raw":
        return lambda X: X
    mu = Etr.mean(0)
    if kind == "center":
        return lambda X: l2(X - mu)
    _, S, Vt = np.linalg.svd(Etr - mu, full_matrices=False)
    s = S[:n] / np.sqrt(len(Etr) - 1)
    W = Vt[:n].T * (s + 1e-6) ** (-alpha)
    return lambda X: l2((X - mu) @ W)


def run(df, E, fold, classes, kind, n, alpha):
    y = df["Score1"].to_numpy(); es = df["EssaySet"].to_numpy(); kmax = max(KS)
    cfgs = list(itertools.product(KS, TEMPS, AGGS))
    pred = {c: np.zeros(len(df), dtype=int) for c in cfgs}
    for f in range(T.N_FOLDS):
        for q in np.unique(es):
            tr = np.where((es == q) & (fold != f))[0]; va = np.where((es == q) & (fold == f))[0]
            tf = fit_tf(E[tr], kind, n, alpha)
            sims = tf(E[va]) @ tf(E[tr]).T
            idx = np.argsort(-sims, axis=1)[:, :kmax]
            s = np.take_along_axis(sims, idx, axis=1); lab = y[tr][idx]
            for (k, t, ag) in cfgs:
                sk = s[:, :k]
                w = np.exp((sk - sk.max(1, keepdims=True)) / t)
                pred[(k, t, ag)][va] = A.aggregate(lab[:, :k], w, classes[q], ag)
    rows = []
    for c in cfgs:
        pf = [np.mean([T.qwk(y[(es == q) & (fold == f)], pred[c][(es == q) & (fold == f)], classes[q])
                       for q in np.unique(es)]) for f in range(T.N_FOLDS)]
        rows.append(dict(k=c[0], temp=c[1], agg=c[2], mean_qwk=np.mean(pf), std_qwk=np.std(pf)))
    return pd.DataFrame(rows).sort_values("mean_qwk", ascending=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--csv", default="training_dataset.csv")
    ap.add_argument("--cache", default="cache"); args = ap.parse_args()
    df = pd.read_csv(args.csv); fold = T.make_folds(df)
    classes = {q: sorted(df.loc[df.EssaySet == q, "Score1"].unique().tolist()) for q in df.EssaySet.unique()}
    embs = {p.stem[4:]: np.load(p) for p in sorted(Path(args.cache).glob("emb_*.npy"))}
    print("Found cached encoders:", list(embs))
    if len(embs) >= 2:  # concatenation of all available encoders
        embs["concat_all"] = l2(np.hstack([embs[k] for k in sorted(embs)]))
    out = []
    for name, E in embs.items():
        for kind, n, alpha in TRANSFORMS:
            r = run(df, E, fold, classes, kind, n, alpha).iloc[0]
            row = dict(encoder=name, transform=kind, pca_dims=n, alpha=alpha, **r.to_dict())
            out.append(row); print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    res = pd.DataFrame(out).sort_values("mean_qwk", ascending=False)
    res.to_csv("results_approach2_experiments.csv", index=False)
    print("\nTOP 8:\n", res.head(8).to_string(index=False))