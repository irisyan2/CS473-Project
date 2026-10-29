"""Experiments 2 and 3 with Approach 2 (the best approach in Experiment 1).

Experiment 2 - QWK vs. training-set size (random, grade-stratified selection):
    python experiments/exp2_exp3.py --csv training_dataset_updated.csv --method random --splits 0-9

Experiment 3 - clustered selection with N clusters per question (K-Means++ on the embeddings):
    python experiments/exp2_exp3.py --csv training_dataset_updated.csv --method cluster --n 8 --splits 0-1 \
        --sizes 0.1,0.2,0.3,0.5,0.7,0.9
  (the report's heatmap runs --n 2, 4, 8 and 16 on splits 0-1 with those six sizes)

Design:
  * Within each question, answers are sorted by grade and dealt into 10 parts. Each part in turn
    is the prediction set P (10%); the other nine parts are the training set T (90%).
  * For each size p, each question keeps round(p * |T_q|) graded answers (at least 1):
      random : a grade-stratified systematic sample (covers every grade level)
      cluster: K-Means++ into N clusters (N <= number of picks); each cluster gets a share
               proportional to its size (>= 1), filled with the answers nearest its centre.
  * Approach 2 is retrained on the selected answers with its saved settings
    (approach2/weights_updated/approach2_config.json) and predicts P.
  * Predictions are pooled over the splits; QWK per question, averaged (see common.py).
Each run saves experiments/results/<method>[_N<n>].npz; make_figures.py turns them into the plots.
Runtime: about 5 minutes per split for --method random (9 sizes), a little more for cluster.
"""
import argparse
import json
import tempfile
import warnings

import numpy as np
from sklearn.cluster import KMeans

from common import OUT, ROOT, A2_CFG, add_path, load_data, question_info, stored_embeddings, use_stored_embeddings

warnings.filterwarnings("ignore")
SIZES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def split_parts(df, seed=0, parts=10):
    rng = np.random.RandomState(seed)
    Q, Y = df.EssaySet.to_numpy(), df.Score1.to_numpy()
    part = np.empty(len(df), int)
    for q in np.unique(Q):
        idx = rng.permutation(np.where(Q == q)[0])
        idx = idx[np.argsort(Y[idx], kind="stable")]
        part[idx] = (np.arange(len(idx)) + rng.randint(parts)) % parts
    return part


def pick_random(idx, m, Y, rng):
    idx = rng.permutation(idx)
    idx = idx[np.argsort(Y[idx], kind="stable")]
    pos = np.unique(np.linspace(0, len(idx) - 1, m).round().astype(int))
    return idx[pos]


def pick_cluster(idx, m, N, E, rng):
    N = max(1, min(N, m, len(idx)))
    km = KMeans(n_clusters=N, init="k-means++", n_init=3, random_state=rng.randint(1 << 30)).fit(E[idx])
    d = np.linalg.norm(E[idx] - km.cluster_centers_[km.labels_], axis=1)
    queues = [list(idx[km.labels_ == c][np.argsort(d[km.labels_ == c])]) for c in range(N)]
    sizes = np.array([len(qu) for qu in queues], float)
    quota = np.maximum(1, np.floor(m * sizes / sizes.sum())).astype(int)
    while quota.sum() > m:
        quota[np.argmax(quota)] -= 1
    while quota.sum() < m:
        quota[np.argmax(sizes - quota)] += 1
    return np.array([i for c in range(N) for i in queues[c][:quota[c]]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "training_dataset_updated.csv"))
    ap.add_argument("--method", choices=["random", "cluster"], default="random")
    ap.add_argument("--n", type=int, default=8, help="clusters per question (cluster method)")
    ap.add_argument("--splits", default="0-9", help="which of the 10 splits to run, e.g. 0-9 or 0-1")
    ap.add_argument("--sizes", default=",".join(map(str, SIZES)))
    args = ap.parse_args()

    add_path(ROOT / "approach2")
    import approach2 as A
    import train_approach2 as T
    df = load_data(args.csv)
    _, questions = question_info(df)
    E, look = stored_embeddings(df)
    use_stored_embeddings(A, E, look)
    cfg = json.load(open(A2_CFG))
    dfa = df.copy()
    dfa["EssaySet"] = dfa["EssaySet"].astype(int)
    Q, Y = df.EssaySet.to_numpy(), df.Score1.to_numpy()
    part = split_parts(df)
    sizes = [float(s) for s in args.sizes.split(",")]
    a, b = (int(x) for x in args.splits.split("-"))
    splits = list(range(a, b + 1))

    preds = {p: np.full(len(df), np.nan) for p in sizes}
    ntrain = {p: [] for p in sizes}
    for s in splits:
        P = np.where(part == s)[0]
        Tmask = part != s
        for p in sizes:
            rng = np.random.RandomState(1000 * s + int(round(p * 100)))
            sel = []
            for q in np.unique(Q):
                tq = np.where(Tmask & (Q == q))[0]
                m = max(1, int(round(p * len(tq))))
                sel.append(pick_random(tq, m, Y, rng) if args.method == "random" else pick_cluster(tq, m, args.n, E, rng))
            tr = np.concatenate(sel)
            ntrain[p].append(len(tr))
            with tempfile.TemporaryDirectory() as d:
                T.fit_artifacts(dfa.iloc[tr].reset_index(drop=True), E[tr], cfg, d)
                rows = [{"id": int(df.Id.iat[i]), "question": str(Q[i]), "text": str(df.EssayText.iat[i])} for i in P]
                preds[p][P] = [o["score"] for o in A.load_model(d, {}, None).predict(rows, questions)]
        print(f"{args.method} {'N=' + str(args.n) if args.method == 'cluster' else ''} split {s} done", flush=True)

    name = "random" if args.method == "random" else f"cluster_N{args.n}"
    np.savez(OUT / f"{name}.npz", sizes=np.array(sizes), preds=np.stack([preds[p] for p in sizes]),
             ntrain=np.array([np.mean(ntrain[p]) for p in sizes]), splits=np.array(splits))
    print(f"saved experiments/results/{name}.npz")


if __name__ == "__main__":
    main()
