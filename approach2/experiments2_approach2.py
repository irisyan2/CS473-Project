"""Local-only. Tests whether per-question hyperparameters and/or a TF-IDF hybrid similarity
beat the whitened-KNN baseline (CV QWK 0.5760). Uses cached embeddings and folds.csv, and
fits whitening + TF-IDF on the TRAIN part of each fold only, per essay set.

  python experiments2_approach2.py --csv "$HOME/Downloads/training dataset.csv"

Similarity = (1 - lam) * whitened-embedding cosine + lam * TF-IDF cosine  (lam = 0 -> baseline).

HONEST NUMBERS: picking the best of ~170 configs on the same CV folds inflates the score.
So besides the optimistic "best config" number, this reports NESTED scores: for each
held-out fold, the config (global, or per question) is chosen using only the OTHER folds'
out-of-fold predictions, then scored on the held-out fold. Trust the nested numbers.

Writes results_approach2_hybrid.csv and perq_config_approach2.json.
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import approach2 as A
import train_approach2 as T

ENC = ["minilm", "bge_small"]
PCA, ALPHA = 64, 0.5
KS = [10, 20, 30, 50]
TEMPS = [0.05, 0.1, 0.2]
AGGS = ["vote", "mean"]
LAMBDAS = [0.2, 0.4, 0.6]
KINDS = {"word": dict(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, min_df=1),
         "char": dict(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2)}
SIMS = [("emb", 0.0)] + [(k, l) for k in KINDS for l in LAMBDAS]
BASE = (("emb", 0.0), 30, 0.1, "vote")  # current whitened production config


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--csv", default="training_dataset.csv")
    args = ap.parse_args()
    df = pd.read_csv(args.csv); fold = T.make_folds(df)
    y = df["Score1"].to_numpy(); es = df["EssaySet"].to_numpy()
    texts = df["EssayText"].fillna("").astype(str).to_numpy()
    classes = {q: sorted(df.loc[df.EssaySet == q, "Score1"].unique().tolist()) for q in df.EssaySet.unique()}
    E = T.get_combined(df, ENC)
    qlist = sorted(np.unique(es)); nq, nf = len(qlist), T.N_FOLDS
    cfgs = [(s, k, t, a) for s in SIMS for k in KS for t in TEMPS for a in AGGS]
    cidx = {c: i for i, c in enumerate(cfgs)}; nc = len(cfgs); kmax = max(KS)
    pred = np.zeros((nc, len(df)), dtype=int)

    for f in range(nf):
        for q in qlist:
            tr = np.where((es == q) & (fold != f))[0]; va = np.where((es == q) & (fold == f))[0]
            mu, W = A.whiten_fit(E[tr], PCA, ALPHA)
            proj = lambda X: A.l2((X - mu) @ W)
            S_emb = proj(E[va]) @ proj(E[tr]).T
            S_tf = {}
            for kind, kw in KINDS.items():
                v = TfidfVectorizer(**kw); Xtr = v.fit_transform(texts[tr])
                S_tf[kind] = (v.transform(texts[va]) @ Xtr.T).toarray()
            for sim in SIMS:
                kind, lam = sim
                S = S_emb if kind == "emb" else (1 - lam) * S_emb + lam * S_tf[kind]
                idx = np.argsort(-S, axis=1)[:, :kmax]
                s = np.take_along_axis(S, idx, axis=1); lab = y[tr][idx]
                for k in KS:
                    for t in TEMPS:
                        w = A.weights_for(s[:, :k], t)
                        for a in AGGS:
                            pred[cidx[(sim, k, t, a)], va] = A.aggregate(lab[:, :k], w, classes[q], a)
        print("fold", f, "done", flush=True)

    # QWK tensors: Qf = on the held-out fold only; Qo = on all OTHER folds (for nested selection)
    Qf = np.zeros((nc, nq, nf)); Qo = np.zeros((nc, nq, nf))
    for ci in range(nc):
        for qi, q in enumerate(qlist):
            m = es == q
            for f in range(nf):
                a, b = m & (fold == f), m & (fold != f)
                Qf[ci, qi, f] = T.qwk(y[a], pred[ci][a], classes[q])
                Qo[ci, qi, f] = T.qwk(y[b], pred[ci][b], classes[q])

    def report(name, per_fold):
        print("%-34s mean %.4f  std %.4f" % (name, np.mean(per_fold), np.std(per_fold)))
        return np.mean(per_fold)

    emb_only = np.array([c[0][0] == "emb" for c in cfgs])
    b = cidx[BASE]
    print("\n=== Results (mean over folds of mean-over-questions QWK) ===")
    report("baseline whitened (K30,t.1,vote)", Qf[b].mean(0))
    g = Qf.mean(axis=(1, 2)); gi = int(np.argmax(g))
    report("best single config (OPTIMISTIC)", Qf[gi].mean(0)); print("   ->", cfgs[gi])
    # nested selections
    glob = [Qf[int(np.argmax(Qo[:, :, f].mean(1))), :, f].mean() for f in range(nf)]
    report("NESTED global config", glob)
    for label, mask in [("NESTED per-question (emb only)", emb_only), ("NESTED per-question (+tfidf)", None)]:
        sc = []
        for f in range(nf):
            vals = []
            for qi in range(nq):
                o = Qo[:, qi, f].copy()
                if mask is not None:
                    o[~mask] = -9
                vals.append(Qf[int(np.argmax(o)), qi, f])
            sc.append(np.mean(vals))
        report(label, sc)

    # per-question picks using all folds (config to productionize if the nested score wins)
    print("\n=== Per-question best config (all folds; optimistic per-question score) ===")
    perq = {}
    for qi, q in enumerate(qlist):
        o = Qf[:, qi, :].mean(1); ci = int(np.argmax(o)); (kind, lam), k, t, a = cfgs[ci]
        print("set %-3s base %.3f -> %.3f  sim=%s lam=%.1f K=%d temp=%.2f agg=%s"
              % (q, Qf[b, qi].mean(), o[ci], kind, lam, k, t, a))
        perq[str(q)] = dict(sim=kind, lam=lam, k=k, temp=t, agg=a)
    json.dump(perq, open("perq_config_approach2.json", "w"), indent=1)
    rows = [dict(sim=c[0][0], lam=c[0][1], k=c[1], temp=c[2], agg=c[3],
                 mean_qwk=Qf[i].mean(), std_qwk=Qf[i].mean(0).std()) for i, c in enumerate(cfgs)]
    pd.DataFrame(rows).sort_values("mean_qwk", ascending=False).to_csv("results_approach2_hybrid.csv", index=False)
    print("\nSaved results_approach2_hybrid.csv and perq_config_approach2.json")


if __name__ == "__main__":
    main()