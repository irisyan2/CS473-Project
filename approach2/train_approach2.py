"""Local-only: cross-validate Approach 2 and write the artifacts to weights/.

  python train_approach2.py --csv training_dataset.csv [--assets local_assets/assets.json]
Default (no flags): searches PCA-whitened minilm+bge_small embeddings blended with word/char
TF-IDF similarity (weights, K, temp, vote/mean/median) by 5-fold CV, prints the optimistic best
and a NESTED estimate, and saves the best model to weights/. (Previous best: CV QWK ~0.5869.)
  --fixed   skip the search, build one config from --tfidf/--lam/--k/--temp
  --search  the original centering-only search (~0.5748)
Verify with check_approach2.py --folds all.
Outputs: folds.csv (reuse for approaches 1 & 3), cache/emb_<enc>.npy,
         results_approach2.csv, weights/approach2_knn.npz + approach2_config.json
"""
import argparse, itertools, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
import approach2 as A

ENCODER_SETS = [["minilm"], ["bge_small"], ["minilm", "bge_small"]]  # >=2 encoders compared
KS = [5, 10, 20, 30, 50, 80]
TEMPS = [0.05, 0.1, 0.2, 1e6]          # 1e6 == uniform weights
AGGS = ["vote", "mean"]
CENTERS = [False, True]                # ablation: per-question centering off / on
N_FOLDS, SEED = 5, 42


def make_folds(df, path="folds.csv"):
    if Path(path).exists():
        f = pd.read_csv(path)
        return df[["Id"]].merge(f, on="Id", how="left")["fold"].to_numpy()
    fold = np.zeros(len(df), dtype=int)
    strat = df["EssaySet"].astype(str) + "_" + df["Score1"].astype(str)
    for i, (_, va) in enumerate(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(df, strat)):
        fold[va] = i
    pd.DataFrame({"Id": df["Id"], "fold": fold}).to_csv(path, index=False)
    return fold


def get_embeddings(df, enc, assets=None):
    p = Path("cache") / f"emb_{enc}.npy"
    p.parent.mkdir(exist_ok=True)
    if p.exists():
        return np.load(p)
    E = A.encode_texts(A.get_encoder(enc, assets), df["EssayText"].tolist(), enc)
    np.save(p, E)
    return E


def get_combined(df, names, assets=None):
    return A.l2(np.hstack([get_embeddings(df, n, assets) for n in names]))


def qwk(y, p, allowed):
    if len(set(y)) < 2 and len(set(p)) < 2:
        return 1.0 if list(y)[:1] == list(p)[:1] else 0.0
    return cohen_kappa_score(y, p, labels=allowed, weights="quadratic")


def cross_validate(df, E, fold, classes):
    configs = list(itertools.product(CENTERS, KS, TEMPS, AGGS))
    pred = {c: np.zeros(len(df), dtype=int) for c in configs}
    y = df["Score1"].to_numpy(); es = df["EssaySet"].to_numpy(); kmax = max(KS)
    for f in range(N_FOLDS):
        for q in np.unique(es):
            tr = np.where((es == q) & (fold != f))[0]; va = np.where((es == q) & (fold == f))[0]
            mu = E[tr].mean(0)  # fit on the training part only
            for center in CENTERS:
                Etr, Eva = (A.l2(E[tr] - mu), A.l2(E[va] - mu)) if center else (E[tr], E[va])
                sims = Eva @ Etr.T
                idx = np.argsort(-sims, axis=1)[:, :kmax]
                s = np.take_along_axis(sims, idx, axis=1); lab = y[tr][idx]
                for (c0, k, t, ag) in configs:
                    if c0 == center:
                        pred[(c0, k, t, ag)][va] = A.aggregate(lab[:, :k], A.weights_for(s[:, :k], t),
                                                               classes[q], ag)
    rows = []
    for c in configs:
        per_fold = [np.mean([qwk(y[(es == q) & (fold == f)], pred[c][(es == q) & (fold == f)], classes[q])
                             for q in np.unique(es)]) for f in range(N_FOLDS)]
        rows.append(dict(center=c[0], k=c[1], temp=c[2], agg=c[3],
                         mean_qwk=np.mean(per_fold), std_qwk=np.std(per_fold)))
    return pd.DataFrame(rows)


# ---- hybrid search: whitened embeddings + word/char TF-IDF similarity -------------------
HYB_ENC = ["minilm", "bge_small"]
PCA_DIMS, PCA_ALPHA = 64, 0.5
HYB_KS = [20, 30, 50]
HYB_TEMPS = [0.05, 0.1, 0.2]
HYB_AGGS = ["vote", "mean", "median"]
HYB_WEIGHTS = []  # (w_emb, w_word, w_char), sums to 1
for _we in (0.2, 0.3, 0.4):
    _r = 1 - _we
    HYB_WEIGHTS += [(_we, _r, 0.0), (_we, _r / 2, _r / 2), (_we, 0.0, _r)]
HYB_WEIGHTS = [tuple(round(x, 4) for x in w) for w in HYB_WEIGHTS]


def search_hybrid(df, E, fold, classes):
    """CV over similarity = w_emb*whitened-emb + w_word*word-TFIDF + w_char*char-TFIDF, K, temp, agg.
    Whitening and TF-IDF are fit on the train part of each fold only (per essay set).
    Returns (results DataFrame sorted by mean_qwk, nested (mean, std)). The nested score picks the
    config on the OTHER folds only, so it is a less optimistic estimate than the best row."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    y = df["Score1"].to_numpy(); es = df["EssaySet"].to_numpy()
    texts = df["EssayText"].fillna("").astype(str).to_numpy()
    qlist = sorted(np.unique(es)); nq, nf = len(qlist), N_FOLDS
    cfgs = [(w, k, t, a) for w in HYB_WEIGHTS for k in HYB_KS for t in HYB_TEMPS for a in HYB_AGGS]
    cidx = {c: i for i, c in enumerate(cfgs)}; nc = len(cfgs); kmax = max(HYB_KS)
    pred = np.zeros((nc, len(df)), dtype=int)
    for f in range(nf):
        for q in qlist:
            tr = np.where((es == q) & (fold != f))[0]; va = np.where((es == q) & (fold == f))[0]
            mu, W = A.whiten_fit(E[tr], PCA_DIMS, PCA_ALPHA)
            proj = lambda X: A.l2((X - mu) @ W)
            S_e = proj(E[va]) @ proj(E[tr]).T
            S = {}
            for kind in ("word", "char"):
                v = TfidfVectorizer(**A.TFIDF_KW[kind]); Xtr = v.fit_transform(texts[tr])
                S[kind] = (v.transform(texts[va]) @ Xtr.T).toarray()
            for w in HYB_WEIGHTS:
                Sim = w[0] * S_e + w[1] * S["word"] + w[2] * S["char"]
                idx = np.argsort(-Sim, axis=1)[:, :kmax]
                s = np.take_along_axis(Sim, idx, axis=1); lab = y[tr][idx]
                for k in HYB_KS:
                    for t in HYB_TEMPS:
                        wt = A.weights_for(s[:, :k], t)
                        for a in HYB_AGGS:
                            pred[cidx[(w, k, t, a)], va] = A.aggregate(lab[:, :k], wt, classes[q], a)
        print("  hybrid search: fold", f, "done", flush=True)
    Qf = np.zeros((nc, nq, nf)); Qo = np.zeros((nc, nq, nf))
    for ci in range(nc):
        for qi, q in enumerate(qlist):
            m = es == q
            for f in range(nf):
                a, b = m & (fold == f), m & (fold != f)
                Qf[ci, qi, f] = qwk(y[a], pred[ci][a], classes[q])
                Qo[ci, qi, f] = qwk(y[b], pred[ci][b], classes[q])
    nested = [Qf[int(np.argmax(Qo[:, :, f].mean(1))), :, f].mean() for f in range(nf)]
    rows = [dict(w_emb=c[0][0], w_word=c[0][1], w_char=c[0][2], k=c[1], temp=c[2], agg=c[3],
                 mean_qwk=Qf[i].mean(), std_qwk=Qf[i].mean(0).std()) for i, c in enumerate(cfgs)]
    return pd.DataFrame(rows).sort_values("mean_qwk", ascending=False), (np.mean(nested), np.std(nested))


def fit_artifacts(df, E, cfg, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    qs = df["EssaySet"].astype(str).to_numpy().astype("U16")
    uq = np.unique(qs)
    extra = {}
    if cfg.get("transform") == "white":  # per-question PCA whitening (+ pooled fallback)
        n, a = cfg["pca_dims"], cfg["alpha"]
        fits = [A.whiten_fit(E[qs == q], n, a) for q in uq]
        mu = np.stack([f[0] for f in fits]).astype(np.float32)
        extra["W"] = np.stack([f[1] for f in fits]).astype(np.float32)
        mu_pool, W_pool = A.whiten_fit(E, n, a)
        extra["W_pool"] = W_pool
    else:
        zero = np.zeros(E.shape[1])
        mu = np.stack([E[qs == q].mean(0) if cfg["center"] else zero for q in uq]).astype(np.float32)
        mu_pool = (E.mean(0) if cfg["center"] else zero).astype(np.float32)
    if cfg.get("tfidf"):  # TF-IDF is refit at load time from the stored training texts
        texts = df["EssayText"].fillna("").astype(str).tolist()
        extra["texts_json"] = np.frombuffer(json.dumps(texts).encode("utf-8"), dtype=np.uint8)
    np.savez_compressed(out / "approach2_knn.npz", emb=E.astype(np.float16),
                        y=df["Score1"].to_numpy().astype(np.int16), qid=qs,
                        mu_qid=uq, mu=mu, mu_pool=mu_pool, **extra)
    (out / "approach2_config.json").write_text(json.dumps(cfg))


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--csv", default="training_dataset.csv")
    ap.add_argument("--assets", default="local_assets/assets.json" if Path("local_assets/assets.json").exists() else None)
    ap.add_argument("--search", action="store_true",
                    help="run the ORIGINAL centering-only config search (CV ~0.5748)")
    ap.add_argument("--fixed", action="store_true",
                    help="skip all searching; build one fixed config from --tfidf/--lam/--k/--temp")
    ap.add_argument("--whiten", action="store_true", help="(ignored; kept for compatibility)")
    ap.add_argument("--out-dir", default="weights", help="where to save artifacts")
    ap.add_argument("--tfidf", choices=["word", "char", "none"], default="word",
                    help="--fixed only: TF-IDF kind ('none' = embeddings only)")
    ap.add_argument("--lam", type=float, default=0.6, help="--fixed only: TF-IDF weight (0..1)")
    ap.add_argument("--k", type=int, default=30); ap.add_argument("--temp", type=float, default=0.1)
    args = ap.parse_args(argv)
    assets = json.load(open(args.assets)) if args.assets else None
    print("Using local assets:", args.assets)
    df = pd.read_csv(args.csv)
    fold = make_folds(df)
    classes = {q: sorted(df.loc[df.EssaySet == q, "Score1"].unique().tolist()) for q in df.EssaySet.unique()}
    if args.fixed:  # one fixed config, no search (fast)
        cfg = dict(encoders=list(HYB_ENC), center=True, transform="white",
                   pca_dims=PCA_DIMS, alpha=PCA_ALPHA, k=args.k, temp=args.temp, agg="vote",
                   classes={str(k): v for k, v in classes.items()})
        if args.tfidf != "none":
            cfg["tfidf"] = {args.tfidf: args.lam}
        print("Config:", {k: v for k, v in cfg.items() if k != "classes"})
        fit_artifacts(df, get_combined(df, cfg["encoders"], assets), cfg, args.out_dir)
        print("Saved artifacts to", args.out_dir); return
    if not args.search:  # DEFAULT: hybrid search (whitened emb + word/char TF-IDF), then save best
        E = get_combined(df, HYB_ENC, assets)
        res, nested = search_hybrid(df, E, fold, classes)
        res.to_csv("results_approach2.csv", index=False)
        base = res[(res["w_emb"] == 0.4) & (res["w_word"] == 0.6) & (res["w_char"] == 0)
                   & (res["k"] == 30) & (res["temp"] == 0.1) & (res["agg"] == "vote")]
        if len(base):
            print("previous model (word TF-IDF lam 0.6, K30): mean %.4f" % base["mean_qwk"].iloc[0])
        print("\nTOP 8:\n", res.head(8).to_string(index=False))
        best = res.iloc[0]
        print("\nBEST (optimistic, picked on these folds): mean %.4f std %.4f" % (best["mean_qwk"], best["std_qwk"]))
        print("NESTED estimate (config chosen on other folds only): mean %.4f std %.4f" % nested)
        tf = {kd: float(best[c]) for kd, c in (("word", "w_word"), ("char", "w_char")) if best[c] > 0}
        cfg = dict(encoders=list(HYB_ENC), center=True, transform="white", pca_dims=PCA_DIMS,
                   alpha=PCA_ALPHA, k=int(best["k"]), temp=float(best["temp"]), agg=str(best["agg"]),
                   tfidf=tf, classes={str(k): v for k, v in classes.items()})
        print("Config:", {k: v for k, v in cfg.items() if k != "classes"})
        fit_artifacts(df, E, cfg, args.out_dir)
        print("Saved artifacts to", args.out_dir); return
    all_res = []
    for names in ENCODER_SETS:
        label = "+".join(names)
        print("Encoders:", label, flush=True)
        r = cross_validate(df, get_combined(df, names, assets), fold, classes)
        r.insert(0, "encoders", label); all_res.append(r)
        print(r.sort_values("mean_qwk", ascending=False).head(5).to_string(index=False), flush=True)
    res = pd.concat(all_res).sort_values("mean_qwk", ascending=False)
    res.to_csv("results_approach2.csv", index=False)
    # ablation summary for the report: best QWK per (encoders, center)
    print("\nBest per encoder set / centering:\n",
          res.groupby(["encoders", "center"])["mean_qwk"].max().unstack().round(4).to_string())
    best = res.iloc[0]; print("\nBEST:\n", best)
    cfg = dict(encoders=best["encoders"].split("+"), center=bool(best["center"]), k=int(best["k"]),
               temp=float(best["temp"]), agg=str(best["agg"]),
               classes={str(k): v for k, v in classes.items()})
    fit_artifacts(df, get_combined(df, cfg["encoders"], assets), cfg, args.out_dir)
    print("Saved artifacts to", args.out_dir)


if __name__ == "__main__":
    main()