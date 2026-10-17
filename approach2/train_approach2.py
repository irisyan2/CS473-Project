"""Local-only: cross-validate Approach 2 and write the artifacts to weights/.

  python train_approach2.py --csv training_dataset.csv [--assets local_assets/assets.json]
Default (no flags): builds the best-known model (PCA-whitened minilm+bge_small embeddings +
word TF-IDF hybrid, K=30, temp 0.1, vote; 5-fold CV QWK ~0.5869) and saves it to weights/.
Use --search for the original centering-only search. Verify with check_approach2.py --folds all.
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
                    help="run the ORIGINAL centering-only config search (CV ~0.5748) instead of "
                         "building the default best model")
    ap.add_argument("--whiten", action="store_true", help="(default behaviour; kept for compatibility)")
    ap.add_argument("--out-dir", default="weights", help="where to save artifacts")
    ap.add_argument("--tfidf", choices=["word", "char", "none"], default="word",
                    help="TF-IDF hybrid similarity kind ('none' = embeddings only)")
    ap.add_argument("--lam", type=float, default=0.6, help="TF-IDF weight in the hybrid (0..1)")
    ap.add_argument("--k", type=int, default=30); ap.add_argument("--temp", type=float, default=0.1)
    args = ap.parse_args(argv)
    assets = json.load(open(args.assets)) if args.assets else None
    print("Using local assets:", args.assets)
    df = pd.read_csv(args.csv)
    fold = make_folds(df)
    classes = {q: sorted(df.loc[df.EssaySet == q, "Score1"].unique().tolist()) for q in df.EssaySet.unique()}
    if not args.search:  # DEFAULT: whitened embeddings + word TF-IDF hybrid (CV QWK ~0.5869)
        cfg = dict(encoders=["minilm", "bge_small"], center=True, transform="white",
                   pca_dims=64, alpha=0.5, k=args.k, temp=args.temp, agg="vote",
                   classes={str(k): v for k, v in classes.items()})
        if args.tfidf != "none":
            cfg["tfidf"] = dict(kind=args.tfidf, lam=args.lam)
        print("Config:", {k: v for k, v in cfg.items() if k != "classes"})
        fit_artifacts(df, get_combined(df, cfg["encoders"], assets), cfg, args.out_dir)
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