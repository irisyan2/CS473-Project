"""Local-only: cross-validate Approach 2 (cosine KNN on sentence embeddings) and write artifacts.

  python train_approach2.py --csv training_dataset.csv [--assets local_assets/assets.json]

Modes
  (default)    Search cosine KNN on PCA-whitened MiniLM+BGE-small embeddings (K, temperature,
               vote/median), with and without a per-question supervised LDA/NCA projection of
               that space. The projection is kept only if it beats plain embedding KNN by more
               than MET_MARGIN in CV. Prints the best config, a NESTED estimate (config chosen on
               the other folds only), and per-set QWK. Saves the winner to --out-dir.
  --no-metric  Same search, but always save the plain embedding-KNN model (no projection).
  --fixed      No search: build one whitened-embedding KNN from --k / --temp (vote).
  --search     The original ablation: encoder sets x per-question centering x K x temp x agg.
               Use its printed table for the report.

Verify any result with: python check_approach2.py --csv ... --folds all
Outputs: folds.csv (reuse for approaches 1 & 3), cache/emb_<enc>.npy, results_approach2.csv,
         <out-dir>/approach2_knn.npz + approach2_config.json   (default out-dir: weights/)
"""
import argparse, itertools, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
import approach2 as A

N_FOLDS, SEED = 5, 42

# --- original ablation search (--search) -------------------------------------------------
ENCODER_SETS = [["minilm"], ["bge_small"], ["minilm", "bge_small"]]  # >=2 encoders compared
KS = [5, 10, 20, 30, 50, 80]
TEMPS = [0.05, 0.1, 0.2, 1e6]          # 1e6 == uniform weights
AGGS = ["vote", "mean"]
CENTERS = [False, True]                # ablation: per-question centering off / on

# --- default search: whitened embeddings (+ optional supervised metric) ------------------
ENC = ["minilm", "bge_small"]
PCA_DIMS, PCA_ALPHA = 64, 0.5
MET_KINDS = ["lda", "nca"]
MET_WEIGHTS = [0.25, 0.5, 0.75, 1.0]   # weight of the metric-space similarity
MET_DIMS = {"lda": 3, "nca": 16}
MET_KS = [20, 30]
MET_TEMPS = [0.05, 0.1, 0.2]
MET_AGGS = ["vote", "median"]
MET_MARGIN = 0.003                     # metric model must beat the baseline by this to be used


# ---------------------------------------------------------------- data / embeddings
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


# ---------------------------------------------------------------- original ablation (--search)
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


# ---------------------------------------------------------------- default search
def search_metric(df, E, fold, classes):
    """CV over whitened-embedding cosine KNN, optionally blended with cosine in an LDA/NCA space
    learned on each fold's training data (per essay set); K, temperature, vote/median.
    metric == "none" is plain embedding KNN. Whitening and the metric are fit on the train part
    of each fold only. Returns (results sorted by mean_qwk, nested (mean, std), per-set table).
    The nested score picks the config on the OTHER folds only, so it is less optimistic than
    the best row."""
    y = df["Score1"].to_numpy(); es = df["EssaySet"].to_numpy()
    qlist = sorted(np.unique(es)); nq, nf = len(qlist), N_FOLDS
    msims = [("none", 0.0)] + [(kd, m) for kd in MET_KINDS for m in MET_WEIGHTS]
    cfgs = [(ms, k, t, a) for ms in msims for k in MET_KS for t in MET_TEMPS for a in MET_AGGS]
    cidx = {c: i for i, c in enumerate(cfgs)}; nc = len(cfgs); kmax = max(MET_KS)
    pred = np.zeros((nc, len(df)), dtype=int)
    for f in range(nf):
        for q in qlist:
            tr = np.where((es == q) & (fold != f))[0]; va = np.where((es == q) & (fold == f))[0]
            mu, W = A.whiten_fit(E[tr], PCA_DIMS, PCA_ALPHA)
            Ztr, Zva = A.project(E[tr], mu, W), A.project(E[va], mu, W)
            S_base = Zva @ Ztr.T  # cosine in the whitened embedding space
            S_met = {}
            for kd in MET_KINDS:
                P, b = A.fit_metric(Ztr, y[tr], kd, MET_DIMS[kd])
                S_met[kd] = A.l2(Zva @ P + b) @ A.l2(Ztr @ P + b).T
            for ms in msims:
                kd, m = ms
                Sim = S_base if kd == "none" else (1 - m) * S_base + m * S_met[kd]
                idx = np.argsort(-Sim, axis=1)[:, :kmax]
                s = np.take_along_axis(Sim, idx, axis=1); lab = y[tr][idx]
                for k in MET_KS:
                    for t in MET_TEMPS:
                        wt = A.weights_for(s[:, :k], t)
                        for a in MET_AGGS:
                            pred[cidx[(ms, k, t, a)], va] = A.aggregate(lab[:, :k], wt, classes[q], a)
        print("  search: fold", f, "done", flush=True)
    Qf = np.zeros((nc, nq, nf)); Qo = np.zeros((nc, nq, nf))
    for ci in range(nc):
        for qi, q in enumerate(qlist):
            m_ = es == q
            for f in range(nf):
                a_, b_ = m_ & (fold == f), m_ & (fold != f)
                Qf[ci, qi, f] = qwk(y[a_], pred[ci][a_], classes[q])
                Qo[ci, qi, f] = qwk(y[b_], pred[ci][b_], classes[q])
    nested = [Qf[int(np.argmax(Qo[:, :, f].mean(1))), :, f].mean() for f in range(nf)]
    rows = [dict(metric=c[0][0], m_weight=c[0][1], k=c[1], temp=c[2], agg=c[3],
                 mean_qwk=Qf[i].mean(), std_qwk=Qf[i].mean(0).std()) for i, c in enumerate(cfgs)]
    res = pd.DataFrame(rows).sort_values("mean_qwk", ascending=False)
    base_i = cidx[cfgs[int(res[res["metric"] == "none"].index[0])]]; best_i = cidx[cfgs[int(res.index[0])]]
    perq = pd.DataFrame({"set": qlist, "plain_knn": Qf[base_i].mean(1), "best": Qf[best_i].mean(1)})
    return res, (np.mean(nested), np.std(nested)), perq


# ---------------------------------------------------------------- artifacts
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
    else:  # per-question centering (or none, if cfg["center"] is False)
        zero = np.zeros(E.shape[1])
        mu = np.stack([E[qs == q].mean(0) if cfg["center"] else zero for q in uq]).astype(np.float32)
        mu_pool = (E.mean(0) if cfg["center"] else zero).astype(np.float32)
    if cfg.get("metric"):  # per-question supervised metric, fit on the whitened training vectors
        assert cfg.get("transform") == "white", "metric needs the whitened space"
        mc = cfg["metric"]; ys = df["Score1"].to_numpy(); Ps, bs = [], []
        for q, (mu_q, W_q) in zip(uq, fits):
            m_ = qs == q
            P, b = A.fit_metric(A.project(E[m_], mu_q, W_q), ys[m_], mc["kind"], mc["dims"])
            Ps.append(P); bs.append(b)
        dm = max(p_.shape[1] for p_ in Ps)  # LDA dims vary with #grades: zero-pad (no effect on cosine)
        extra["M_P"] = np.stack([np.pad(p_, ((0, 0), (0, dm - p_.shape[1]))) for p_ in Ps])
        extra["M_b"] = np.stack([np.pad(b_, (0, dm - len(b_))) for b_ in bs])
    np.savez_compressed(out / "approach2_knn.npz", emb=E.astype(np.float16),
                        y=df["Score1"].to_numpy().astype(np.int16), qid=qs,
                        mu_qid=uq, mu=mu, mu_pool=mu_pool, **extra)
    (out / "approach2_config.json").write_text(json.dumps(cfg))


def whitened_cfg(classes, k, temp, agg):
    return dict(encoders=list(ENC), center=True, transform="white", pca_dims=PCA_DIMS,
                alpha=PCA_ALPHA, k=int(k), temp=float(temp), agg=str(agg),
                classes={str(q): v for q, v in classes.items()})


def show(cfg):
    print("Config:", {k: v for k, v in cfg.items() if k != "classes"})


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="training_dataset.csv")
    ap.add_argument("--assets", default="local_assets/assets.json" if Path("local_assets/assets.json").exists() else None)
    ap.add_argument("--out-dir", default="weights", help="where to save artifacts")
    ap.add_argument("--search", action="store_true", help="original encoder x centering ablation (CV ~0.5748)")
    ap.add_argument("--fixed", action="store_true", help="no search: whitened embedding KNN from --k/--temp")
    ap.add_argument("--no-metric", action="store_true", help="always save plain embedding KNN (no LDA/NCA)")
    ap.add_argument("--k", type=int, default=30); ap.add_argument("--temp", type=float, default=0.1)
    args = ap.parse_args(argv)
    assets = json.load(open(args.assets)) if args.assets else None
    print("Using local assets:", args.assets)
    df = pd.read_csv(args.csv)
    fold = make_folds(df)
    classes = {q: sorted(df.loc[df.EssaySet == q, "Score1"].unique().tolist()) for q in df.EssaySet.unique()}

    if args.fixed:
        cfg = whitened_cfg(classes, args.k, args.temp, "vote"); show(cfg)
        fit_artifacts(df, get_combined(df, cfg["encoders"], assets), cfg, args.out_dir)
        print("Saved artifacts to", args.out_dir); return

    if not args.search:  # DEFAULT
        E = get_combined(df, ENC, assets)
        res, nested, perq = search_metric(df, E, fold, classes)
        res.to_csv("results_approach2.csv", index=False)
        base = res[res["metric"] == "none"].iloc[0]  # best plain embedding KNN
        best = res.iloc[0]
        print("\nPlain embedding KNN (best K/temp/agg): mean %.4f std %.4f  K=%d temp=%.2f agg=%s"
              % (base["mean_qwk"], base["std_qwk"], base["k"], base["temp"], base["agg"]))
        print("\nTOP 8:\n", res.head(8).to_string(index=False))
        print("\nBest per metric kind:\n", res.groupby("metric")["mean_qwk"].max().round(4).to_string())
        print("\nBEST (optimistic, picked on these folds): mean %.4f std %.4f" % (best["mean_qwk"], best["std_qwk"]))
        print("NESTED estimate (config chosen on other folds only): mean %.4f std %.4f" % nested)
        print("\nPer-set QWK, plain KNN vs best:\n", perq.round(3).to_string(index=False))
        use = best
        if args.no_metric and best["metric"] != "none":
            print("\n--no-metric: saving plain embedding KNN."); use = base
        elif best["metric"] != "none" and best["mean_qwk"] < base["mean_qwk"] + MET_MARGIN:
            print("\nMetric gain < %.3f margin -> treated as noise; saving plain embedding KNN." % MET_MARGIN)
            use = base
        cfg = whitened_cfg(classes, use["k"], use["temp"], use["agg"])
        if use["metric"] != "none":
            cfg["metric"] = dict(kind=str(use["metric"]), weight=float(use["m_weight"]),
                                 dims=MET_DIMS[str(use["metric"])])
        show(cfg)
        fit_artifacts(df, E, cfg, args.out_dir)
        print("Saved artifacts to", args.out_dir); return

    all_res = []  # --search
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