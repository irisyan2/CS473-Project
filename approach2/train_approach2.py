"""Local-only: cross-validate Approach 2 (cosine KNN on sentence embeddings) and write artifacts.

  python train_approach2.py --csv training_dataset_updated.csv --folds folds_updated.csv
  python train_approach2.py --csv ... --folds ... --compare     # also compare encoder sets (report)

The updated data has two kinds of questions, so settings are chosen separately for each:
  large : >= LARGE_N graded answers (the original 10 questions): centre / PCA-whiten (+ NCA metric)
  small : the 81 new questions (~25 answers, 0-5 scale in half-point steps): per-question centring
Selection metric = mean over questions of QWK computed on that question's pooled out-of-fold
predictions (a 5-row fold is too small for a stable QWK). Questions whose true grades are all
equal are skipped (QWK undefined). The handout-style per-fold mean +/- std is printed as well.
Aggregators: mean | median | vote | cal. "cal" is the weighted mean stretched away from the
question's mean grade by a per-question factor gamma (see approach2.py): neighbour averages are
too close to the middle, which hurts quadratic-weighted kappa. gamma is fit by leave-one-out on the
TRAINING part of each fold only, so there is no leakage; the CV search keeps "cal" only if it wins.
Options: --encoders a b c (e.g. add e5_small multiqa_minilm), --no-nca (skip the slow NCA search).
Outputs: cache/emb_<enc>_<datakey>.npy, results_approach2_*.csv, weights/approach2_*.
"""
import argparse, hashlib, itertools, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import cohen_kappa_score
import approach2 as A

N_FOLDS, SEED = 5, 42
LARGE_N = 200
DEFAULT_ENC = ["minilm", "bge_small"]
ENCODER_SETS = [["minilm"], ["bge_small"], ["minilm", "bge_small"]]
PCA_DIMS, PCA_ALPHA = 64, 0.5
MET = {"kind": "nca", "weight": 0.5, "dims": 16}
MET_MARGIN = 0.003                     # NCA metric is used only if it beats the baseline by this
LARGE_KS, LARGE_TEMPS, LARGE_AGGS = [10, 20, 30, 50], [0.05, 0.1, 0.2], ["mean", "median", "vote", "cal"]
SMALL_KS, SMALL_TEMPS, SMALL_AGGS = [2, 3, 4, 5, 7, 10, 15], [0.05, 0.1, 0.2, 0.3, 1e6], ["mean", "median", "vote", "cal"]
SMALL_CENTERS = [False, True]          # ablation: per-question centring off / on
XQ_BETAS = [0.1, 0.2, 0.3, 0.4]                  # weight of the cross-question blend (small questions, agg "cal"); 0 = off
FALLBACK = {"k": 15, "temp": 0.1}      # unseen questions


# ---------------------------------------------------------------- data
def grade_grid(scores):
    """Allowed grades for a question as inferred from its data (the grader supplies the real ones
    at test time): 0..ceil(max); half-point steps if any half-point appears or the max is 5."""
    s = np.asarray(scores, float)
    mx = float(np.ceil(s.max()))
    step = 0.5 if (mx >= 5 or np.any(s % 1 != 0)) else 1.0
    return np.arange(0, mx + 1e-9, step)


def load_data(csv):
    df = pd.read_csv(csv)
    df["Score1"] = df["Score1"].astype(float)
    df["EssaySet"] = df["EssaySet"].astype(int)
    classes = {q: grade_grid(df.loc[df.EssaySet == q, "Score1"]) for q in sorted(df.EssaySet.unique())}
    return df, classes


def get_folds(df, path):
    """Shared folds (Id, [EssaySet], fold). If absent, build them: inside each question, order by
    grade and deal rows round-robin so every fold sees every grade level."""
    if path and Path(path).exists():
        f = pd.read_csv(path)[["Id", "fold"]]
        fold = df[["Id"]].merge(f, on="Id", how="left")["fold"]
        assert fold.notna().all(), f"{path} does not cover every Id in the CSV"
        return fold.to_numpy().astype(int)
    rng = np.random.RandomState(SEED)
    fold = np.zeros(len(df), dtype=int)
    for q, idx in df.groupby("EssaySet").indices.items():
        o = idx[np.lexsort((rng.rand(len(idx)), df.Score1.to_numpy()[idx]))]
        fold[o] = np.arange(len(o)) % N_FOLDS
    out = path or "folds.csv"
    pd.DataFrame({"Id": df.Id, "EssaySet": df.EssaySet, "fold": fold}).to_csv(out, index=False)
    print("wrote", out)
    return fold


def data_key(df):
    h = hashlib.md5(("|".join(map(str, df.Id)) + "".join(df.EssayText.astype(str))).encode())
    return h.hexdigest()[:8]


def get_embeddings(df, enc, assets=None):
    """Cached per encoder AND per dataset version, so a changed CSV never reuses stale vectors."""
    p = Path("cache") / f"emb_{enc}_{data_key(df)}.npy"
    p.parent.mkdir(exist_ok=True)
    if p.exists():
        return np.load(p)
    E = A.encode_texts(A.get_encoder(enc, assets), df["EssayText"].tolist(), enc)
    np.save(p, E)
    return E


def get_combined(df, names, assets=None):
    return A.l2(np.hstack([get_embeddings(df, n, assets) for n in names]))


# ---------------------------------------------------------------- metric
def qwk_q(yt, yp, grid):
    """Quadratic weighted kappa on grade-grid indices; NaN if the true grades are constant."""
    yt = np.asarray(yt, float); yp = np.asarray(yp, float)
    if len(np.unique(yt)) < 2:
        return np.nan
    ti = np.abs(yt[:, None] - grid[None, :]).argmin(1)
    pi = np.abs(yp[:, None] - grid[None, :]).argmin(1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cohen_kappa_score(ti, pi, labels=list(range(len(grid))), weights="quadratic")


def oof_mean(df, pred, qs, classes):
    y = df.Score1.to_numpy(); es = df.EssaySet.to_numpy()
    r = [qwk_q(y[es == q], pred[es == q], classes[q]) for q in qs]
    return float(np.nanmean(r))


def foldwise(df, pred, fold, qs, classes):
    """Handout-style: per fold, mean over questions of QWK on that fold's rows; mean/std over folds."""
    y = df.Score1.to_numpy(); es = df.EssaySet.to_numpy(); out = []
    for f in range(N_FOLDS):
        r = [qwk_q(y[(es == q) & (fold == f)], pred[(es == q) & (fold == f)], classes[q]) for q in qs]
        if not np.all(np.isnan(r)):
            out.append(np.nanmean(r))
    return float(np.mean(out)), float(np.std(out))


# ---------------------------------------------------------------- searches
def search_small(df, E, fold, classes, qs, centers=SMALL_CENTERS):
    y = df.Score1.to_numpy(); es = df.EssaySet.to_numpy(); kmax = max(SMALL_KS)
    cfgs = [(ce, k, t, a, b) for ce in centers for k in SMALL_KS for t in SMALL_TEMPS for a in SMALL_AGGS
            for b in ([0.0] + XQ_BETAS if a == "cal" else [0.0])]
    pred = {c: np.full(len(df), np.nan) for c in cfgs}
    frac = y / np.array([max(classes[q]) for q in es])
    poolc = {}

    def pool(f):  # training rows of fold f, in the pooled cross-question space (as in approach2.py)
        if f not in poolc:
            trm = fold != f; mp = E[trm].mean(0)
            poolc[f] = (A.l2(E[trm] - mp), frac[trm], es[trm], mp)
        return poolc[f]
    for q in qs:
        for f in range(N_FOLDS):
            tr = np.where((es == q) & (fold != f))[0]; va = np.where((es == q) & (fold == f))[0]
            if len(tr) == 0 or len(va) == 0:
                continue
            cx = None
            if XQ_BETAS:  # nearest answers to OTHER questions, as a grade for this question
                Zp, fp, qp, mp = pool(f); other = qp != q
                cx = A.knn_mean(A.l2(E[va] - mp) @ Zp[other].T, fp[other], FALLBACK["k"], FALLBACK["temp"]) * max(classes[q])
            for center in centers:
                mu = E[tr].mean(0) if center else 0.0
                Ztr, Zva = A.l2(E[tr] - mu), A.l2(E[va] - mu)
                S = Zva @ Ztr.T
                idx = np.argsort(-S, axis=1)[:, :kmax]
                s = np.take_along_axis(S, idx, 1); lab = y[tr][idx]
                tt = A.loo_topk(Ztr @ Ztr.T, kmax); gc = {}  # leave-one-out neighbours of the training part
                for c in cfgs:
                    if c[0] == center:
                        w = A.weights_for(s[:, :c[1]], c[2])
                        if c[3] == "cal":
                            if (c[1], c[2]) not in gc:
                                gc[(c[1], c[2])] = A.spread_from_topk(tt[0], tt[1], y[tr], c[1], c[2])
                            g, yb = gc[(c[1], c[2])]
                            cont = A.calibrated_cont(lab[:, :c[1]], w, g, yb)
                            if c[4]:
                                cont = (1 - c[4]) * cont + c[4] * cx
                            pred[c][va] = A.snap(cont, classes[q])
                        else:
                            pred[c][va] = A.aggregate(lab[:, :c[1]], w, classes[q], c[3])
    rows = [dict(center=c[0], k=c[1], temp=c[2], agg=c[3], xq=c[4], oof_qwk=oof_mean(df, pred[c], qs, classes))
            for c in cfgs]
    return pd.DataFrame(rows).sort_values("oof_qwk", ascending=False), pred


def search_large(df, E, fold, classes, qs, full=True, nca=True):
    y = df.Score1.to_numpy(); es = df.EssaySet.to_numpy(); kmax = max(LARGE_KS)
    variants = [("center", None)] + ([("white", None)] + ([("white", "nca")] if nca else []) if full else [])
    cfgs = list(itertools.product(variants, LARGE_KS, LARGE_TEMPS, LARGE_AGGS))
    pred = {c: np.full(len(df), np.nan) for c in cfgs}
    for q in qs:
        for f in range(N_FOLDS):
            tr = np.where((es == q) & (fold != f))[0]; va = np.where((es == q) & (fold == f))[0]
            mu = E[tr].mean(0)
            Zc_tr, Zc_va = A.l2(E[tr] - mu), A.l2(E[va] - mu)
            sims = {("center", None): Zc_va @ Zc_tr.T}
            tts = {("center", None): A.train_sims(Zc_tr)}   # train x train, for the "cal" stretch
            if full:
                muw, W = A.whiten_fit(E[tr], PCA_DIMS, PCA_ALPHA)
                Ztr, Zva = A.project(E[tr], muw, W), A.project(E[va], muw, W)
                sims[("white", None)] = Zva @ Ztr.T
                tts[("white", None)] = A.train_sims(Ztr)
                if nca:
                    P, b = A.fit_metric(Ztr, y[tr], MET["kind"], MET["dims"])
                    Zm_tr, Zm_va = A.l2(Ztr @ P + b), A.l2(Zva @ P + b)
                    wgt = MET["weight"]
                    sims[("white", "nca")] = (1 - wgt) * sims[("white", None)] + wgt * (Zm_va @ Zm_tr.T)
                    tts[("white", "nca")] = A.train_sims(Ztr, Zm_tr, wgt)
            for v in variants:
                S = sims[v]; idx = np.argsort(-S, axis=1)[:, :kmax]
                s = np.take_along_axis(S, idx, 1); lab = y[tr][idx]
                tt = A.loo_topk(tts[v], kmax); gc = {}
                for c in cfgs:
                    if c[0] == v:
                        w = A.weights_for(s[:, :c[1]], c[2])
                        if c[3] == "cal":
                            if (c[1], c[2]) not in gc:
                                gc[(c[1], c[2])] = A.spread_from_topk(tt[0], tt[1], y[tr], c[1], c[2])
                            g, yb = gc[(c[1], c[2])]
                            pred[c][va] = A.calibrated(lab[:, :c[1]], w, g, yb, classes[q])
                        else:
                            pred[c][va] = A.aggregate(lab[:, :c[1]], w, classes[q], c[3])
        print(f"  large question {q} done", flush=True)
    rows = [dict(transform=c[0][0], metric=c[0][1] or "none", k=c[1], temp=c[2], agg=c[3],
                 oof_qwk=oof_mean(df, pred[c], qs, classes)) for c in cfgs]
    return pd.DataFrame(rows).sort_values("oof_qwk", ascending=False), pred, cfgs


def pick_large(res, cfgs, pred):
    """Best config; the NCA metric must beat the best non-metric config by MET_MARGIN."""
    nm = res[res["metric"] == "none"].iloc[0]
    best = nm
    mt = res[res["metric"] == "nca"]
    if len(mt) and mt.iloc[0]["oof_qwk"] > nm["oof_qwk"] + MET_MARGIN:
        best = mt.iloc[0]
    key = next(c for c in cfgs if c[0][0] == best["transform"] and (c[0][1] or "none") == best["metric"]
               and c[1] == best["k"] and c[2] == best["temp"] and c[3] == best["agg"])
    return best, key


# ---------------------------------------------------------------- artifacts
def fit_artifacts(df, E, cfg, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    y = df["Score1"].to_numpy(np.float32)
    qs = df["EssaySet"].astype(str).to_numpy().astype("U16")
    uq = np.unique(qs)
    lg, sm = cfg["groups"]["large"], cfg["groups"]["small"]
    gmax = np.array([max(cfg["classes"][q]) for q in qs], dtype=np.float32)
    mus, Wq, Ws, Ps, bs = [], [], [], [], []
    calq, calg, caly = [], [], []
    for q in uq:
        m = qs == q; Eq = E[m]; is_large = m.sum() >= cfg["large_n"]
        W = P = b = None
        if is_large and lg["transform"] == "white":
            mu, W = A.whiten_fit(Eq, lg["pca_dims"], lg["alpha"])
            Wq.append(q); Ws.append(W)
            if lg.get("metric"):
                P, b = A.fit_metric(A.project(Eq, mu, W), y[m], lg["metric"]["kind"], lg["metric"]["dims"])
                Ps.append(P); bs.append(b)
        elif is_large:
            mu = Eq.mean(0)
        else:
            mu = Eq.mean(0) if sm["center"] else np.zeros(E.shape[1])
        mus.append(np.asarray(mu, np.float32))
        grp = lg if is_large else sm
        if grp["agg"] == "cal":  # per-question stretch from leave-one-out predictions on its own answers
            Z = A.project(Eq, mu, W)
            Zm = A.l2(Z @ P + b) if P is not None else None
            gamma, yb = A.fit_spread(A.train_sims(Z, Zm, lg["metric"]["weight"] if P is not None else 0.0),
                                     y[m], grp["k"], grp["temp"])
            calq.append(q); calg.append(gamma); caly.append(yb)
    extra = {}
    if Ws:
        extra.update(W_qid=np.array(Wq).astype("U16"), W=np.stack(Ws).astype(np.float32))
    if Ps:
        extra.update(M_P=np.stack(Ps).astype(np.float32), M_b=np.stack(bs).astype(np.float32))
    if calq:
        extra.update(cal_qid=np.array(calq).astype("U16"), cal_gamma=np.array(calg, np.float32),
                     cal_ybar=np.array(caly, np.float32))
    np.savez_compressed(out / "approach2_knn.npz", emb=E.astype(np.float16), y=y, yfrac=y / gmax,
                        qid=qs, mu_qid=uq, mu=np.stack(mus), mu_pool=E.mean(0).astype(np.float32), **extra)
    (out / "approach2_config.json").write_text(json.dumps(cfg))


def pyval(v):
    return v.item() if hasattr(v, "item") else v


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="training_dataset_updated.csv")
    ap.add_argument("--folds", default="folds_updated.csv" if Path("folds_updated.csv").exists() else "folds.csv")
    ap.add_argument("--assets", default="local_assets/assets.json" if Path("local_assets/assets.json").exists() else None)
    ap.add_argument("--out-dir", default="weights")
    ap.add_argument("--compare", action="store_true", help="also compare encoder sets (for the report)")
    ap.add_argument("--encoders", nargs="+", default=DEFAULT_ENC,
                    help="encoders to concatenate, e.g. minilm bge_small e5_small multiqa_minilm "
                         "(each needs an entry in assets.json)")
    ap.add_argument("--no-nca", action="store_true", help="skip the (slow) NCA metric search")
    args = ap.parse_args(argv)
    assets = json.load(open(args.assets)) if args.assets else None
    print("assets:", args.assets, "| folds:", args.folds)
    df, classes = load_data(args.csv)
    fold = get_folds(df, args.folds)
    cnt = df.groupby("EssaySet").size()
    large = [q for q in classes if cnt[q] >= LARGE_N]; small = [q for q in classes if cnt[q] < LARGE_N]
    print(f"{len(df)} rows, {len(classes)} questions: {len(large)} large, {len(small)} small")

    if args.compare:
        rows = []
        for names in ENCODER_SETS:
            Ec = get_combined(df, names, assets)
            rs, _ = search_small(df, Ec, fold, classes, small, centers=[True])
            rl, _, _ = search_large(df, Ec, fold, classes, large, full=False)
            rows.append(dict(encoders="+".join(names), small_best=rs.iloc[0]["oof_qwk"],
                             large_best=rl.iloc[0]["oof_qwk"]))
            print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv("results_approach2_encoders.csv", index=False)

    E = get_combined(df, args.encoders, assets)
    print("encoders:", args.encoders)
    print("\n== small questions ==", flush=True)
    rs, ps = search_small(df, E, fold, classes, small)
    rs.to_csv("results_approach2_small.csv", index=False); print(rs.head(8).to_string(index=False))
    print("centering ablation (best OOF QWK):", rs.groupby("center").oof_qwk.max().round(4).to_dict())
    bs_ = rs.iloc[0]
    skey = (bool(bs_["center"]), int(bs_["k"]), float(bs_["temp"]), str(bs_["agg"]), float(bs_["xq"]))

    print("\n== large questions (whitening + NCA search; takes a few minutes) ==", flush=True)
    rl, pl, lcfgs = search_large(df, E, fold, classes, large, nca=not args.no_nca)
    rl.to_csv("results_approach2_large.csv", index=False); print(rl.head(8).to_string(index=False))
    bl, lkey = pick_large(rl, lcfgs, pl)

    # combined out-of-fold predictions with the chosen configs
    pred = np.full(len(df), np.nan)
    es = df.EssaySet.to_numpy()
    for q in small:
        pred[es == q] = ps[skey][es == q]
    for q in large:
        pred[es == q] = pl[lkey][es == q]
    allq = sorted(classes)
    print("\n== chosen settings, out-of-fold QWK ==")
    print(f"  large ({len(large)} q): {oof_mean(df, pred, large, classes):.4f}")
    print(f"  small ({len(small)} q): {oof_mean(df, pred, small, classes):.4f}")
    print(f"  ALL questions, equal weight: {oof_mean(df, pred, allq, classes):.4f}")
    fm, fs = foldwise(df, pred, fold, allq, classes)
    print(f"  handout-style per-fold mean +/- std (all questions): {fm:.4f} +/- {fs:.4f}")
    y = df.Score1.to_numpy()
    pd.DataFrame({"EssaySet": allq, "n": [cnt[q] for q in allq],
                  "qwk": [qwk_q(y[es == q], pred[es == q], classes[q]) for q in allq]}
                 ).to_csv("results_approach2_per_question.csv", index=False)

    cfg = dict(encoders=list(args.encoders), large_n=LARGE_N, fallback=FALLBACK,
               groups=dict(
                   large=dict(transform=str(bl["transform"]), pca_dims=PCA_DIMS, alpha=PCA_ALPHA,
                              metric=(MET if bl["metric"] == "nca" else None),
                              k=int(bl["k"]), temp=float(bl["temp"]), agg=str(bl["agg"])),
                   small=dict(center=bool(bs_["center"]), k=int(bs_["k"]), temp=float(bs_["temp"]), agg=str(bs_["agg"]),
                              **({"xq": {"weight": float(bs_["xq"])}} if bs_["xq"] > 0 else {}))),
               classes={str(q): [float(x) for x in g] for q, g in classes.items()})
    fit_artifacts(df, E, cfg, args.out_dir)
    print("\nsaved artifacts to", args.out_dir, "\nconfig:", {k: v for k, v in cfg.items() if k != "classes"})


if __name__ == "__main__":
    main()