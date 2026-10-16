"""Local-only: cross-validate Approach 2 and write the artifacts to weights/.

  python train_approach2.py --csv training_dataset.csv
Outputs: folds.csv (reuse for approaches 1 & 3), cache/emb_<enc>.npy,
         results_approach2.csv, weights/approach2_knn.npz + approach2_config.json
"""
import argparse, itertools, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
import approach2 as A

ENCODERS = ["minilm", "bge_small"]          # compare >= 2 (add multiqa_minilm, e5_small if time)
KS = [1, 3, 5, 7, 10, 15, 20, 30, 50]
WEIGHTINGS = ["uniform", "softmax"]
AGGS = ["vote", "mean"]
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


def qwk(y, p, allowed):
    if len(set(y)) < 2 and len(set(p)) < 2:
        return 1.0 if list(y)[:1] == list(p)[:1] else 0.0
    return cohen_kappa_score(y, p, labels=allowed, weights="quadratic")


def cross_validate(df, E, fold, classes):
    configs = list(itertools.product(KS, WEIGHTINGS, AGGS))
    pred = {c: np.zeros(len(df), dtype=int) for c in configs}
    y = df["Score1"].to_numpy(); es = df["EssaySet"].to_numpy()
    kmax = max(KS)
    for f in range(N_FOLDS):
        for q in np.unique(es):
            tr = np.where((es == q) & (fold != f))[0]; va = np.where((es == q) & (fold == f))[0]
            sims = E[va] @ E[tr].T
            idx = np.argsort(-sims, axis=1)[:, :kmax]
            s = np.take_along_axis(sims, idx, axis=1); lab = y[tr][idx]
            for (k, wt, ag) in configs:
                pred[(k, wt, ag)][va] = A.aggregate(lab[:, :k], A.weights_for(s[:, :k], wt), classes[q], ag)
    rows = []
    for c in configs:
        per_fold = []
        for f in range(N_FOLDS):
            per_q = [qwk(y[(es == q) & (fold == f)], pred[c][(es == q) & (fold == f)], classes[q])
                     for q in np.unique(es)]
            per_fold.append(np.mean(per_q))
        rows.append(dict(k=c[0], weighting=c[1], agg=c[2],
                         mean_qwk=np.mean(per_fold), std_qwk=np.std(per_fold)))
    return pd.DataFrame(rows)


def fit_artifacts(df, E, cfg, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "approach2_knn.npz", emb=E.astype(np.float32),
                        y=df["Score1"].to_numpy().astype(np.int16),
                        qid=df["EssaySet"].astype(str).to_numpy().astype("U16"))
    (out / "approach2_config.json").write_text(json.dumps(cfg))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--csv", default="training_dataset.csv")
    ap.add_argument("--assets", default="local_assets/assets.json" if Path("local_assets/assets.json").exists() else None)
    args = ap.parse_args()
    assets = json.load(open(args.assets)) if args.assets else None
    print("Using local assets:" , args.assets)
    df = pd.read_csv(args.csv)
    fold = make_folds(df)
    classes = {q: sorted(df.loc[df.EssaySet == q, "Score1"].unique().tolist()) for q in df.EssaySet.unique()}
    all_res = []
    for enc in ENCODERS:
        print("Encoding with", enc, flush=True)
        E = get_embeddings(df, enc, assets)
        r = cross_validate(df, E, fold, classes); r.insert(0, "encoder", enc); all_res.append(r)
        print(r.sort_values("mean_qwk", ascending=False).head(5).to_string(index=False), flush=True)
    res = pd.concat(all_res).sort_values("mean_qwk", ascending=False)
    res.to_csv("results_approach2.csv", index=False)
    best = res.iloc[0]; print("\nBEST:\n", best)
    cfg = dict(encoder=best["encoder"], k=int(best["k"]), weighting=best["weighting"], agg=best["agg"],
               classes={str(k): v for k, v in classes.items()})
    fit_artifacts(df, get_embeddings(df, best["encoder"], assets), cfg, "weights")
    print("Saved artifacts to weights/")