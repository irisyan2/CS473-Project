"""Leak-free test of the real inference code. For each chosen fold, rebuild artifacts from the
OTHER folds with the saved config, run approach2.load_model/predict exactly as the grader would,
validate the output and report QWK. Also tests the unseen-question fallback.

  python check_approach2.py --csv training_dataset_updated.csv --folds folds_updated.csv [--fold all]
"""
import argparse, json, tempfile
from pathlib import Path
import numpy as np
import approach2 as A
import train_approach2 as T


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="training_dataset_updated.csv")
    ap.add_argument("--folds", default="folds_updated.csv" if Path("folds_updated.csv").exists() else "folds.csv")
    ap.add_argument("--assets", default="local_assets/assets.json" if Path("local_assets/assets.json").exists() else None)
    ap.add_argument("--fold", default="0", help="0-4 or 'all'")
    ap.add_argument("--weights", default="weights")
    args = ap.parse_args(argv)
    assets = json.load(open(args.assets)) if args.assets else None
    df, classes = T.load_data(args.csv)
    fold = T.get_folds(df, args.folds)
    cfg = json.load(open(Path(args.weights) / "approach2_config.json"))
    E = T.get_combined(df, cfg["encoders"], assets)
    qdict = {q: {"allowed_grades": g} for q, g in cfg["classes"].items()}
    folds = range(T.N_FOLDS) if args.fold == "all" else [int(args.fold)]
    pred = np.full(len(df), np.nan)
    for f in folds:
        tr, va = fold != f, fold == f
        with tempfile.TemporaryDirectory() as tmp:
            T.fit_artifacts(df[tr], E[tr], cfg, tmp)
            model = A.load_model(tmp, {}, assets)
            v = df[va]
            rows = [{"id": str(i), "question": str(q), "text": t} for i, q, t in zip(v.Id, v.EssaySet, v.EssayText)]
            out = model.predict(rows, qdict)
            fb = model.predict([{"id": "u1", "question": "99999", "text": "an answer to an unseen question"}],
                               {"99999": {"allowed_grades": [0, 0.5, 1, 1.5, 2]}})
        assert [o["id"] for o in out] == [r["id"] for r in rows], "ids must be returned once, in order"
        for o, r in zip(out, rows):
            assert type(o["score"]) in (int, float) and np.isfinite(o["score"]), "bad score type"
            assert any(abs(o["score"] - g) < 1e-9 for g in cfg["classes"][r["question"]]), "score not on the grade grid"
        assert fb[0]["score"] in [0, 0.5, 1, 1.5, 2], "fallback off-grid"
        pred[va] = [o["score"] for o in out]
        print(f"fold {f}: interface OK ({len(out)} predictions); unseen-question fallback -> {fb[0]['score']}")
    m = ~np.isnan(pred)
    d2 = df[m]; p2 = pred[m]; es = d2.EssaySet.to_numpy(); y = d2.Score1.to_numpy()
    cnt = df.groupby("EssaySet").size()
    qs = sorted(classes)
    r = {q: T.qwk_q(y[es == q], p2[es == q], classes[q]) for q in qs}
    big = [r[q] for q in qs if cnt[q] >= cfg["large_n"]]; small = [r[q] for q in qs if cnt[q] < cfg["large_n"]]
    print(f"\nQWK on checked folds ({'all' if args.fold == 'all' else 'fold ' + args.fold}); "
          "constant-grade questions excluded")
    print(f"  large questions: {np.nanmean(big):.4f}   small questions: {np.nanmean(small):.4f}   "
          f"ALL (equal weight): {np.nanmean(list(r.values())):.4f}")
    if args.fold != "all":
        print("  (one fold = only 5 answers per small question, so this number is noisy; use --fold all)")


if __name__ == "__main__":
    main()