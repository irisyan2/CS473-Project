"""Leak-free sanity test: rebuild artifacts from folds != f using a config, run
approach2.load_model/predict on fold f exactly as the grader would, and validate output.
  python check_approach2.py --csv training_dataset.csv [--assets local_assets/assets.json]
         [--config weights/approach2_config.json] [--folds 0 | 0,1 | all]
--folds all gives full 5-fold CV through the production path (compare with the
experiment CSV's mean_qwk).
"""
import argparse, json, os, tempfile
import numpy as np, pandas as pd
import approach2 as A
import train_approach2 as T

ap = argparse.ArgumentParser(); ap.add_argument("--csv", default="training_dataset.csv")
ap.add_argument("--assets", default="local_assets/assets.json" if os.path.exists("local_assets/assets.json") else None)
ap.add_argument("--config", default="weights/approach2_config.json")
ap.add_argument("--folds", default="0", help="comma list of fold ids, or 'all'")
args = ap.parse_args()
assets = json.load(open(args.assets)) if args.assets else None
df = pd.read_csv(args.csv); fold = T.make_folds(df)
cfg = json.load(open(args.config))
E = T.get_combined(df, cfg["encoders"], assets)
folds = range(T.N_FOLDS) if args.folds == "all" else [int(x) for x in args.folds.split(",")]
questions = {str(q): {"allowed_grades": g} for q, g in cfg["classes"].items()}
scores = []
for f in folds:
    tr, va = fold != f, fold == f
    with tempfile.TemporaryDirectory() as tmp:
        T.fit_artifacts(df[tr], E[tr], cfg, tmp)
        model = A.load_model(tmp, {}, assets)
        val = df[va]
        rows = [{"id": str(i), "question": str(q), "text": t} for i, q, t in zip(val.Id, val.EssaySet, val.EssayText)]
        out = model.predict(rows, questions)
    # interface checks
    assert len(out) == len(rows) and [o["id"] for o in out] == [r["id"] for r in rows], "ids"
    assert all(type(o["score"]) is int for o in out), "scores must be python ints"
    assert all(o["score"] in questions[r["question"]]["allowed_grades"] for o, r in zip(out, rows)), "out of scale"
    print("Fold %d interface OK: %d predictions" % (f, len(out)))
    p = pd.Series([o["score"] for o in out], index=val.index)
    qs = [T.qwk(val.Score1[val.EssaySet == q], p[val.EssaySet == q], cfg["classes"][str(q)]) for q in sorted(val.EssaySet.unique())]
    scores.append(np.mean(qs))
    print("Fold %d mean QWK: %.4f" % (f, scores[-1])); print([round(x, 3) for x in qs])
if len(scores) > 1:
    print("Mean QWK over %d folds: %.4f (std %.4f)" % (len(scores), np.mean(scores), np.std(scores)))
