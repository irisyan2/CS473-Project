"""Leak-free sanity test: rebuild artifacts from folds != 0 using the saved best config,
run approach2.load_model/predict on fold 0 exactly as the grader would, and validate output.
  python check_approach2.py --csv training_dataset.csv [--assets local_assets/assets.json]
"""
import argparse, json, tempfile
import numpy as np, pandas as pd
import approach2 as A
import train_approach2 as T

ap = argparse.ArgumentParser(); ap.add_argument("--csv", default="training_dataset.csv")
import os; ap.add_argument("--assets", default="local_assets/assets.json" if os.path.exists("local_assets/assets.json") else None); args = ap.parse_args()
assets = json.load(open(args.assets)) if args.assets else None
df = pd.read_csv(args.csv); fold = T.make_folds(df)
cfg = json.load(open("weights/approach2_config.json"))
E = T.get_embeddings(df, cfg["encoder"], assets)
tr, va = fold != 0, fold == 0
with tempfile.TemporaryDirectory() as tmp:
    T.fit_artifacts(df[tr], E[tr], cfg, tmp)
    model = A.load_model(tmp, {}, assets)
    val = df[va]
    rows = [{"id": str(i), "question": str(q), "text": t} for i, q, t in zip(val.Id, val.EssaySet, val.EssayText)]
    questions = {str(q): {"allowed_grades": g} for q, g in cfg["classes"].items()}
    out = model.predict(rows, questions)
# interface checks
assert len(out) == len(rows) and [o["id"] for o in out] == [r["id"] for r in rows], "ids"
assert all(type(o["score"]) is int for o in out), "scores must be python ints"
assert all(o["score"] in questions[r["question"]]["allowed_grades"] for o, r in zip(out, rows)), "out of scale"
print("Interface OK:", len(out), "predictions")
p = pd.Series([o["score"] for o in out], index=val.index)
qs = [T.qwk(val.Score1[val.EssaySet == q], p[val.EssaySet == q], cfg["classes"][str(q)]) for q in sorted(val.EssaySet.unique())]
print("Fold-0 mean QWK: %.4f" % np.mean(qs)); print([round(x, 3) for x in qs])