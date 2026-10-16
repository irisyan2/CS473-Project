"""Per-question QWK 5-fold check of the packaged approach1.py (grader metric)."""
import os, tempfile, json
import numpy as np, pandas as pd, joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
import train_approach1 as T
import approach1

df = pd.read_csv("training_dataset.csv")
if os.path.exists("folds.csv"):
    df = df.merge(pd.read_csv("folds.csv"), on="Id")
else:
    strat = df["EssaySet"].astype(str) + "_" + df["Score1"].astype(str)
    df["fold"] = 0
    for f, (_, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(df, strat)):
        df.loc[df.index[va], "fold"] = f

questions = {str(q): {"allowed_grades": [0, 1, 2, 3]} for q in sorted(df.EssaySet.unique())}
scores = []
for f in range(5):
    tr, va = df[df.fold != f], df[df.fold == f]
    with tempfile.TemporaryDirectory() as d:
        joblib.dump(T.build(tr), os.path.join(d, "approach1.joblib"), compress=3)
        model = approach1.load_model(d, {}, {})
        rows = [{"id": str(r.Id), "question": str(r.EssaySet), "text": str(r.EssayText)} for r in va.itertuples()]
        pred = {p["id"]: p["score"] for p in model.predict(rows, questions)}
    va = va.assign(pred=[pred[str(i)] for i in va.Id])
    per_q = [cohen_kappa_score(g.Score1, g.pred, weights="quadratic") for _, g in va.groupby("EssaySet")]
    scores.append(np.mean(per_q))
    print(f"fold {f}: per-question QWK {scores[-1]:.4f}", flush=True)
print(f"\nmean {np.mean(scores):.4f}  std {np.std(scores):.4f}")