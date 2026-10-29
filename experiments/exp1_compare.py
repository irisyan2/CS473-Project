"""Experiment 1: compare the three approaches on the SAME 5 folds with the SAME metric.

    python experiments/exp1_compare.py --csv training_dataset_updated.csv [--only 1 2 3]

For each fold, each approach is trained on the other 4 folds with its own training code and
predicts the held-out fold through its submitted predict() interface.
  Approach 1: approach1/train_approach1.py + approach1/approach1.py
  Approach 2: approach2/train_approach2.py + approach2/approach2.py, with the settings in
              approach2/weights_updated/approach2_config.json. Text vectors come from the stored
              MiniLM+BGE embeddings in approach2_knn.npz (same models; no download needed).
  Approach 3: the older TF-IDF + Ridge version in experiments/approach3_old/ (the one used
              in the submission), with the parameters saved in its submitted bundle.
Writes experiments/results/exp1_summary.csv and exp1_per_question.csv. Takes ~10 minutes.
"""
import argparse
import json
import tempfile
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from common import (EXP, OUT, ROOT, A2_CFG, add_path, fmt, load_data, question_info, score,
                    shared_folds, stored_embeddings, use_stored_embeddings)

warnings.filterwarnings("ignore")


def rows_for(df, idx):
    return [{"id": int(df.Id.iat[i]), "question": str(df.EssaySet.iat[i]), "text": str(df.EssayText.iat[i])}
            for i in idx]


def cv_approach1(df, fold, questions):
    add_path(ROOT / "approach1")
    import approach1 as A
    from train_approach1 import build
    pred = np.full(len(df), np.nan)
    for f in range(5):
        tr, va = np.where(fold != f)[0], np.where(fold == f)[0]
        with tempfile.TemporaryDirectory() as d:
            joblib.dump(build(df.iloc[tr]), Path(d) / "approach1.joblib")
            out = A.load_model(d, {}, {}).predict(rows_for(df, va), questions)
        pred[va] = [o["score"] for o in out]
        print(f"  approach 1 fold {f} done", flush=True)
    return pred


def cv_approach2(df, fold, questions):
    add_path(ROOT / "approach2")
    import approach2 as A
    import train_approach2 as T
    E, look = stored_embeddings(df)
    use_stored_embeddings(A, E, look)
    cfg = json.load(open(A2_CFG))
    dfa = df.copy()
    dfa["EssaySet"] = dfa["EssaySet"].astype(int)
    pred = np.full(len(df), np.nan)
    for f in range(5):
        tr, va = np.where(fold != f)[0], np.where(fold == f)[0]
        with tempfile.TemporaryDirectory() as d:
            T.fit_artifacts(dfa.iloc[tr].reset_index(drop=True), E[tr], cfg, d)
            out = A.load_model(d, {}, None).predict(rows_for(df, va), questions)
        pred[va] = [o["score"] for o in out]
        print(f"  approach 2 fold {f} done", flush=True)
    return pred


def cv_approach3(df, fold, questions, csv):
    add_path(EXP / "approach3_old")
    import approach3 as A
    import train_approach3 as T
    params = joblib.load(ROOT / "submission" / "weights" / "approach3_bundle.joblib")["params"]
    d3 = T.load_data(csv)
    ranges, allowed_by_q = T.get_score_info(d3, None)
    lo = d3.question.map(lambda q: ranges[q][0]).to_numpy(float)
    hi = d3.question.map(lambda q: ranges[q][1]).to_numpy(float)
    y = d3.score.to_numpy(int)
    z = np.divide(y - lo, hi - lo, out=np.zeros_like(lo), where=(hi - lo) != 0)
    sizes = d3.question.value_counts()
    small_q = set(sizes[sizes < T.SMALL_N].index)
    pred = np.full(len(df), np.nan)
    for f in range(5):
        tr, va = np.where(fold != f)[0], np.where(fold == f)[0]
        bundle = T.fit_final(d3.iloc[tr].reset_index(drop=True), z[tr], y[tr], params, ranges, allowed_by_q, small_q)
        with tempfile.TemporaryDirectory() as d:
            joblib.dump(bundle, Path(d) / "approach3_bundle.joblib")
            out = A.load_model(d).predict(rows_for(df, va), questions)
        pred[va] = [o["score"] for o in out]
        print(f"  approach 3 fold {f} done", flush=True)
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "training_dataset_updated.csv"))
    ap.add_argument("--only", nargs="+", type=int, default=[1, 2, 3])
    args = ap.parse_args()
    df = load_data(args.csv)
    fold = shared_folds(df)
    _, questions = question_info(df)
    runs = {1: lambda: cv_approach1(df, fold, questions), 2: lambda: cv_approach2(df, fold, questions),
            3: lambda: cv_approach3(df, fold, questions, args.csv)}
    names = {1: "Approach 1 (KNN, TF-IDF)", 2: "Approach 2 (KNN, embeddings)", 3: "Approach 3 (TF-IDF + Ridge)"}
    summary, per_q = [], {}
    for a in args.only:
        print(f"{names[a]}:", flush=True)
        pred = runs[a]()
        s, per = score(df, pred, fold)
        print(f"  {fmt(s)}", flush=True)
        summary.append(dict(approach=names[a], **s))
        per_q[names[a]] = per
        pd.DataFrame({"Id": df.Id, "pred": pred}).to_csv(OUT / f"exp1_pred_approach{a}.csv", index=False)
    pd.DataFrame(summary).round(4).to_csv(OUT / "exp1_summary.csv", index=False)
    pd.DataFrame(per_q).round(4).to_csv(OUT / "exp1_per_question.csv", index_label="EssaySet")
    print("\nsaved experiments/results/exp1_summary.csv and exp1_per_question.csv")


if __name__ == "__main__":
    main()
