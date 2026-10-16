import re, argparse
import numpy as np, pandas as pd, joblib
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

_stem = PorterStemmer()
_tok = re.compile(r"(?u)\b\w\w+\b")
def stem_text(t):
    return " ".join(_stem.stem(w) for w in _tok.findall(str(t).lower()))

def fit_index(texts, y):
    vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 1), stop_words=None)
    X = vec.fit_transform([stem_text(t) for t in texts]).tocsr()
    return dict(vocab={k: int(v) for k, v in vec.vocabulary_.items()},
                idf=vec.idf_.astype(np.float32),
                data=X.data.astype(np.float32), indices=X.indices, indptr=X.indptr,
                shape=X.shape, y=np.asarray(y, dtype=np.int8))

def build(df, K=15):
    art = {"K": K, "per_q": {}, "pooled": fit_index(df["EssayText"], df["Score1"])}
    for q, g in df.groupby("EssaySet"):
        art["per_q"][str(q)] = fit_index(g["EssayText"], g["Score1"])
    return art

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", default="weights/approach1.joblib")
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    joblib.dump(build(df), a.out, compress=3)
    print("saved", a.out)