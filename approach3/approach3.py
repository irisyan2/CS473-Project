"""approach3.py -- TF-IDF + Ridge with centroid similarity features."""

from pathlib import Path

import joblib
import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize

BUNDLE = "approach3_bundle.joblib"


def _clean(text):
    return "" if text is None else str(text).lower().strip()


class Approach3Model:
    def __init__(self, weights_dir):
        candidates = [
            Path(weights_dir) / BUNDLE,
            Path(__file__).resolve().parent / "weights" / BUNDLE,
        ]

        bundle_path = next((p for p in candidates if p.exists()), None)
        if bundle_path is None:
            raise FileNotFoundError(
                f"Could not find {BUNDLE}. Checked: {candidates}"
            )

        bundle = joblib.load(bundle_path)

        self.wv = bundle["word_vec"]
        self.cv = bundle["char_vec"]
        self.per_q = bundle["per_q"]
        self.glob = bundle["glob"]
        self.gcen = bundle["global_center"]
        self.params = bundle["params"]

    def _centroid_similarities(self, Xw, centroids):
        Xw = normalize(Xw, norm="l2")
        return np.asarray(Xw @ centroids.T)

    def _predict_normalized(self, X, Xw, question_id):
        entry = self.per_q.get(str(question_id))

        # Global model predicts residuals from each question's mean.
        global_residual = self.glob.predict(X)

        if entry is None:
            return self.gcen + global_residual

        center = entry["center"]
        centroids = entry["centroids"]

        similarities = self._centroid_similarities(Xw, centroids)
        X_question = sparse.hstack(
            [X, sparse.csr_matrix(similarities)]
        ).tocsr()

        local_prediction = entry["ridge"].predict(X_question)

        n = entry["n"]
        m = self.params["m"]
        stretch = self.params["s"]
        offset = self.params["b"]

        # Shrink local predictions toward the question mean and
        # the global residual prediction.
        weight = n / (n + m) if m > 0 else 1.0
        combined = (
            weight * local_prediction
            + (1 - weight) * (center + global_residual)
        )

        return center + stretch * (combined - center) + offset

    def predict(self, test_rows, questions):
        texts = [_clean(row.get("text", "")) for row in test_rows]
        question_ids = [str(row["question"]) for row in test_rows]

        Xw = self.wv.transform(texts).tocsr()
        Xc = self.cv.transform(texts).tocsr()
        X = sparse.hstack([Xw, Xc]).tocsr()

        predictions = [None] * len(test_rows)

        for question_id in dict.fromkeys(question_ids):
            indices = [
                i for i, qid in enumerate(question_ids)
                if qid == question_id
            ]

            if question_id not in questions:
                raise KeyError(
                    f"Missing question information for {question_id}"
                )

            allowed = np.array(sorted(
                float(g)
                for g in questions[question_id]["allowed_grades"]
            ))

            low, high = allowed[0], allowed[-1]

            if high == low:
                snapped = np.full(len(indices), low)
            else:
                normalized_prediction = self._predict_normalized(
                    X[indices], Xw[indices], question_id
                )
                raw = low + normalized_prediction * (high - low)
                raw = np.nan_to_num(
                    raw,
                    nan=(low + high) / 2,
                    posinf=high,
                    neginf=low,
                )
                snapped = allowed[
                    np.abs(raw[:, None] - allowed[None, :]).argmin(axis=1)
                ]

            for i, grade in zip(indices, snapped):
                predictions[i] = (
                    int(grade) if float(grade).is_integer()
                    else float(grade)
                )

        return [
            {"id": test_rows[i]["id"], "score": predictions[i]}
            for i in range(len(test_rows))
        ]


def load_model(weights_dir, config=None, assets=None):
    return Approach3Model(weights_dir)