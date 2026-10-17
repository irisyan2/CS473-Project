from pathlib import Path
import joblib #saves and reloads trained models
import pandas as pd


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline #combines text preprocessing and model into one object

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score



#---------------------------------
# TRAIN AND SAVE THE MODELS
#---------------------------------

def create_model(stop_words="english", ngram_range=(1, 1), alpha=1.0, min_df=1, sublinear_tf=False):
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words=stop_words,
            ngram_range=ngram_range,
            min_df=min_df
        )),
        ("classifier", MultinomialNB(alpha=alpha))
    ])

def train_models(train_csv, weights_dir, best_config):

    data_path = Path(train_csv)
    weights_path = Path(weights_dir)
    weights_path.mkdir(parents=True, exist_ok=True)

    # Load training data
    df = pd.read_csv(data_path, encoding='utf-8')

    #Train a separate model for each question
    question_ids = sorted(df['EssaySet'].dropna().unique(), key=lambda x: int(x))

    for question_id in question_ids:
        #filter data for the current question
        question_data = df[df['EssaySet'] == question_id]
        question_data = question_data[['EssayText', 'Score1']].dropna()

        X = question_data['EssayText'].astype(str)
        y = question_data['Score1'].astype(int)

        #Convert to tf-idf vectors, then predict the score
        model = create_model(
            stop_words=best_config["stop_words"],
            ngram_range=best_config["ngram_range"],
            alpha=best_config["alpha"],
            min_df=best_config["min_df"],
            sublinear_tf=best_config["sublinear_tf"]
        )

        #Learn from answers and their known scores
        model.fit(X, y)

        #Save the model for this question
        model_filename = weights_path / f"model_question_{question_id}.joblib"
        joblib.dump(model, model_filename)
        print(f"Trained question {question_id}: {len(X)} answers")


def evaluate_config(df, config):
    fold_scores = []

    # Use the same five folds for each configuration
    splitter = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    for train_idx, valid_idx in splitter.split(
        df, df["EssaySet"]
    ):
        train_df = df.iloc[train_idx]
        valid_df = df.iloc[valid_idx]

        question_scores = []

        for question_id in valid_df["EssaySet"].unique():
            question_train = train_df[
                train_df["EssaySet"] == question_id
            ]
            question_valid = valid_df[
                valid_df["EssaySet"] == question_id
            ]

            if question_train.empty:
                continue

            model = create_model(
                stop_words=config["stop_words"],
                ngram_range=config["ngram_range"],
                alpha=config["alpha"],
                min_df=config["min_df"],
                sublinear_tf=config["sublinear_tf"]
            )

            model.fit(
                question_train["EssayText"].astype(str),
                question_train["Score1"].astype(int)
            )

            predictions = model.predict(
                question_valid["EssayText"].astype(str)
            )

            qwk = cohen_kappa_score(
                question_valid["Score1"].astype(int),
                predictions,
                weights="quadratic"
            )

            if np.isfinite(qwk):
                question_scores.append(qwk)

        if question_scores:
            fold_scores.append(np.mean(question_scores))

    if len(fold_scores) != 5:
        raise ValueError(
            "Could not calculate QWK for all five folds. "
            "Check the number and distribution of training examples."
        )

    return float(np.mean(fold_scores))


def tune_models(train_csv):
    df = pd.read_csv(train_csv, encoding="utf-8")
    df = df[["EssaySet", "EssayText", "Score1"]].dropna().copy()

    df["EssaySet"] = df["EssaySet"].astype(str)
    df["EssayText"] = df["EssayText"].astype(str)
    df["Score1"] = df["Score1"].astype(int)

    # Every question needs enough examples for five-fold CV.
    if df["EssaySet"].value_counts().min() < 5:
        raise ValueError(
            "Use the full training dataset: each question "
            "needs at least five examples for five-fold CV."
        )

    configurations = []

    for stop_words in [None, "english"]:
        for ngram_range in [(1, 1), (1, 2)]:
            for alpha in [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]:
                for min_df in [1, 2, 3]:
                    for sublinear_tf in [False, True]:
                        configurations.append({
                            "stop_words": stop_words,
                            "ngram_range": ngram_range,
                            "alpha": alpha,
                            "min_df": min_df,
                            "sublinear_tf": sublinear_tf
                        })

    best_config = None
    best_qwk = -float("inf")

    for config in configurations:
        mean_qwk = evaluate_config(df, config)

        print(f"Config: {config}")
        print(f"Mean CV QWK: {mean_qwk:.4f}")

        if mean_qwk > best_qwk:
            best_qwk = mean_qwk
            best_config = config

    print("\nBest configuration:", best_config)
    print(f"Best mean CV QWK: {best_qwk:.4f}")

    return best_config

if __name__ == "__main__":
    best_config = tune_models("training dataset.csv")
    train_models(
        train_csv="training dataset.csv",
        weights_dir="weights",
        best_config=best_config
    )