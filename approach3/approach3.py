"""Approach 3: Open Approach -- NBC with TF-IDF.
"""

from pathlib import Path
import joblib #saves and reloads trained models


#---------------------------------
# LOAD THE SAVED MODELS
#---------------------------------
class Approach3Model:
    def __init__(self, weights_dir):
        self.weights_dir = Path(weights_dir)
        self.models = {}

        #find and load every saved question model
        for model_file in self.weights_dir.glob("model_question_*.joblib"):
            question_id = model_file.stem.split("_")[-1]
            self.models[question_id] = joblib.load(model_file)
            print(f"Loaded model for question {question_id}")

        if not self.models:
            raise FileNotFoundError("No models found in weights_dir. Please train models first.")

    #PREDICT SCORES
    def predict(self, test_rows, questions): #test_rows are the student answers needed to be graded, and questions is the metadata about the questions, including allowed grades
        predictions = []
        for row in test_rows:
            question_id = str(row["question"])
            answer_text = str(row.get("text", "")) 

            #select the model for this question
            model = self.models.get(question_id)

            if model is None:
                raise ValueError(f"No model found for question {question_id}")

            #Predict the score from the answer text
            score = int(model.predict([answer_text])[0]) #answer_text is a list of one item, since scikit-learn expects a collection of examples; [0] gets the first (and only) prediction

            #Ensure the score is valid for this question
            allowed = questions[question_id]["allowed_grades"]

            if score not in allowed:
                score = int(min(
                    allowed,
                    key=lambda x: abs(x - score)
                ))
            predictions.append({
                "id": row["id"],
                "score": score,
            })  

        return predictions
def load_model(weights_dir, config, assets):
    #The grader calls this once, then calls predict() once on the evaluation set.
    return Approach3Model(weights_dir)
