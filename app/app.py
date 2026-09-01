from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "ctr_pipeline.joblib"

app = Flask(__name__)


def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def make_features(form):
    timestamp = pd.to_datetime(form["timestamp"])
    row = {
        "userid": str(form["userid"]),
        "offerid": str(form["offerid"]),
        "countrycode": str(form["countrycode"]),
        "category": str(form["category"]),
        "merchant": str(form["merchant"]),
        "hour": timestamp.hour,
        "dayofweek": timestamp.dayofweek,
        "is_weekend": int(timestamp.dayofweek >= 5),
    }
    return pd.DataFrame([row])


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    if request.method == "POST":
        try:
            model = load_model()
            if model is None:
                raise FileNotFoundError(
                    "Model not found. Run notebook 03_model_training.ipynb first."
                )

            X = make_features(request.form)

            # The training notebook saves a dictionary containing the
            # preprocessor and LightGBM model. The old app tried to call
            # predict_proba() directly on that dictionary, which caused:
            # "dict object has no attribute predict_proba".
            if isinstance(model, dict):
                preprocessor = model["preprocessor"]
                estimator = model["model"]
                X = preprocessor.transform(X)
            else:
                # Also support a normal sklearn Pipeline saved as one object.
                estimator = model

            probability = float(estimator.predict_proba(X)[0, 1])
            result = {
                "probability": probability * 100,
                "label": "Likely to click" if probability >= 0.5 else "Less likely to click",
            }
        except Exception as exc:
            error = str(exc)

    return render_template("index.html", result=result, error=error)


if __name__ == "__main__":
    app.run(debug=True)
