from pathlib import Path
import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request


app = Flask(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.environ.get("MODEL_DIR", PROJECT_ROOT / "models"))

MODEL_FILES = {
    "top": "xgboost_log1p_top.joblib",
    "middle": "xgboost_log1p_middle.joblib",
    "bottom": "xgboost_log1p_bottom.joblib",
}

models = {}


def load_models():
    for group, filename in MODEL_FILES.items():
        model_path = MODEL_DIR / filename

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        models[group] = joblib.load(model_path)

    print("Loaded models:", list(models.keys()))


def get_feature_names(model):
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    if hasattr(model, "get_booster"):
        feature_names = model.get_booster().feature_names
        if feature_names is not None:
            return list(feature_names)

    raise ValueError("Could not detect feature names from model.")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "loaded_models": list(models.keys()),
        "model_dir": str(MODEL_DIR)
    })


@app.route("/features/<sales_group>", methods=["GET"])
def features(sales_group):
    if sales_group not in models:
        return jsonify({
            "error": "Invalid sales_group",
            "allowed_sales_groups": list(models.keys())
        }), 400

    feature_names = get_feature_names(models[sales_group])

    return jsonify({
        "sales_group": sales_group,
        "feature_count": len(feature_names),
        "features": feature_names
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if data is None:
        return jsonify({
            "error": "Request body must be JSON."
        }), 400

    sales_group = data.get("sales_group")
    input_features = data.get("features", {})

    if sales_group not in models:
        return jsonify({
            "error": "Invalid sales_group",
            "allowed_sales_groups": list(models.keys())
        }), 400

    if not isinstance(input_features, dict):
        return jsonify({
            "error": "features must be a JSON object."
        }), 400

    model = models[sales_group]
    feature_names = get_feature_names(model)

    row = {}
    missing_features = []

    for feature in feature_names:
        if feature in input_features:
            row[feature] = input_features[feature]
        else:
            row[feature] = 0
            missing_features.append(feature)

    X = pd.DataFrame([row], columns=feature_names)

    pred_log = model.predict(X)[0]
    pred_sales = float(np.expm1(pred_log))
    pred_sales = max(0.0, pred_sales)

    return jsonify({
        "sales_group": sales_group,
        "model_variant": "xgboost_log1p",
        "predicted_sales": pred_sales,
        "missing_feature_count": len(missing_features),
        "note": "Missing features were filled with 0. For production use, generate all features from the same feature pipeline."
    })


if __name__ == "__main__":
    load_models()
    app.run(host="0.0.0.0", port=5000, debug=False)
