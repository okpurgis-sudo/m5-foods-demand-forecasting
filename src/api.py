from pathlib import Path
import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, render_template_string


app = Flask(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.environ.get("MODEL_DIR", PROJECT_ROOT / "models"))

MODEL_FILES = {
    "top": "xgboost_log1p_top.joblib",
    "middle": "xgboost_log1p_middle.joblib",
    "bottom": "xgboost_log1p_bottom.joblib",
}

models = {}


HTML_PAGE = """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <title>M5 FOODS Demand Forecasting API</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #f5f7fa;
            margin: 0;
            padding: 40px;
            color: #222;
        }
        .container {
            max-width: 920px;
            margin: auto;
            background: white;
            padding: 32px;
            border-radius: 14px;
            box-shadow: 0 8px 30px rgba(0,0,0,0.08);
        }
        h1 {
            margin-top: 0;
            font-size: 28px;
        }
        .description {
            line-height: 1.7;
            color: #555;
            margin-bottom: 24px;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 16px;
        }
        label {
            font-weight: bold;
            font-size: 13px;
            display: block;
            margin-bottom: 6px;
        }
        input, select {
            width: 100%;
            box-sizing: border-box;
            padding: 10px;
            border: 1px solid #ccd3dd;
            border-radius: 8px;
            font-size: 14px;
        }
        .checkbox-area {
            display: flex;
            gap: 20px;
            margin-top: 18px;
            margin-bottom: 24px;
        }
        .checkbox-area label {
            font-weight: normal;
        }
        .checkbox-area input {
            width: auto;
            margin-right: 6px;
        }
        button {
            background: #1f6feb;
            color: white;
            border: none;
            padding: 13px 24px;
            border-radius: 8px;
            font-size: 15px;
            cursor: pointer;
        }
        button:hover {
            background: #195cc5;
        }
        .result {
            margin-top: 28px;
            padding: 20px;
            background: #f0f6ff;
            border-radius: 10px;
            border-left: 5px solid #1f6feb;
            white-space: pre-wrap;
            font-family: Consolas, monospace;
        }
        .links {
            margin-top: 24px;
            font-size: 14px;
        }
        .links a {
            color: #1f6feb;
            margin-right: 16px;
        }
        .note {
            margin-top: 18px;
            color: #666;
            font-size: 13px;
            line-height: 1.6;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>M5 FOODS Demand Forecasting Demo</h1>

        <div class="description">
            Kaggle M5 Forecasting Accuracy の FOODSカテゴリを対象にした需要予測APIです。<br>
            Linux環境で学習した XGBoost log1p モデルをEC2上のFlask APIで読み込み、入力された特徴量から販売数を予測します。
        </div>

        <div class="grid">
            <div>
                <label>Sales group</label>
                <select id="sales_group">
                    <option value="top">top</option>
                    <option value="middle">middle</option>
                    <option value="bottom">bottom</option>
                </select>
            </div>

            <div>
                <label>lag_7</label>
                <input id="lag_7" type="number" step="0.1" value="10">
            </div>

            <div>
                <label>lag_14</label>
                <input id="lag_14" type="number" step="0.1" value="9">
            </div>

            <div>
                <label>lag_21</label>
                <input id="lag_21" type="number" step="0.1" value="8">
            </div>

            <div>
                <label>lag_28</label>
                <input id="lag_28" type="number" step="0.1" value="11">
            </div>

            <div>
                <label>rolling_mean_7</label>
                <input id="rolling_mean_7" type="number" step="0.1" value="10.5">
            </div>

            <div>
                <label>rolling_mean_14</label>
                <input id="rolling_mean_14" type="number" step="0.1" value="9.8">
            </div>

            <div>
                <label>rolling_mean_28</label>
                <input id="rolling_mean_28" type="number" step="0.1" value="9.2">
            </div>

            <div>
                <label>sell_price</label>
                <input id="sell_price" type="number" step="0.1" value="3.5">
            </div>

            <div>
                <label>dow_ratio_x_rolling_mean_7</label>
                <input id="dow_ratio_x_rolling_mean_7" type="number" step="0.1" value="12.0">
            </div>
        </div>

        <div class="checkbox-area">
            <label><input id="is_weekend" type="checkbox" checked> is_weekend</label>
            <label><input id="snap" type="checkbox"> snap</label>
            <label><input id="has_event" type="checkbox"> has_event</label>
        </div>

        <button onclick="predict()">Predict</button>

        <div id="result" class="result">Prediction result will appear here.</div>

        <div class="links">
            <a href="/health" target="_blank">/health</a>
            <a href="/features/top" target="_blank">/features/top</a>
            <a href="/features/middle" target="_blank">/features/middle</a>
            <a href="/features/bottom" target="_blank">/features/bottom</a>
        </div>

        <div class="note">
            Note: This is a portfolio demo. Missing engineered features are filled with 0.
            The full feature engineering and training pipeline is available in the notebook and Linux execution scripts.
        </div>
    </div>

    <script>
        function getNumber(id) {
            return Number(document.getElementById(id).value);
        }

        async function predict() {
            const payload = {
                sales_group: document.getElementById("sales_group").value,
                features: {
                    lag_7: getNumber("lag_7"),
                    lag_14: getNumber("lag_14"),
                    lag_21: getNumber("lag_21"),
                    lag_28: getNumber("lag_28"),
                    rolling_mean_7: getNumber("rolling_mean_7"),
                    rolling_mean_14: getNumber("rolling_mean_14"),
                    rolling_mean_28: getNumber("rolling_mean_28"),
                    sell_price: getNumber("sell_price"),
                    is_weekend: document.getElementById("is_weekend").checked ? 1 : 0,
                    snap: document.getElementById("snap").checked ? 1 : 0,
                    has_event: document.getElementById("has_event").checked ? 1 : 0,
                    dow_ratio_x_rolling_mean_7: getNumber("dow_ratio_x_rolling_mean_7")
                }
            };

            const resultArea = document.getElementById("result");
            resultArea.textContent = "Predicting...";

            try {
                const response = await fetch("/predict", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify(payload)
                });

                const data = await response.json();
                resultArea.textContent = JSON.stringify(data, null, 2);
            } catch (error) {
                resultArea.textContent = "Error: " + error;
            }
        }
    </script>
</body>
</html>
"""


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


@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_PAGE)


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
        "predicted_sales": round(pred_sales, 3),
        "missing_feature_count": len(missing_features),
        "note": "Missing features were filled with 0. For production use, generate all features from the same feature pipeline."
    })


if __name__ == "__main__":
    load_models()
    app.run(host="0.0.0.0", port=5000, debug=False)
