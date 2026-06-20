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
    <title>食品需要予測デモ</title>
    <style>
        body {
            font-family: Arial, "Hiragino Sans", "Yu Gothic", sans-serif;
            background: #f4f7fb;
            margin: 0;
            color: #1f2937;
        }

        .hero {
            background: linear-gradient(135deg, #155eef, #6d28d9);
            color: white;
            padding: 38px 24px;
        }

        .hero-inner {
            max-width: 1050px;
            margin: auto;
        }

        .hero h1 {
            margin: 0 0 12px 0;
            font-size: 32px;
        }

        .hero p {
            margin: 0;
            line-height: 1.8;
            font-size: 15px;
            opacity: 0.96;
        }

        .container {
            max-width: 1050px;
            margin: 28px auto;
            padding: 0 20px 48px 20px;
        }

        .card {
            background: white;
            border-radius: 14px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 8px 26px rgba(15, 23, 42, 0.08);
        }

        h2 {
            margin-top: 0;
            font-size: 22px;
        }

        h3 {
            margin-top: 0;
            font-size: 17px;
            color: #334155;
        }

        .lead {
            line-height: 1.8;
            color: #475569;
        }

        .step-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 14px;
        }

        .step-card {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 16px;
            line-height: 1.7;
        }

        .step-card strong {
            color: #155eef;
        }

        .form-section {
            margin-top: 24px;
            padding-top: 20px;
            border-top: 1px solid #e5e7eb;
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 16px;
        }

        label {
            font-weight: bold;
            font-size: 14px;
            display: block;
            margin-bottom: 6px;
        }

        .tech-name {
            font-size: 11px;
            color: #64748b;
            font-weight: normal;
            margin-left: 4px;
        }

        .hint {
            font-size: 12px;
            color: #64748b;
            line-height: 1.5;
            margin-top: 5px;
        }

        input, select {
            width: 100%;
            box-sizing: border-box;
            padding: 10px;
            border: 1px solid #cbd5e1;
            border-radius: 8px;
            font-size: 14px;
            background: white;
        }

        .checkbox-area {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 14px;
            margin-top: 14px;
        }

        .check-card {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 12px;
            line-height: 1.6;
        }

        .check-card label {
            font-weight: bold;
            margin: 0 0 4px 0;
        }

        .check-card input {
            width: auto;
            margin-right: 6px;
        }

        .button-row {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            margin-top: 24px;
        }

        button {
            background: #155eef;
            color: white;
            border: none;
            padding: 13px 22px;
            border-radius: 9px;
            font-size: 15px;
            cursor: pointer;
            font-weight: bold;
        }

        button.secondary {
            background: #475569;
        }

        button:hover {
            opacity: 0.92;
        }

        .result-main {
            margin-top: 22px;
            padding: 22px;
            background: #eff6ff;
            border-radius: 12px;
            border-left: 6px solid #155eef;
        }

        .result-number {
            font-size: 38px;
            font-weight: bold;
            color: #155eef;
            margin: 10px 0;
        }

        .result-sub {
            color: #475569;
            line-height: 1.8;
        }

        .json-box {
            margin-top: 16px;
            padding: 16px;
            background: #0f172a;
            color: #e2e8f0;
            border-radius: 10px;
            white-space: pre-wrap;
            font-family: Consolas, monospace;
            font-size: 13px;
            overflow-x: auto;
        }

        details {
            margin-top: 16px;
        }

        summary {
            cursor: pointer;
            font-weight: bold;
            color: #155eef;
        }

        .dictionary {
            width: 100%;
            border-collapse: collapse;
            margin-top: 14px;
            font-size: 14px;
        }

        .dictionary th,
        .dictionary td {
            border: 1px solid #e2e8f0;
            padding: 10px;
            text-align: left;
            vertical-align: top;
        }

        .dictionary th {
            background: #f1f5f9;
        }

        .badge {
            display: inline-block;
            padding: 4px 9px;
            border-radius: 999px;
            background: #dbeafe;
            color: #1d4ed8;
            font-size: 12px;
            font-weight: bold;
            margin-right: 6px;
        }

        .warning {
            background: #fff7ed;
            border-left: 6px solid #f97316;
        }

        .links a {
            color: #155eef;
            margin-right: 16px;
            text-decoration: none;
            font-weight: bold;
        }

        @media (max-width: 850px) {
            .grid, .step-grid, .checkbox-area {
                grid-template-columns: 1fr;
            }

            .hero h1 {
                font-size: 25px;
            }
        }
    </style>
</head>
<body>
    <div class="hero">
        <div class="hero-inner">
            <h1>食品の販売数を予測するデモ</h1>
            <p>
                この画面は、過去の販売数・最近の売れ行き・価格・曜日やイベント情報をもとに、食品商品の需要を予測するデモです。<br>
                学習済みの機械学習モデルをAWS EC2上のFlask APIで動かし、ブラウザから予測できるようにしています。
            </p>
        </div>
    </div>

    <div class="container">

        <div class="card">
            <h2>このデモの見方</h2>
            <p class="lead">
                難しい特徴量名を知らなくても使えるように、画面上では日本語の説明にしています。
                内部ではKaggle M5データで作成した特徴量を使い、XGBoostモデルが予測を行います。
            </p>

            <div class="step-grid">
                <div class="step-card">
                    <strong>1. 商品タイプを選ぶ</strong><br>
                    よく売れる商品、平均的な商品、あまり売れない商品から選びます。
                </div>
                <div class="step-card">
                    <strong>2. 販売状況を入力</strong><br>
                    1週間前の販売数や、直近の平均販売数を入力します。
                </div>
                <div class="step-card">
                    <strong>3. 予測結果を見る</strong><br>
                    モデルが予測した販売数を、画面に大きく表示します。
                </div>
            </div>
        </div>

        <div class="card">
            <h2>需要予測フォーム</h2>

            <div class="form-section">
                <h3>① 商品タイプ</h3>
                <div class="grid">
                    <div>
                        <label>商品タイプ</label>
                        <select id="sales_group">
                            <option value="top">よく売れる商品</option>
                            <option value="middle">平均的な商品</option>
                            <option value="bottom">あまり売れない商品</option>
                        </select>
                        <div class="hint">
                            商品を累計販売数で3つに分けています。売れ方によって効く特徴量が変わるためです。
                        </div>
                    </div>
                </div>
            </div>

            <div class="form-section">
                <h3>② 過去の販売数</h3>
                <p class="lead">
                    同じ商品の過去の販売数です。食品需要は周期性があるため、1週間前・2週間前などの売れ方が重要になります。
                </p>

                <div class="grid">
                    <div>
                        <label>1週間前の販売数 <span class="tech-name">lag_7</span></label>
                        <input id="lag_7" type="number" step="0.1" value="10">
                        <div class="hint">例：1週間前に10個売れたなら 10</div>
                    </div>

                    <div>
                        <label>2週間前の販売数 <span class="tech-name">lag_14</span></label>
                        <input id="lag_14" type="number" step="0.1" value="9">
                        <div class="hint">例：2週間前に9個売れたなら 9</div>
                    </div>

                    <div>
                        <label>3週間前の販売数 <span class="tech-name">lag_21</span></label>
                        <input id="lag_21" type="number" step="0.1" value="8">
                        <div class="hint">長めの周期を見るために使います。</div>
                    </div>

                    <div>
                        <label>4週間前の販売数 <span class="tech-name">lag_28</span></label>
                        <input id="lag_28" type="number" step="0.1" value="11">
                        <div class="hint">約1か月前の販売数です。</div>
                    </div>
                </div>
            </div>

            <div class="form-section">
                <h3>③ 最近の売れ行き</h3>
                <p class="lead">
                    一日だけの販売数ではなく、直近数日間の平均を見ることで、一時的なブレを減らします。
                </p>

                <div class="grid">
                    <div>
                        <label>直近1週間の平均販売数 <span class="tech-name">rolling_mean_7</span></label>
                        <input id="rolling_mean_7" type="number" step="0.1" value="10.5">
                        <div class="hint">最近よく売れているかを見る指標です。</div>
                    </div>

                    <div>
                        <label>直近2週間の平均販売数 <span class="tech-name">rolling_mean_14</span></label>
                        <input id="rolling_mean_14" type="number" step="0.1" value="9.8">
                        <div class="hint">1週間平均より少し安定した傾向を見ます。</div>
                    </div>

                    <div>
                        <label>直近4週間の平均販売数 <span class="tech-name">rolling_mean_28</span></label>
                        <input id="rolling_mean_28" type="number" step="0.1" value="9.2">
                        <div class="hint">長めの販売傾向を見る指標です。</div>
                    </div>

                    <div>
                        <label>曜日を考慮した最近の売れ行き <span class="tech-name">dow_ratio_x_rolling_mean_7</span></label>
                        <input id="dow_ratio_x_rolling_mean_7" type="number" step="0.1" value="12.0">
                        <div class="hint">
                            「週末に売れやすい」などの曜日効果と、最近の売れ行きを組み合わせた指標です。
                        </div>
                    </div>
                </div>
            </div>

            <div class="form-section">
                <h3>④ 価格・カレンダー条件</h3>
                <p class="lead">
                    価格、週末、イベント日などは食品の販売数に影響するため、補助情報として使います。
                </p>

                <div class="grid">
                    <div>
                        <label>販売価格 <span class="tech-name">sell_price</span></label>
                        <input id="sell_price" type="number" step="0.1" value="3.5">
                        <div class="hint">商品の価格です。例：3.5ドルなら 3.5</div>
                    </div>
                </div>

                <div class="checkbox-area">
                    <div class="check-card">
                        <label><input id="is_weekend" type="checkbox" checked> 週末</label>
                        <div class="hint">土日など、平日とは買われ方が変わりやすい日です。</div>
                    </div>

                    <div class="check-card">
                        <label><input id="snap" type="checkbox"> SNAP対象日</label>
                        <div class="hint">米国の食品購買支援制度の対象日です。食品カテゴリでは重要になる場合があります。</div>
                    </div>

                    <div class="check-card">
                        <label><input id="has_event" type="checkbox"> 祝日・イベント日</label>
                        <div class="hint">スポーツイベントや祝日など、需要が変わる可能性がある日です。</div>
                    </div>
                </div>
            </div>

            <div class="button-row">
                <button onclick="predict()">予測する</button>
                <button class="secondary" onclick="setExample('top')">よく売れる商品の例</button>
                <button class="secondary" onclick="setExample('middle')">平均的な商品の例</button>
                <button class="secondary" onclick="setExample('bottom')">あまり売れない商品の例</button>
            </div>

            <div id="result" class="result-main">
                <strong>予測結果</strong>
                <div class="result-sub">
                    「予測する」ボタンを押すと、ここに予測販売数が表示されます。
                </div>
            </div>

            <details>
                <summary>技術者向け：APIのJSONレスポンスを見る</summary>
                <div id="jsonResult" class="json-box">API response JSON will appear here.</div>
            </details>
        </div>

        <div class="card warning">
            <h2>確認してほしいポイント</h2>
            <ul>
                <li><span class="badge">機械学習</span> 学習済みXGBoostモデルで予測している</li>
                <li><span class="badge">Linux</span> Lubuntuで学習・モデル生成済み</li>
                <li><span class="badge">AWS</span> S3に保存したモデルをEC2側で利用している</li>
                <li><span class="badge">API</span> Flask APIとして `/predict` を公開している</li>
                <li><span class="badge">画面</span> HTMLフォームから予測できる</li>
            </ul>

            <p class="lead">
                この画面はポートフォリオ用の軽量デモです。
                実務では、商品IDと日付を入力すると、API側で必要な特徴量を自動生成する構成に発展できます。
            </p>
        </div>

        <div class="card">
            <h2>特徴量名の対応表</h2>
            <p class="lead">
                画面では分かりやすい日本語名にしていますが、内部では以下の特徴量名を使っています。
            </p>

            <table class="dictionary">
                <tr>
                    <th>画面での表記</th>
                    <th>内部の特徴量名</th>
                    <th>意味</th>
                </tr>
                <tr>
                    <td>1週間前の販売数</td>
                    <td>lag_7</td>
                    <td>7日前に売れた個数</td>
                </tr>
                <tr>
                    <td>2週間前の販売数</td>
                    <td>lag_14</td>
                    <td>14日前に売れた個数</td>
                </tr>
                <tr>
                    <td>直近1週間の平均販売数</td>
                    <td>rolling_mean_7</td>
                    <td>最近7日間の平均販売数</td>
                </tr>
                <tr>
                    <td>曜日を考慮した最近の売れ行き</td>
                    <td>dow_ratio_x_rolling_mean_7</td>
                    <td>曜日による売れやすさと、最近の売れ行きを組み合わせた特徴量</td>
                </tr>
                <tr>
                    <td>販売価格</td>
                    <td>sell_price</td>
                    <td>商品の販売価格</td>
                </tr>
            </table>
        </div>

        <div class="card">
            <h2>API確認用リンク</h2>
            <div class="links">
                <a href="/health" target="_blank">API状態確認</a>
                <a href="/features/top" target="_blank">よく売れる商品の特徴量一覧</a>
                <a href="/features/middle" target="_blank">平均的な商品の特徴量一覧</a>
                <a href="/features/bottom" target="_blank">あまり売れない商品の特徴量一覧</a>
            </div>
        </div>

    </div>

    <script>
        function getNumber(id) {
            return Number(document.getElementById(id).value);
        }

        function setValue(id, value) {
            document.getElementById(id).value = value;
        }

        function setCheck(id, value) {
            document.getElementById(id).checked = value;
        }

        function setExample(type) {
            document.getElementById("sales_group").value = type;

            if (type === "top") {
                setValue("lag_7", 10);
                setValue("lag_14", 9);
                setValue("lag_21", 8);
                setValue("lag_28", 11);
                setValue("rolling_mean_7", 10.5);
                setValue("rolling_mean_14", 9.8);
                setValue("rolling_mean_28", 9.2);
                setValue("sell_price", 3.5);
                setValue("dow_ratio_x_rolling_mean_7", 12.0);
                setCheck("is_weekend", true);
                setCheck("snap", false);
                setCheck("has_event", false);
            }

            if (type === "middle") {
                setValue("lag_7", 2);
                setValue("lag_14", 1);
                setValue("lag_21", 1);
                setValue("lag_28", 2);
                setValue("rolling_mean_7", 1.8);
                setValue("rolling_mean_14", 1.6);
                setValue("rolling_mean_28", 1.5);
                setValue("sell_price", 2.8);
                setValue("dow_ratio_x_rolling_mean_7", 2.0);
                setCheck("is_weekend", false);
                setCheck("snap", true);
                setCheck("has_event", false);
            }

            if (type === "bottom") {
                setValue("lag_7", 0);
                setValue("lag_14", 1);
                setValue("lag_21", 0);
                setValue("lag_28", 0);
                setValue("rolling_mean_7", 0.4);
                setValue("rolling_mean_14", 0.3);
                setValue("rolling_mean_28", 0.2);
                setValue("sell_price", 1.9);
                setValue("dow_ratio_x_rolling_mean_7", 0.3);
                setCheck("is_weekend", false);
                setCheck("snap", false);
                setCheck("has_event", true);
            }
        }

        function makeInterpretation(predictedSales, salesGroup) {
            let groupText = {
                "top": "よく売れる商品",
                "middle": "平均的な商品",
                "bottom": "あまり売れない商品"
            }[salesGroup];

            let level = "少なめ";
            if (predictedSales >= 10) {
                level = "多め";
            } else if (predictedSales >= 2) {
                level = "中くらい";
            }

            return `${groupText}として、次の需要は${level}と推定されます。`;
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
            const jsonArea = document.getElementById("jsonResult");

            resultArea.innerHTML = "<strong>予測中...</strong>";
            jsonArea.textContent = "Waiting for API response...";

            try {
                const response = await fetch("/predict", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify(payload)
                });

                const data = await response.json();

                if (!response.ok) {
                    resultArea.innerHTML = "<strong>エラーが発生しました</strong><div class='result-sub'>入力内容またはAPIの状態を確認してください。</div>";
                    jsonArea.textContent = JSON.stringify(data, null, 2);
                    return;
                }

                const predicted = data.predicted_sales;
                const interpretation = makeInterpretation(predicted, data.sales_group);

                resultArea.innerHTML = `
                    <strong>予測結果</strong>
                    <div class="result-number">約 ${predicted} 個</div>
                    <div class="result-sub">
                        この条件では、次に売れる数量は <strong>約 ${predicted} 個</strong> と予測されました。<br>
                        商品タイプ：${document.getElementById("sales_group").selectedOptions[0].text}<br>
                        使用モデル：${data.model_variant}<br>
                        解釈：${interpretation}<br>
                        補足：このデモでは、画面に出していない詳細特徴量 ${data.missing_feature_count} 個は0で補完しています。
                    </div>
                `;

                jsonArea.textContent = JSON.stringify(data, null, 2);
            } catch (error) {
                resultArea.innerHTML = "<strong>通信エラー</strong><div class='result-sub'>EC2のAPIが起動しているか、Security Groupの5000番ポートが開いているか確認してください。</div>";
                jsonArea.textContent = String(error);
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
