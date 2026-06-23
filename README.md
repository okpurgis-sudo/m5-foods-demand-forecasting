# M5 FOODS Demand Forecasting

Kaggle の **M5 Forecasting Accuracy** データセットを用いて、FOODS カテゴリ商品の需要予測を行う機械学習プロジェクトです。

Notebook 上で分析するだけではなく、Linux 上で再現可能なパイプライン化、学習済みモデルの保存、AWS S3 への成果物保存、EC2 上での Flask API / HTML デモ画面による予測も実装しました。

## プロジェクト概要

本プロジェクトでは、食品商品の過去販売数・価格・曜日・イベント・SNAP などの情報をもとに、将来の販売数を予測する回帰モデルを構築しました。

M5 データは、商品・店舗・日付ごとの販売数を持つ時系列データです。
食品カテゴリでは、よく売れる商品とほとんど売れない商品で販売パターンが大きく異なるため、商品を販売規模ごとに分類して分析しました。

本プロジェクトでは、商品ごとの累計販売数に基づいて以下の 3 グループを作成しています。

* `top`: よく売れる商品
* `middle`: 平均的な商品
* `bottom`: あまり売れない商品

各グループごとにモデルを学習・評価し、販売規模による予測難易度の違いも確認しました。

## 使用技術

* Python
* pandas
* NumPy
* scikit-learn
* XGBoost
* Flask
* joblib
* Jupyter Notebook
* Linux / Bash
* Git / GitHub
* AWS S3
* AWS EC2

## プロジェクト構成

```text
m5-foods-demand-forecasting/
├── notebooks/
│   └── m5_foods_demand_forecasting.ipynb
├── src/
│   ├── run_pipeline.py
│   └── api.py
├── scripts/
│   ├── run_linux.sh
│   ├── run_api.sh
│   └── upload_results_to_s3.sh
├── results/
├── images/
├── models/
├── requirements.txt
├── .gitignore
└── README.md
```

## データソース

使用データは Kaggle の **M5 Forecasting Accuracy** データセットです。

主に以下のデータを使用しています。

| ファイル                         | 内容                         |
| ---------------------------- | -------------------------- |
| `sales_train_validation.csv` | 商品・店舗ごとの日次販売数              |
| `calendar.csv`               | 日付、曜日、イベント、SNAP などのカレンダー情報 |
| `sell_prices.csv`            | 商品・店舗・週ごとの販売価格             |

本プロジェクトでは、`sales_train_validation.csv` から FOODS カテゴリの商品を対象にし、販売規模別に top / middle / bottom のグループを作成しました。

## 処理の流れ

全体の処理は以下の流れです。

```text
1. Kaggle M5 データの読み込み
2. FOODS カテゴリ商品の抽出
3. 商品ごとの累計販売数を計算
4. top / middle / bottom グループへ分類
5. 横持ちの日次販売データを縦持ちデータへ変換
6. calendar / sell_prices と結合
7. lag / rolling / 価格 / イベント / SNAP 特徴量を作成
8. 時系列分割で train / test を作成
9. XGBoost モデルを学習
10. 特徴量セットを変えながら Ablation Study
11. log1p 目的変数変換モデルを作成
12. 最終モデルを joblib で保存
13. Flask API から予測できる形に展開
14. HTML 画面から予測を実行
15. 成果物を S3 に保存し、EC2 上でも動作確認
```

## 特徴量設計

需要予測では、過去の販売傾向・曜日・価格・イベントなどが重要になります。
本プロジェクトでは、以下の特徴量を作成しました。

### lag 特徴量

過去の同一商品の販売数を特徴量として使用します。

| 特徴量      | 意味       |
| -------- | -------- |
| `lag_7`  | 7日前の販売数  |
| `lag_14` | 14日前の販売数 |
| `lag_21` | 21日前の販売数 |
| `lag_28` | 28日前の販売数 |

食品販売には曜日周期や月次に近い周期があるため、1週間前・2週間前・4週間前の販売数を参照しています。

### rolling 特徴量

直近数日間の平均販売数を特徴量として使用します。

| 特徴量               | 意味           |
| ----------------- | ------------ |
| `rolling_mean_7`  | 直近7日間の平均販売数  |
| `rolling_mean_14` | 直近14日間の平均販売数 |
| `rolling_mean_21` | 直近21日間の平均販売数 |
| `rolling_mean_28` | 直近28日間の平均販売数 |
| `rolling_sum_28`  | 直近28日間の合計販売数 |

一日単位の販売数はブレが大きいため、移動平均を使うことで最近の売れ行きを安定して表現します。

rolling 特徴量では、予測対象日の販売数が混入しないように `shift(1)` を使っています。

```python
model_df["rolling_mean_7"] = model_df.groupby("id")["sales"].transform(
    lambda x: x.shift(1).rolling(7).mean()
)
```

これにより、予測時点より未来の情報を使わないようにしています。

### zero rate 特徴量

販売数が少ない商品では、「売れなかった日がどれくらいあるか」が重要になります。

| 特徴量                 | 意味                |
| ------------------- | ----------------- |
| `sold_last_28_flag` | 過去28日間に1回でも売れたか   |
| `zero_rate_28`      | 過去28日間で販売数が0だった割合 |

特に bottom グループでは 0 販売が多いため、zero rate は販売規模の小さい商品の予測に有効な特徴量です。

### 曜日・カレンダー特徴量

| 特徴量                      | 意味          |
| ------------------------ | ----------- |
| `wday_1` ～ `wday_7`      | 曜日の one-hot |
| `month`                  | 月           |
| `year`                   | 年           |
| `week_of_month`          | 月内の週        |
| `is_weekend`             | 週末フラグ       |
| `has_event` / `is_event` | イベント日フラグ    |
| `snap`                   | SNAP対象日フラグ  |

食品需要は曜日やイベントの影響を受けるため、カレンダー情報を特徴量として追加しています。

### 価格特徴量

| 特徴量                | 意味         |
| ------------------ | ---------- |
| `sell_price`       | 販売価格       |
| `price_vs_mean`    | 平均価格との差    |
| `price_ratio`      | 平均価格に対する比率 |
| `price_pct_change` | 価格変化率      |

価格の変動は需要に影響するため、単純な価格だけでなく、平均価格との差や変化率も特徴量として使用しました。

### カテゴリ・店舗特徴量

| 特徴量       | 意味   |
| --------- | ---- |
| `cat_*`   | カテゴリ |
| `dept_*`  | 部門   |
| `store_*` | 店舗   |
| `state_*` | 州    |

商品や店舗によって販売傾向が異なるため、カテゴリ・部門・店舗・州を one-hot 化して特徴量に加えました。

## モデル選定

本プロジェクトでは、最終的なモデルとして **XGBoost Regressor** を使用しました。

### XGBoost を選んだ理由

需要予測には ARIMA や LSTM などの時系列モデルもありますが、本プロジェクトでは以下の理由から XGBoost を採用しました。

### 1. 特徴量エンジニアリングとの相性が良い

M5 データは、単一の時系列ではなく、商品・店舗・日付・価格・イベントなど複数の情報を持つテーブルデータです。

XGBoost は、lag / rolling / price / event / category などの特徴量を組み合わせた表形式データに強く、今回のような特徴量設計ベースの需要予測に適しています。

### 2. 非線形な関係を扱える

需要は、価格・曜日・イベント・過去販売数などが複雑に絡み合って決まります。

線形回帰では表現しにくい非線形な関係も、XGBoost であれば木構造によって扱いやすくなります。

### 3. 学習が比較的速く、ポートフォリオとして再現しやすい

LSTM などの深層学習モデルは、計算コストが高く、学習環境への依存も大きくなります。

XGBoost は比較的軽量に学習でき、Linux 環境や EC2 上での再現・推論にも向いています。

### 4. 特徴量重要度を確認できる

XGBoost は feature importance を確認できるため、どの特徴量が予測に効いているかを説明しやすいです。

ポートフォリオでは、単に精度を出すだけでなく、「なぜその特徴量を使ったのか」「どの特徴量が効いたのか」を説明できることが重要です。

## 比較した特徴量セット

本プロジェクトでは、特徴量を段階的に追加して性能を比較する **Ablation Study** を行いました。

| バージョン                 | 内容                                |
| --------------------- | --------------------------------- |
| `v1_base`             | lag / rolling / calendar などの基本特徴量 |
| `v2_zero`             | v1 に zero rate 特徴量を追加             |
| `v3_dow`              | v2 に曜日効果特徴量を追加                    |
| `v4_price_event_dept` | v3 に価格・イベント・SNAP・部門情報を追加          |

最終的に、全グループで `v4_price_event_dept` が最も良い性能となりました。

これは、食品需要が単純な過去販売数だけでなく、価格・イベント・SNAP・部門情報にも影響されていることを示しています。

## 目的変数の log1p 変換

販売数データは、少数の商品が多く売れ、多くの商品は販売数が少ないという歪んだ分布を持ちます。

そのため、目的変数 `sales` に対して `log1p` 変換を適用しました。

```python
y_train_log = np.log1p(y_train)
```

予測後は `expm1` で元の販売数スケールに戻しています。

```python
preds = np.expm1(model.predict(X_test))
```

log1p 変換を使うことで、大きな販売数にモデルが引っ張られすぎることを抑え、小さい販売数の商品にも対応しやすくしました。

## 評価方法

時系列データであるため、ランダム分割ではなく、日付で train / test を分割しました。

```text
train: 2016-04-01 より前
test : 2016-04-01 以降
```

評価指標は以下を使用しました。

| 指標        | 意味                |
| --------- | ----------------- |
| MAE       | 予測誤差の絶対値平均        |
| RMSE      | 大きな誤差をより重く見る指標    |
| MAE ratio | 平均販売数に対する MAE の割合 |

販売規模が top / middle / bottom で大きく異なるため、単純な MAE だけでなく、平均販売数に対する誤差割合である MAE ratio も確認しました。

## モデル評価結果

最終的な XGBoost + log1p モデルの評価結果です。

| sales_group | model         |    MAE |   RMSE | actual_mean | MAE ratio |
| ----------- | ------------- | -----: | -----: | ----------: | --------: |
| top         | XGBoost_log1p | 5.1927 | 8.9086 |     16.1387 |    0.3218 |
| middle      | XGBoost_log1p | 0.9364 | 1.4473 |      1.1484 |    0.8154 |
| bottom      | XGBoost_log1p | 0.3982 | 0.7594 |      0.3398 |    1.1719 |

### 結果の解釈

`top` グループでは平均販売数が大きいため、モデルが販売パターンを比較的学習しやすく、MAE ratio は約 0.32 となりました。

`middle` / `bottom` グループでは販売数が少なく、0 販売の日も多いため予測難易度が高くなります。
特に `bottom` グループでは平均販売数が非常に小さいため、わずかな誤差でも MAE ratio が大きくなります。

## ソースコードの説明

### `notebooks/m5_foods_demand_forecasting.ipynb`

分析・検証用の Notebook です。

主に以下を行います。

* M5 データの読み込み
* FOODS カテゴリの抽出
* 販売規模別グループ作成
* EDA
* 特徴量作成
* モデル学習
* 評価
* 特徴量重要度の確認
* Ablation Study

### `src/run_pipeline.py`

Notebook 依存から脱却し、Linux 上でも同じ処理を再現できるようにするための実行スクリプトです。

Notebook だけでは、実行環境や手作業に依存しやすいため、パイプラインとしてまとめることで再現性を高めました。

主な役割は以下です。

* プロジェクトルートの解決
* 必要な処理の実行
* 結果ファイルの出力
* モデルファイルの保存
* Linux 上での再実行をしやすくする

### `src/api.py`

Flask API と HTML デモ画面を提供するファイルです。

主なエンドポイントは以下です。

| エンドポイント                   | 内容            |
| ------------------------- | ------------- |
| `/`                       | HTML デモ画面     |
| `/health`                 | API の状態確認     |
| `/features/<sales_group>` | モデルが要求する特徴量一覧 |
| `/predict`                | 需要予測          |

API 起動時に `models/` フォルダから学習済みモデルを読み込みます。

予測時には、入力された特徴量をモデルの要求する特徴量に合わせ、不足している特徴量は 0 で補完します。

```text
画面から主要特徴量を入力
↓
API が JSON を受け取る
↓
モデルに必要な特徴量の形に変換
↓
不足特徴量を 0 で補完
↓
XGBoost モデルで予測
↓
log1p モデルの場合は expm1 で販売数に戻す
↓
JSON と HTML 画面に結果を返す
```

Prediction Demo

!["デモ画面"]("images/demo_top.png")

食品需要予測のHTMLデモ画面です。
商品タイプ、過去の販売数、直近の売れ行き、価格、週末・イベント・SNAP条件を入力し、予測販売数を確認できます。




Prediction Result

!["結果表示"]("images/demo_prediction_result.png")

入力値をもとに、Flask APIが学習済みXGBoostモデルを呼び出し、予測販売数を返します。
画面上では「約〇個」のように表示し、技術者向けにはJSONレスポンスも確認できるようにしています。

現在の HTML デモでは、ユーザーに分かりやすいように、機械学習上の特徴量名を日本語に置き換えています。

| 画面での表記      | 内部の特徴量名          |
| ----------- | ---------------- |
| 1週間前の販売数    | `lag_7`          |
| 2週間前の販売数    | `lag_14`         |
| 直近1週間の平均販売数 | `rolling_mean_7` |
| 販売価格        | `sell_price`     |
| 週末          | `is_weekend`     |
| SNAP対象日     | `snap`           |
| 祝日・イベント日    | `has_event`      |

### `scripts/run_linux.sh`

Linux 上でパイプラインを実行するためのシェルスクリプトです。

主な役割は以下です。

* プロジェクトルートへ移動
* 仮想環境の確認
* Python スクリプトの実行
* Linux 環境での再現性確保

### `scripts/run_api.sh`

Flask API を起動するためのシェルスクリプトです。

手動で以下を打つ代わりに、スクリプトとして API を起動できます。

```bash
source .venv/bin/activate
python src/api.py
```

### `scripts/upload_results_to_s3.sh`

結果ファイル・画像・モデルを AWS S3 にアップロードするためのスクリプトです。

アップロード対象は主に以下です。

* `results/`
* `images/`
* `models/`

モデルファイルは GitHub に含めず、S3 に保存する構成にしています。

## Flask API の使い方

### API 起動

```bash
cd ~/m5-foods-demand-forecasting
source .venv/bin/activate
python src/api.py
```

ブラウザで以下にアクセスします。

```text
http://127.0.0.1:5000/
```

Lubuntu など別PCからアクセスする場合は、Lubuntu の IP アドレスを使います。

```text
http://<LubuntuのIP>:5000/
```

EC2 で公開する場合は、EC2 の Public IPv4 を使います。

```text
http://<EC2のPublicIPv4>:5000/
```

### health check

```bash
curl http://127.0.0.1:5000/health
```

レスポンス例:

```json
{
  "status": "ok",
  "loaded_models": ["top", "middle", "bottom"],
  "model_dir": "/path/to/models"
}
```

### prediction

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sales_group": "top",
    "features": {
      "lag_7": 10,
      "lag_14": 9,
      "lag_21": 8,
      "lag_28": 11,
      "rolling_mean_7": 10.5,
      "rolling_mean_14": 9.8,
      "rolling_mean_28": 9.2,
      "sell_price": 3.5,
      "is_weekend": 1,
      "snap": 0,
      "has_event": 0,
      "dow_ratio_x_rolling_mean_7": 12.0
    }
  }'
```

レスポンス例:

```json
{
  "sales_group": "top",
  "model_variant": "xgboost_log1p",
  "predicted_sales": 11.255692481994629,
  "missing_feature_count": 72,
  "note": "Missing features were filled with 0. For production use, generate all features from the same feature pipeline."
}
```

## Linux での実行方法

### 1. リポジトリを clone

```bash
git clone https://github.com/okpurgis-sudo/m5-foods-demand-forecasting.git
cd m5-foods-demand-forecasting
```

### 2. 仮想環境を作成

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. ライブラリをインストール

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. パイプライン実行

```bash
bash scripts/run_linux.sh
```

### 5. Flask API 起動

```bash
source .venv/bin/activate
python src/api.py
```

ブラウザで以下にアクセスします。

```text
http://127.0.0.1:5000/
```

## AWS 連携

本プロジェクトでは、AWS S3 と EC2 を使用しました。

### S3 の役割

S3 は、学習済みモデルや評価結果を保存するために使用しています。

GitHub には大きなモデルファイルや生データを含めず、S3 に保存することで、コードと成果物を分離しています。

```text
GitHub: ソースコード管理
S3    : モデル・結果ファイル保存
EC2   : API 実行環境
```

### EC2 の役割

EC2 は、Flask API を外部からアクセスできる形で起動するために使用しています。

EC2 上では以下を行います。

```text
1. GitHub からコードを取得
2. S3 から学習済みモデルを取得
3. Python 仮想環境を作成
4. Flask API を起動
5. Public IPv4 + 5000番ポートでブラウザからアクセス
```

## GitHub に含めないもの

以下はセキュリティ・容量の観点から GitHub 管理対象外にしています。

* Kaggle API key
* AWS access key
* 学習済みモデルファイル
* 生データ CSV
* 実行時に生成される一部の中間ファイル

## 工夫した点

### 1. データリーク対策

時系列データであるため、ランダム分割ではなく日付で train / test を分けました。

また、rolling 特徴量には `shift(1)` を使用し、予測対象日の販売数が特徴量に混入しないようにしました。

### 2. 販売規模別のモデル作成

販売数が多い商品と少ない商品では、需要の動き方が大きく異なります。

そのため、top / middle / bottom に分けてモデルを作成し、グループごとの予測難易度を比較しました。

### 3. Ablation Study

特徴量を段階的に追加し、どの種類の特徴量が性能改善に寄与するかを確認しました。

これにより、単に多くの特徴量を入れるのではなく、価格・イベント・SNAP・部門情報が有効であることを確認しました。

### 4. log1p 変換

販売数の分布が歪んでいるため、目的変数を log1p 変換しました。

これにより、大きな販売数の商品に引っ張られすぎることを抑え、全グループで MAE の改善を確認しました。

### 5. Notebook 依存からの脱却

Notebook だけで完結させず、Linux 上で再実行できるスクリプトを用意しました。

これにより、ローカル環境だけでなく、Lubuntu や EC2 でも再現できる構成になっています。

### 6. API / HTML デモ化

モデルを作って終わりではなく、Flask API と HTML 画面を作成しました。

これにより、非エンジニアでもブラウザから予測処理を試せる形にしています。

### 7. AWS 連携

S3 にモデルや結果を保存し、EC2 上で API を起動することで、クラウド上でも動作する構成にしました。

## 現在の制限

現在の HTML デモでは、主要な特徴量のみを画面から入力しています。
モデルが要求する全特徴量のうち、画面に表示していない詳細特徴量は 0 で補完しています。

そのため、現在のデモは「API とモデルの連携を見せるための簡易デモ」です。

実務レベルに近づけるには、商品IDと日付を入力すると、API 側で必要な特徴量を自動生成する構成にする必要があります。

## 今後の改善案

* 商品IDと日付を入力すると、API 側で特徴量を自動生成する
* 推論用データ作成処理を API に統合する
* Docker 化する
* Gunicorn / Nginx を使った本番向け構成にする
* 予測結果のグラフ表示を追加する
* 複数日先の需要予測に対応する
* GitHub Actions などで CI/CD を導入する
* モデルの再学習パイプラインを整備する

## このプロジェクトで示せること

このプロジェクトでは、以下のスキルを示すことを目的としています。

* 時系列データを扱った特徴量設計
* 機械学習モデルの比較・評価
* XGBoost を用いた回帰モデル構築
* データリークを意識した検証設計
* Linux 上での再現可能な実行環境構築
* Flask による API 化
* HTML による簡易デモ画面作成
* Git / GitHub によるコード管理
* AWS S3 / EC2 を用いたクラウド連携
* モデルを実際に利用できる形へ展開する力

## まとめ

本プロジェクトでは、Kaggle M5 データを用いた FOODS カテゴリ商品の需要予測モデルを構築しました。

販売規模別に商品を分類し、lag / rolling / zero rate / price / event / SNAP などの特徴量を作成しました。
XGBoost による Ablation Study を行い、log1p 目的変数変換によって全グループで MAE の改善を確認しました。

さらに、Linux で再現可能なパイプライン、Flask API、HTML デモ、AWS S3 / EC2 連携まで実装し、Notebook 上の分析に留まらず、モデルを実際に利用できる形へ展開しました。
