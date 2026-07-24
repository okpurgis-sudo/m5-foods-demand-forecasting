"""保存済みXGBoostモデルを呼び出すポートフォリオ用Flask API。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template_string, request
import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
GROUPS = ("top", "middle", "bottom")

app = Flask(__name__)

GROUP_META = {
    "top": {
        "label": "よく売れる商品",
        "description": "販売量が多く、比較的安定した需要を持つ商品群",
        "badge": "高需要",
    },
    "middle": {
        "label": "平均的な商品",
        "description": "販売量が中程度の商品群",
        "badge": "中需要",
    },
    "bottom": {
        "label": "あまり売れない商品",
        "description": "販売数0の日も多く、需要変動が大きい商品群",
        "badge": "低需要",
    },
}

FEATURE_FIELDS = [
    {
        "name": "lag_7",
        "label": "1週間前の販売数",
        "description": "7日前に販売された個数",
        "section": "過去の販売数",
        "step": "1",
        "min": "0",
        "default": "10",
    },
    {
        "name": "lag_14",
        "label": "2週間前の販売数",
        "description": "14日前に販売された個数",
        "section": "過去の販売数",
        "step": "1",
        "min": "0",
        "default": "9",
    },
    {
        "name": "lag_28",
        "label": "4週間前の販売数",
        "description": "28日前に販売された個数",
        "section": "過去の販売数",
        "step": "1",
        "min": "0",
        "default": "8",
    },
    {
        "name": "rolling_mean_7",
        "label": "直近1週間の平均販売数",
        "description": "直近7日間の販売数の平均",
        "section": "最近の販売傾向",
        "step": "0.01",
        "min": "0",
        "default": "9.5",
    },
    {
        "name": "rolling_mean_28",
        "label": "直近4週間の平均販売数",
        "description": "直近28日間の販売数の平均",
        "section": "最近の販売傾向",
        "step": "0.01",
        "min": "0",
        "default": "8.8",
    },
    {
        "name": "sell_price",
        "label": "販売価格",
        "description": "M5データ上の商品価格（例：3.50）",
        "section": "価格・カレンダー条件",
        "step": "0.01",
        "min": "0",
        "default": "3.5",
    },
    {
        "name": "price_change_1",
        "label": "直近の価格変化率",
        "description": "前週からの価格変化率。変化なしは0",
        "section": "価格・カレンダー条件",
        "step": "0.01",
        "default": "0",
    },
    {
        "name": "has_event",
        "label": "祝日・イベント日",
        "description": "予測対象日にイベントがある場合はオン",
        "section": "価格・カレンダー条件",
        "type": "checkbox",
        "default": "0",
    },
    {
        "name": "is_snap_day",
        "label": "SNAP対象日",
        "description": "米国の食品購買支援制度の対象日はオン",
        "section": "価格・カレンダー条件",
        "type": "checkbox",
        "default": "0",
    },
]

SECTION_DESCRIPTIONS = {
    "過去の販売数": "同じ商品の周期的な売れ方をモデルへ伝える入力です。",
    "最近の販売傾向": "一時的なばらつきを抑え、直近の需要水準を表します。",
    "価格・カレンダー条件": "価格変更やイベントなど、需要を動かす外部条件です。",
}

EXAMPLE_PRESETS = {
    "top": {
        "lag_7": 18,
        "lag_14": 16,
        "lag_28": 17,
        "rolling_mean_7": 17.5,
        "rolling_mean_28": 16.8,
        "sell_price": 3.5,
        "price_change_1": 0,
        "has_event": 0,
        "is_snap_day": 1,
    },
    "middle": {
        "lag_7": 2,
        "lag_14": 1,
        "lag_28": 2,
        "rolling_mean_7": 1.7,
        "rolling_mean_28": 1.5,
        "sell_price": 4.2,
        "price_change_1": 0,
        "has_event": 1,
        "is_snap_day": 0,
    },
    "bottom": {
        "lag_7": 0,
        "lag_14": 1,
        "lag_28": 0,
        "rolling_mean_7": 0.3,
        "rolling_mean_28": 0.4,
        "sell_price": 2.8,
        "price_change_1": -0.05,
        "has_event": 0,
        "is_snap_day": 0,
    },
}

DEFAULT_FORM = {
    "sales_group": "top",
    **{field["name"]: field.get("default", "0") for field in FEATURE_FIELDS},
}

HTML = """
<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="Kaggle M5データで学習したXGBoostモデルによる食品需要予測デモ">
  <title>M5 食品需要予測デモ</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f3f6f8;
      --surface: #ffffff;
      --surface-soft: #f8fafb;
      --text: #17212b;
      --muted: #637180;
      --border: #dce3e8;
      --primary: #176b5b;
      --primary-dark: #0f5145;
      --primary-soft: #e5f3ef;
      --accent: #d98b2b;
      --danger: #a83a3a;
      --danger-soft: #fff0f0;
      --shadow: 0 16px 42px rgba(31, 48, 61, 0.10);
      --radius-lg: 22px;
      --radius-md: 14px;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      background:
        radial-gradient(circle at 12% 4%, rgba(23, 107, 91, 0.10), transparent 30rem),
        radial-gradient(circle at 90% 10%, rgba(217, 139, 43, 0.10), transparent 28rem),
        var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans JP", sans-serif;
      line-height: 1.65;
    }

    button, input { font: inherit; }

    a { color: var(--primary); }

    .page-shell {
      width: min(1180px, calc(100% - 32px));
      margin: 0 auto;
      padding: 36px 0 56px;
    }

    .hero {
      position: relative;
      overflow: hidden;
      padding: clamp(28px, 5vw, 54px);
      border-radius: 28px;
      background: linear-gradient(135deg, #123d36 0%, #176b5b 58%, #24816f 100%);
      color: #fff;
      box-shadow: var(--shadow);
    }

    .hero::after {
      content: "";
      position: absolute;
      width: 300px;
      height: 300px;
      right: -100px;
      top: -150px;
      border: 1px solid rgba(255, 255, 255, .18);
      border-radius: 50%;
      box-shadow: 0 0 0 54px rgba(255, 255, 255, .05), 0 0 0 108px rgba(255, 255, 255, .03);
    }

    .hero-content { position: relative; z-index: 1; max-width: 790px; }

    .eyebrow {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 14px;
      padding: 6px 11px;
      border: 1px solid rgba(255,255,255,.24);
      border-radius: 999px;
      background: rgba(255,255,255,.10);
      font-size: .84rem;
      font-weight: 700;
      letter-spacing: .04em;
    }

    h1 {
      margin: 0;
      font-size: clamp(2rem, 5vw, 3.7rem);
      line-height: 1.16;
      letter-spacing: -.035em;
    }

    .hero p {
      margin: 18px 0 0;
      color: rgba(255,255,255,.86);
      font-size: clamp(.98rem, 2vw, 1.12rem);
    }

    .hero-tags {
      display: flex;
      flex-wrap: wrap;
      gap: 9px;
      margin-top: 24px;
    }

    .hero-tag {
      padding: 6px 10px;
      border-radius: 8px;
      background: rgba(0,0,0,.16);
      font-size: .84rem;
      font-weight: 650;
    }

    .status-line {
      display: flex;
      align-items: center;
      gap: 9px;
      margin-top: 18px;
      color: rgba(255,255,255,.78);
      font-size: .88rem;
    }

    .status-dot {
      width: 9px;
      height: 9px;
      border-radius: 50%;
      background: #ffd36b;
      box-shadow: 0 0 0 4px rgba(255,211,107,.15);
    }

    .status-dot.online { background: #72e0a9; box-shadow: 0 0 0 4px rgba(114,224,169,.16); }
    .status-dot.offline { background: #ff9292; box-shadow: 0 0 0 4px rgba(255,146,146,.16); }

    .process-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 14px;
      margin: 22px 0;
    }

    .process-card {
      display: flex;
      gap: 12px;
      align-items: flex-start;
      padding: 18px;
      border: 1px solid var(--border);
      border-radius: var(--radius-md);
      background: rgba(255,255,255,.76);
    }

    .process-number {
      display: grid;
      flex: 0 0 34px;
      width: 34px;
      height: 34px;
      place-items: center;
      border-radius: 10px;
      background: var(--primary-soft);
      color: var(--primary);
      font-weight: 800;
    }

    .process-card strong { display: block; margin-bottom: 2px; }
    .process-card span { color: var(--muted); font-size: .9rem; }

    .layout {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 340px;
      gap: 22px;
      align-items: start;
    }

    .panel {
      border: 1px solid var(--border);
      border-radius: var(--radius-lg);
      background: var(--surface);
      box-shadow: 0 10px 30px rgba(31,48,61,.06);
    }

    .panel-header {
      padding: 24px 26px 18px;
      border-bottom: 1px solid var(--border);
    }

    .panel-header h2 { margin: 0; font-size: 1.35rem; }
    .panel-header p { margin: 5px 0 0; color: var(--muted); font-size: .92rem; }

    .panel-body { padding: 26px; }

    .form-section + .form-section {
      margin-top: 30px;
      padding-top: 28px;
      border-top: 1px solid var(--border);
    }

    .section-title {
      margin: 0 0 4px;
      font-size: 1.05rem;
    }

    .section-description {
      margin: 0 0 16px;
      color: var(--muted);
      font-size: .9rem;
    }

    .group-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 10px;
    }

    .group-option { position: relative; }
    .group-option input { position: absolute; opacity: 0; pointer-events: none; }

    .group-card {
      display: block;
      min-height: 126px;
      padding: 15px;
      border: 2px solid var(--border);
      border-radius: 14px;
      background: var(--surface-soft);
      cursor: pointer;
      transition: border-color .18s ease, transform .18s ease, background .18s ease;
    }

    .group-card:hover { transform: translateY(-2px); border-color: #9abdb5; }
    .group-option input:checked + .group-card {
      border-color: var(--primary);
      background: var(--primary-soft);
      box-shadow: inset 0 0 0 1px var(--primary);
    }

    .group-badge {
      display: inline-block;
      margin-bottom: 8px;
      padding: 3px 7px;
      border-radius: 999px;
      background: #fff;
      color: var(--primary);
      font-size: .72rem;
      font-weight: 800;
    }

    .group-card strong { display: block; font-size: .96rem; }
    .group-card small { display: block; margin-top: 5px; color: var(--muted); line-height: 1.45; }

    .field-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 13px;
    }

    .field {
      display: block;
      padding: 14px;
      border: 1px solid var(--border);
      border-radius: 13px;
      background: var(--surface-soft);
    }

    .field-label-row {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: baseline;
      margin-bottom: 7px;
    }

    .field-label { font-size: .92rem; font-weight: 750; }

    .feature-code {
      color: var(--muted);
      font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
      font-size: .72rem;
    }

    .field-description {
      display: block;
      min-height: 2.6em;
      margin-bottom: 9px;
      color: var(--muted);
      font-size: .78rem;
      line-height: 1.45;
    }

    input[type="number"] {
      width: 100%;
      padding: 11px 12px;
      border: 1px solid #cbd5dc;
      border-radius: 9px;
      background: #fff;
      color: var(--text);
      outline: none;
      transition: border-color .16s ease, box-shadow .16s ease;
    }

    input[type="number"]:focus {
      border-color: var(--primary);
      box-shadow: 0 0 0 3px rgba(23,107,91,.12);
    }

    .toggle-field {
      display: flex;
      align-items: center;
      justify-content: space-between;
      min-height: 94px;
    }

    .toggle-copy { padding-right: 12px; }
    .toggle-copy .field-description { min-height: 0; margin: 4px 0 0; }

    .switch { position: relative; flex: 0 0 auto; width: 48px; height: 27px; }
    .switch input { position: absolute; opacity: 0; }
    .switch-track {
      position: absolute;
      inset: 0;
      border-radius: 999px;
      background: #cbd5dc;
      cursor: pointer;
      transition: background .18s ease;
    }
    .switch-track::after {
      content: "";
      position: absolute;
      width: 21px;
      height: 21px;
      left: 3px;
      top: 3px;
      border-radius: 50%;
      background: #fff;
      box-shadow: 0 2px 6px rgba(0,0,0,.18);
      transition: transform .18s ease;
    }
    .switch input:checked + .switch-track { background: var(--primary); }
    .switch input:checked + .switch-track::after { transform: translateX(21px); }

    .actions {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 26px;
    }

    .primary-button,
    .secondary-button,
    .preset-button {
      border: 0;
      border-radius: 11px;
      cursor: pointer;
      font-weight: 750;
      transition: transform .15s ease, background .15s ease, box-shadow .15s ease;
    }

    .primary-button {
      flex: 1 1 240px;
      padding: 13px 22px;
      background: var(--primary);
      color: #fff;
      box-shadow: 0 8px 18px rgba(23,107,91,.20);
    }
    .primary-button:hover { background: var(--primary-dark); transform: translateY(-1px); }

    .secondary-button {
      padding: 13px 17px;
      border: 1px solid var(--border);
      background: #fff;
      color: var(--text);
    }
    .secondary-button:hover { background: var(--surface-soft); }

    .preset-row {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 14px;
    }

    .preset-label { width: 100%; color: var(--muted); font-size: .8rem; }
    .preset-button {
      padding: 7px 10px;
      border: 1px solid var(--border);
      background: var(--surface-soft);
      color: var(--primary-dark);
      font-size: .8rem;
    }
    .preset-button:hover { border-color: #9abdb5; background: var(--primary-soft); }

    .sidebar { display: grid; gap: 16px; position: sticky; top: 18px; }

    .result-card {
      overflow: hidden;
      border: 1px solid #b8d7cf;
      border-radius: var(--radius-lg);
      background: linear-gradient(145deg, #f0faf7, #ffffff);
      box-shadow: 0 12px 32px rgba(23,107,91,.10);
    }

    .result-card.empty { border-color: var(--border); background: var(--surface); box-shadow: none; }

    .result-top {
      padding: 22px 22px 0;
      color: var(--muted);
      font-size: .84rem;
      font-weight: 700;
    }

    .result-main { padding: 10px 22px 24px; }
    .result-value {
      display: flex;
      align-items: baseline;
      gap: 7px;
      color: var(--primary-dark);
    }
    .result-value strong { font-size: clamp(2.5rem, 6vw, 4.1rem); line-height: 1; letter-spacing: -.05em; }
    .result-value span { font-size: 1.05rem; font-weight: 700; }
    .result-group { margin-top: 12px; color: var(--muted); font-size: .9rem; }

    .empty-result {
      padding: 26px 22px;
      color: var(--muted);
      text-align: center;
    }
    .empty-icon {
      display: grid;
      width: 52px;
      height: 52px;
      margin: 0 auto 12px;
      place-items: center;
      border-radius: 15px;
      background: var(--primary-soft);
      color: var(--primary);
      font-size: 1.35rem;
      font-weight: 900;
    }

    .error-card {
      padding: 16px;
      border: 1px solid #efb8b8;
      border-radius: var(--radius-md);
      background: var(--danger-soft);
      color: var(--danger);
      font-size: .9rem;
    }

    .info-card { padding: 20px; }
    .info-card h3 { margin: 0 0 9px; font-size: 1rem; }
    .info-card p, .info-card li { color: var(--muted); font-size: .86rem; }
    .info-card ul { margin: 0; padding-left: 18px; }

    .endpoint-links {
      display: grid;
      gap: 8px;
      margin-top: 13px;
    }
    .endpoint-links a {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      padding: 9px 10px;
      border: 1px solid var(--border);
      border-radius: 9px;
      background: var(--surface-soft);
      text-decoration: none;
      font-size: .8rem;
      font-weight: 700;
    }
    .endpoint-links code { color: var(--muted); }

    .footer {
      margin-top: 24px;
      color: var(--muted);
      text-align: center;
      font-size: .82rem;
    }

    @media (max-width: 900px) {
      .layout { grid-template-columns: 1fr; }
      .sidebar { position: static; grid-template-columns: repeat(2, minmax(0,1fr)); }
    }

    @media (max-width: 700px) {
      .page-shell { width: min(100% - 20px, 1180px); padding-top: 10px; }
      .hero { border-radius: 20px; }
      .process-grid, .group-grid, .field-grid, .sidebar { grid-template-columns: 1fr; }
      .process-card { padding: 14px; }
      .panel-header, .panel-body { padding-left: 18px; padding-right: 18px; }
      .group-card { min-height: auto; }
      .field-description { min-height: 0; }
    }
  </style>
</head>
<body>
  <main class="page-shell">
    <header class="hero">
      <div class="hero-content">
        <div class="eyebrow">PORTFOLIO / MACHINE LEARNING</div>
        <h1>M5 食品需要予測デモ</h1>
        <p>
          過去の販売数、直近の需要、価格、イベント条件を入力し、
          Kaggle M5データで学習したXGBoostモデルから予測販売数を算出します。
        </p>
        <div class="hero-tags">
          <span class="hero-tag">XGBoost</span>
          <span class="hero-tag">Flask API</span>
          <span class="hero-tag">Time Series</span>
          <span class="hero-tag">log1p Target</span>
        </div>
        <div class="status-line">
          <span id="status-dot" class="status-dot"></span>
          <span id="status-text">API状態を確認しています</span>
        </div>
      </div>
    </header>

    <section class="process-grid" aria-label="予測の流れ">
      <div class="process-card">
        <div class="process-number">1</div>
        <div><strong>商品群を選択</strong><span>販売規模に対応するモデルを選びます。</span></div>
      </div>
      <div class="process-card">
        <div class="process-number">2</div>
        <div><strong>販売条件を入力</strong><span>過去実績、平均販売数、価格などを入力します。</span></div>
      </div>
      <div class="process-card">
        <div class="process-number">3</div>
        <div><strong>予測結果を確認</strong><span>推定された販売数を画面右側へ表示します。</span></div>
      </div>
    </section>

    <div class="layout">
      <section class="panel">
        <div class="panel-header">
          <h2>需要予測フォーム</h2>
          <p>入力例を利用して、商品群ごとの予測結果をすぐに比較できます。</p>
        </div>

        <form method="post" id="forecast-form" class="panel-body">
          <section class="form-section">
            <h3 class="section-title">商品グループ</h3>
            <p class="section-description">評価開始前の販売履歴から、販売規模別に固定した商品群です。</p>
            <div class="group-grid">
              {% for group in groups %}
                <div class="group-option">
                  <input
                    type="radio"
                    id="group-{{ group }}"
                    name="sales_group"
                    value="{{ group }}"
                    {% if form_data.get('sales_group', 'top') == group %}checked{% endif %}
                  >
                  <label class="group-card" for="group-{{ group }}">
                    <span class="group-badge">{{ group_meta[group].badge }}</span>
                    <strong>{{ group_meta[group].label }}</strong>
                    <small>{{ group_meta[group].description }}</small>
                  </label>
                </div>
              {% endfor %}
            </div>
          </section>

          {% for section_name, section_fields in sections.items() %}
            <section class="form-section">
              <h3 class="section-title">{{ section_name }}</h3>
              <p class="section-description">{{ section_descriptions[section_name] }}</p>
              <div class="field-grid">
                {% for field in section_fields %}
                  {% if field.get('type') == 'checkbox' %}
                    <div class="field toggle-field">
                      <div class="toggle-copy">
                        <div class="field-label-row">
                          <span class="field-label">{{ field.label }}</span>
                          <span class="feature-code">{{ field.name }}</span>
                        </div>
                        <span class="field-description">{{ field.description }}</span>
                      </div>
                      <label class="switch" aria-label="{{ field.label }}">
                        <input
                          type="checkbox"
                          name="{{ field.name }}"
                          value="1"
                          {% if form_data.get(field.name, '0')|string == '1' %}checked{% endif %}
                        >
                        <span class="switch-track"></span>
                      </label>
                    </div>
                  {% else %}
                    <label class="field">
                      <span class="field-label-row">
                        <span class="field-label">{{ field.label }}</span>
                        <span class="feature-code">{{ field.name }}</span>
                      </span>
                      <span class="field-description">{{ field.description }}</span>
                      <input
                        type="number"
                        name="{{ field.name }}"
                        step="{{ field.get('step', 'any') }}"
                        {% if field.get('min') is not none %}min="{{ field.min }}"{% endif %}
                        value="{{ form_data.get(field.name, field.get('default', '0')) }}"
                        inputmode="decimal"
                        required
                      >
                    </label>
                  {% endif %}
                {% endfor %}
              </div>
            </section>
          {% endfor %}

          <div class="preset-row">
            <span class="preset-label">サンプル値を入力</span>
            <button type="button" class="preset-button" data-preset="top">よく売れる商品の例</button>
            <button type="button" class="preset-button" data-preset="middle">平均的な商品の例</button>
            <button type="button" class="preset-button" data-preset="bottom">あまり売れない商品の例</button>
          </div>

          <div class="actions">
            <button type="submit" class="primary-button">予測を実行する</button>
            <button type="reset" class="secondary-button">入力をリセット</button>
          </div>
        </form>
      </section>

      <aside class="sidebar">
        {% if prediction is not none %}
          <section class="result-card" aria-live="polite">
            <div class="result-top">予測販売数</div>
            <div class="result-main">
              <div class="result-value">
                <strong>{{ '%.2f'|format(prediction) }}</strong><span>個</span>
              </div>
              <div class="result-group">
                {{ group_meta[selected_group].label }}のモデルで算出
              </div>
            </div>
          </section>
        {% else %}
          <section class="result-card empty">
            <div class="empty-result">
              <div class="empty-icon">予</div>
              <strong>ここに予測結果が表示されます</strong>
              <p>フォームを入力して「予測を実行する」を押してください。</p>
            </div>
          </section>
        {% endif %}

        {% if error %}
          <div class="error-card" role="alert">
            <strong>予測できませんでした</strong><br>
            {{ error }}
          </div>
        {% endif %}

        <section class="panel info-card">
          <h3>このデモについて</h3>
          <ul>
            <li>学習済みXGBoostモデルを使用</li>
            <li>商品群ごとに別モデルを読み込み</li>
            <li>予測値はlog1pから元スケールへ復元</li>
          </ul>
          <p>
            画面にない特徴量は0で補完する、モデルとAPIの連携確認用デモです。
            実務では商品IDと日付から全特徴量を自動生成します。
          </p>
        </section>

        <section class="panel info-card">
          <h3>API確認</h3>
          <div class="endpoint-links">
            <a href="/health" target="_blank" rel="noopener"><span>状態確認</span><code>/health</code></a>
            <a href="/features/top" target="_blank" rel="noopener"><span>高需要モデル</span><code>/features/top</code></a>
            <a href="/features/middle" target="_blank" rel="noopener"><span>中需要モデル</span><code>/features/middle</code></a>
            <a href="/features/bottom" target="_blank" rel="noopener"><span>低需要モデル</span><code>/features/bottom</code></a>
          </div>
        </section>
      </aside>
    </div>

    <footer class="footer">
      M5 Forecasting Accuracy / Portfolio Demo
    </footer>
  </main>

  <script>
    const presets = {{ presets | tojson }};

    document.querySelectorAll('[data-preset]').forEach((button) => {
      button.addEventListener('click', () => {
        const group = button.dataset.preset;
        const values = presets[group];
        const groupInput = document.querySelector(`input[name="sales_group"][value="${group}"]`);
        if (groupInput) groupInput.checked = true;

        Object.entries(values).forEach(([name, value]) => {
          const input = document.querySelector(`[name="${name}"]`);
          if (!input) return;
          if (input.type === 'checkbox') {
            input.checked = Number(value) === 1;
          } else {
            input.value = value;
          }
        });
      });
    });

    fetch('/health')
      .then((response) => {
        if (!response.ok) throw new Error('health check failed');
        return response.json();
      })
      .then((data) => {
        const models = data.models || {};
        const readyCount = Object.values(models).filter(Boolean).length;
        const dot = document.getElementById('status-dot');
        const text = document.getElementById('status-text');
        if (readyCount === 3) {
          dot.classList.add('online');
          text.textContent = 'API稼働中・3モデル読み込み可能';
        } else {
          dot.classList.add('offline');
          text.textContent = `API稼働中・モデル ${readyCount}/3`; 
        }
      })
      .catch(() => {
        document.getElementById('status-dot').classList.add('offline');
        document.getElementById('status-text').textContent = 'API状態を取得できませんでした';
      });
  </script>
</body>
</html>
"""


def grouped_fields() -> dict[str, list[dict[str, str]]]:
    """画面表示用に特徴量をセクション単位へまとめる。"""
    sections: dict[str, list[dict[str, str]]] = {}
    for field in FEATURE_FIELDS:
        sections.setdefault(field["section"], []).append(field)
    return sections


def model_path(group: str) -> Path:
    if group not in GROUPS:
        raise ValueError(f"sales_group must be one of {GROUPS}")
    return MODELS_DIR / f"xgboost_log1p_{group}.joblib"


@lru_cache(maxsize=3)
def load_model(group: str) -> Any:
    path = model_path(group)
    if not path.is_file():
        raise FileNotFoundError(
            f"Model file was not found: {path}. Run the notebook to generate models."
        )
    return joblib.load(path)


def feature_names(model: Any) -> list[str]:
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        return [str(name) for name in names]
    booster = getattr(model, "get_booster", lambda: None)()
    names = getattr(booster, "feature_names", None)
    if names:
        return [str(name) for name in names]
    raise ValueError("The model does not expose feature names.")


def parse_number(value: Any, name: str) -> float:
    if value in (None, ""):
        return 0.0
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def make_frame(model: Any, payload: dict[str, Any]) -> pd.DataFrame:
    names = feature_names(model)
    values = {name: parse_number(payload.get(name, 0.0), name) for name in names}
    return pd.DataFrame([values], columns=names)


def predict(group: str, payload: dict[str, Any]) -> float:
    model = load_model(group)
    frame = make_frame(model, payload)
    log_prediction = float(model.predict(frame)[0])
    return max(0.0, float(np.expm1(log_prediction)))


@app.get("/health")
def health() -> Any:
    available = {group: model_path(group).is_file() for group in GROUPS}
    return jsonify({"status": "ok", "models": available})


@app.get("/features/<group>")
def features(group: str) -> Any:
    try:
        model = load_model(group)
        return jsonify({"sales_group": group, "features": feature_names(model)})
    except (ValueError, FileNotFoundError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/predict")
def predict_endpoint() -> Any:
    payload = request.get_json(silent=True) or request.form.to_dict()
    group = str(payload.get("sales_group", "top"))
    try:
        value = predict(group, payload)
        return jsonify({"sales_group": group, "prediction": value})
    except (ValueError, FileNotFoundError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/", methods=["GET", "POST"])
def index() -> str:
    prediction_value = None
    error = None
    form_data = dict(DEFAULT_FORM)
    selected_group = "top"

    if request.method == "POST":
        form_data.update(request.form.to_dict())
        selected_group = request.form.get("sales_group", "top")
        form_data["sales_group"] = selected_group
        try:
            prediction_value = predict(selected_group, request.form.to_dict())
        except (ValueError, FileNotFoundError) as exc:
            error = str(exc)
    else:
        selected_group = form_data["sales_group"]

    return render_template_string(
        HTML,
        groups=GROUPS,
        group_meta=GROUP_META,
        sections=grouped_fields(),
        section_descriptions=SECTION_DESCRIPTIONS,
        presets=EXAMPLE_PRESETS,
        form_data=form_data,
        selected_group=selected_group,
        prediction=prediction_value,
        error=error,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
