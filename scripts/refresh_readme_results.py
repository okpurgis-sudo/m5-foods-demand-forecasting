"""Notebookの結果CSVをREADMEの結果欄へ反映する。"""

from __future__ import annotations

from pathlib import Path
import math
import re

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
README_PATH = PROJECT_ROOT / "README.md"
RESULTS_DIR = PROJECT_ROOT / "results"


def replace_block(text: str, name: str, content: str) -> str:
    start = f"<!-- {name}_START -->"
    end = f"<!-- {name}_END -->"
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.DOTALL)
    replacement = f"{start}\n{content.strip()}\n{end}"
    updated, count = pattern.subn(replacement, text, count=1)
    if count != 1:
        raise ValueError(f"README marker was not found or duplicated: {name}")
    return updated


def format_number(value: object, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return "-"
    return f"{number:.{digits}f}"


def markdown_table(frame: pd.DataFrame, columns: list[tuple[str, str, int]]) -> str:
    headers = [label for _, label, _ in columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for column, _, digits in columns:
            value = row[column]
            values.append(format_number(value, digits) if digits >= 0 else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def feature_selection_text(feature_compare: pd.DataFrame) -> str:
    best = (
        feature_compare.sort_values(["sales_group", "mae"])
        .groupby("sales_group", observed=True, as_index=False)
        .first()
    )
    table = markdown_table(
        best,
        [
            ("sales_group", "グループ", -1),
            ("feature_version", "MAE最小の特徴量セット", -1),
            ("feature_count", "特徴量数", 0),
            ("mae", "MAE", 4),
            ("rmse", "RMSE", 4),
            ("mae_ratio", "MAE ratio", 4),
        ],
    )
    return "特徴量セットごとの比較結果から、各グループでMAEが最小となった構成は以下です。\n\n" + table


def model_selection_text(final_compare: pd.DataFrame) -> str:
    best = (
        final_compare.sort_values(["sales_group", "mae"])
        .groupby("sales_group", observed=True, as_index=False)
        .first()
    )
    table = markdown_table(
        best,
        [
            ("sales_group", "グループ", -1),
            ("model", "MAE最小モデル", -1),
            ("mae", "MAE", 4),
            ("rmse", "RMSE", 4),
            ("mae_ratio", "MAE ratio", 4),
        ],
    )
    return "3方式を同じホールドアウト期間で比較し、MAEを主指標として選定した結果です。\n\n" + table


def result_table(final_compare: pd.DataFrame) -> str:
    ordered = final_compare.sort_values(["sales_group", "mae"]).copy()
    return markdown_table(
        ordered,
        [
            ("sales_group", "グループ", -1),
            ("model", "モデル", -1),
            ("mae", "MAE", 4),
            ("rmse", "RMSE", 4),
            ("mean_sales", "平均販売数", 4),
            ("mae_ratio", "MAE ratio", 4),
        ],
    )


def interpretation(final_compare: pd.DataFrame) -> str:
    best = (
        final_compare.sort_values(["sales_group", "mae"])
        .groupby("sales_group", observed=True, as_index=False)
        .first()
        .set_index("sales_group")
    )
    parts = []
    for group in ("top", "middle", "bottom"):
        if group not in best.index:
            continue
        row = best.loc[group]
        parts.append(
            f"* `{group}`: MAE {float(row['mae']):.4f}、平均販売数に対するMAE ratioは "
            f"{float(row['mae_ratio']):.4f} でした。"
        )
    parts.append(
        "絶対誤差だけでなくMAE ratioも併用することで、販売規模が異なるグループ間の予測難易度を比較しています。"
    )
    return "\n".join(parts)


def main() -> None:
    feature_path = RESULTS_DIR / "m5_feature_compare_summary.csv"
    final_path = RESULTS_DIR / "m5_final_model_compare.csv"
    if not feature_path.is_file() or not final_path.is_file():
        raise FileNotFoundError(
            "Required result CSVs were not found. Run the notebook before refreshing README."
        )

    feature_compare = pd.read_csv(feature_path)
    final_compare = pd.read_csv(final_path)
    required_feature = {"sales_group", "feature_version", "feature_count", "mae", "rmse", "mae_ratio"}
    required_final = {"sales_group", "model", "mae", "rmse", "mean_sales", "mae_ratio"}
    if missing := required_feature.difference(feature_compare.columns):
        raise ValueError(f"m5_feature_compare_summary.csv is missing: {sorted(missing)}")
    if missing := required_final.difference(final_compare.columns):
        raise ValueError(f"m5_final_model_compare.csv is missing: {sorted(missing)}")

    text = README_PATH.read_text(encoding="utf-8")
    text = replace_block(text, "FEATURE_SELECTION_RESULT", feature_selection_text(feature_compare))
    text = replace_block(text, "MODEL_SELECTION_RESULT", model_selection_text(final_compare))
    text = replace_block(text, "FINAL_RESULTS_TABLE", result_table(final_compare))
    text = replace_block(text, "RESULT_INTERPRETATION", interpretation(final_compare))
    README_PATH.write_text(text, encoding="utf-8")
    print(f"README results were refreshed: {README_PATH}")


if __name__ == "__main__":
    main()
