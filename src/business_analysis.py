"""M5 FOODSカテゴリの事業分析。

評価開始前の履歴だけで固定した商品グループを使い、販売傾向、ゼロ販売率、
イベント、SNAP、価格帯との関係を集計してCSVと画像を出力します。
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.item_grouping import apply_item_groups, select_item_groups
RESULTS_DIR = PROJECT_ROOT / "results"
IMAGES_DIR = PROJECT_ROOT / "images"
GROUPING_CUTOFF_DATE = "2015-10-01"
ITEMS_PER_GROUP = 30
GROUP_ORDER = ["top", "middle", "bottom"]

DATA_SEARCH_DIRS = (
    PROJECT_ROOT / "data" / "raw",
    PROJECT_ROOT / "data",
    PROJECT_ROOT / "input",
    PROJECT_ROOT / "m5-forecasting-accuracy",
)


def find_data_file(filename: str) -> Path:
    """既知の保存場所からM5ファイルを検索する。"""
    for base_dir in DATA_SEARCH_DIRS:
        direct = base_dir / filename
        if direct.is_file():
            return direct
        if base_dir.is_dir():
            matches = list(base_dir.rglob(filename))
            if matches:
                return matches[0]
    searched = ", ".join(str(path.relative_to(PROJECT_ROOT)) for path in DATA_SEARCH_DIRS)
    raise FileNotFoundError(
        f"{filename} was not found. Please place Kaggle M5 files under {searched}."
    )


def load_m5_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sales = pd.read_csv(find_data_file("sales_train_validation.csv"))
    calendar = pd.read_csv(find_data_file("calendar.csv"))
    prices = pd.read_csv(find_data_file("sell_prices.csv"))
    calendar["date"] = pd.to_datetime(calendar["date"], errors="raise")
    return sales, calendar, prices


def prepare_long_data(
    sales: pd.DataFrame,
    calendar: pd.DataFrame,
    prices: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """FOODS商品を固定グループ化し、縦持ち・結合済みデータを返す。"""
    foods = sales.loc[sales["cat_id"].eq("FOODS")].copy()
    if foods.empty:
        raise ValueError("FOODS category was not found in sales data.")

    selected_path = RESULTS_DIR / "selected_item_groups.csv"
    selected = select_item_groups(
        foods,
        calendar,
        cutoff_date=GROUPING_CUTOFF_DATE,
        items_per_group=ITEMS_PER_GROUP,
        output_path=selected_path,
    )
    grouped = apply_item_groups(foods, selected)

    day_columns = [column for column in grouped.columns if column.startswith("d_")]
    id_columns = [
        column
        for column in (
            "id",
            "item_id",
            "dept_id",
            "cat_id",
            "store_id",
            "state_id",
            "sales_group",
        )
        if column in grouped.columns
    ]
    long_sales = grouped.melt(
        id_vars=id_columns,
        value_vars=day_columns,
        var_name="d",
        value_name="sales",
    )

    calendar_columns = [
        column
        for column in (
            "d",
            "date",
            "wm_yr_wk",
            "weekday",
            "wday",
            "month",
            "year",
            "event_name_1",
            "event_type_1",
            "event_name_2",
            "event_type_2",
            "snap_CA",
            "snap_TX",
            "snap_WI",
        )
        if column in calendar.columns
    ]
    merged = long_sales.merge(
        calendar[calendar_columns], on="d", how="left", validate="many_to_one"
    )
    merged = merged.merge(
        prices[["store_id", "item_id", "wm_yr_wk", "sell_price"]],
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left",
        validate="many_to_one",
    )
    merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
    merged["sales"] = pd.to_numeric(merged["sales"], errors="coerce").fillna(0.0)
    merged["is_zero_sale"] = merged["sales"].eq(0)
    merged["has_event"] = merged[["event_name_1", "event_name_2"]].notna().any(axis=1)

    snap_columns = {"CA": "snap_CA", "TX": "snap_TX", "WI": "snap_WI"}
    merged["is_snap_day"] = 0
    for state_id, column in snap_columns.items():
        if column in merged.columns:
            mask = merged["state_id"].eq(state_id)
            merged.loc[mask, "is_snap_day"] = (
                pd.to_numeric(merged.loc[mask, column], errors="coerce")
                .fillna(0)
                .astype(int)
            )
    return merged, selected


def _ordered(frame: pd.DataFrame, column: str = "sales_group") -> pd.DataFrame:
    result = frame.copy()
    result[column] = pd.Categorical(result[column], GROUP_ORDER, ordered=True)
    return result.sort_values(column).reset_index(drop=True)


def create_summary(merged: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    item_counts = selected.groupby("sales_group")["item_id"].nunique()
    summary = (
        merged.groupby("sales_group", observed=True)
        .agg(
            item_count=("item_id", "nunique"),
            series_count=("id", "nunique"),
            observation_count=("sales", "size"),
            total_sales=("sales", "sum"),
            mean_daily_sales=("sales", "mean"),
            median_daily_sales=("sales", "median"),
            sales_std=("sales", "std"),
            zero_sales_rate=("is_zero_sale", "mean"),
            mean_sell_price=("sell_price", "mean"),
        )
        .reset_index()
    )
    summary["item_count"] = summary["sales_group"].map(item_counts).fillna(
        summary["item_count"]
    )
    summary["coefficient_of_variation"] = np.where(
        summary["mean_daily_sales"].ne(0),
        summary["sales_std"] / summary["mean_daily_sales"],
        np.nan,
    )
    return _ordered(summary)


def create_event_summary(merged: pd.DataFrame) -> pd.DataFrame:
    result = (
        merged.groupby(["sales_group", "has_event"], observed=True)
        .agg(
            observation_count=("sales", "size"),
            mean_sales=("sales", "mean"),
            total_sales=("sales", "sum"),
            zero_sales_rate=("is_zero_sale", "mean"),
        )
        .reset_index()
    )
    result["event_status"] = np.where(result["has_event"], "event", "no_event")
    return result.drop(columns="has_event")


def create_snap_summary(merged: pd.DataFrame) -> pd.DataFrame:
    result = (
        merged.groupby(["sales_group", "is_snap_day"], observed=True)
        .agg(
            observation_count=("sales", "size"),
            mean_sales=("sales", "mean"),
            total_sales=("sales", "sum"),
            zero_sales_rate=("is_zero_sale", "mean"),
        )
        .reset_index()
    )
    result["snap_status"] = np.where(result["is_snap_day"].eq(1), "snap", "no_snap")
    return result.drop(columns="is_snap_day")


def create_price_summary(merged: pd.DataFrame) -> pd.DataFrame:
    priced = merged.dropna(subset=["sell_price"]).copy()
    if priced.empty:
        return pd.DataFrame(
            columns=[
                "sales_group",
                "price_band",
                "observation_count",
                "mean_sell_price",
                "mean_sales",
                "total_sales",
                "zero_sales_rate",
            ]
        )

    def add_price_band(group: pd.DataFrame) -> pd.DataFrame:
        result = group.copy()
        unique_prices = result["sell_price"].nunique(dropna=True)
        if unique_prices < 3:
            result["price_band"] = "single_or_limited_price"
            return result
        codes = pd.qcut(
            result["sell_price"], q=3, labels=False, duplicates="drop"
        )
        result["price_band"] = codes.map({0.0: "low", 1.0: "middle", 2.0: "high"}).fillna("single_or_limited_price")
        return result

    priced = (
        priced.groupby("sales_group", group_keys=False, observed=True)
        .apply(add_price_band)
        .reset_index(drop=True)
    )
    return (
        priced.groupby(["sales_group", "price_band"], observed=True)
        .agg(
            observation_count=("sales", "size"),
            mean_sell_price=("sell_price", "mean"),
            mean_sales=("sales", "mean"),
            total_sales=("sales", "sum"),
            zero_sales_rate=("is_zero_sale", "mean"),
        )
        .reset_index()
    )


def _save_bar(
    frame: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    pivot = frame.pivot(index=x, columns="sales_group", values=y)
    pivot = pivot.reindex(columns=GROUP_ORDER)
    ax = pivot.plot(kind="bar", figsize=(10, 6))
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=0)
    ax.figure.tight_layout()
    ax.figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(ax.figure)


def create_images(
    merged: pd.DataFrame,
    summary: pd.DataFrame,
    event_summary: pd.DataFrame,
    snap_summary: pd.DataFrame,
    price_summary: pd.DataFrame,
) -> None:
    monthly = (
        merged.assign(month=merged["date"].dt.to_period("M").dt.to_timestamp())
        .groupby(["month", "sales_group"], observed=True)["sales"]
        .mean()
        .reset_index()
    )
    monthly_pivot = monthly.pivot(
        index="month", columns="sales_group", values="sales"
    ).reindex(columns=GROUP_ORDER)
    ax = monthly_pivot.plot(figsize=(12, 6))
    ax.set_title("Average daily sales trend by sales group")
    ax.set_xlabel("Month")
    ax.set_ylabel("Average daily sales")
    ax.figure.tight_layout()
    ax.figure.savefig(
        IMAGES_DIR / "business_sales_trend_by_group.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(ax.figure)

    zero = summary.set_index("sales_group").reindex(GROUP_ORDER)["zero_sales_rate"]
    ax = zero.plot(kind="bar", figsize=(9, 6))
    ax.set_title("Zero-sales rate by sales group")
    ax.set_xlabel("")
    ax.set_ylabel("Zero-sales rate")
    ax.tick_params(axis="x", rotation=0)
    ax.figure.tight_layout()
    ax.figure.savefig(
        IMAGES_DIR / "business_zero_rate_by_group.png", dpi=150, bbox_inches="tight"
    )
    plt.close(ax.figure)

    _save_bar(
        event_summary,
        "event_status",
        "mean_sales",
        "Event-day impact by sales group",
        "Average daily sales",
        IMAGES_DIR / "business_event_impact.png",
    )
    _save_bar(
        snap_summary,
        "snap_status",
        "mean_sales",
        "SNAP-day impact by sales group",
        "Average daily sales",
        IMAGES_DIR / "business_snap_impact.png",
    )
    if not price_summary.empty:
        _save_bar(
            price_summary,
            "price_band",
            "mean_sales",
            "Sales by price band and sales group",
            "Average daily sales",
            IMAGES_DIR / "business_price_vs_sales.png",
        )


def save_csv(frame: pd.DataFrame, filename: str) -> None:
    frame.to_csv(RESULTS_DIR / filename, index=False, encoding="utf-8-sig")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading M5 data...")
    sales, calendar, prices = load_m5_data()
    print("Selecting fixed item groups and preparing long-format data...")
    merged, selected = prepare_long_data(sales, calendar, prices)

    summary = create_summary(merged, selected)
    event_summary = create_event_summary(merged)
    snap_summary = create_snap_summary(merged)
    price_summary = create_price_summary(merged)

    save_csv(summary, "business_analysis_summary.csv")
    save_csv(event_summary, "business_event_impact_summary.csv")
    save_csv(snap_summary, "business_snap_impact_summary.csv")
    save_csv(price_summary, "business_price_vs_sales_summary.csv")
    create_images(merged, summary, event_summary, snap_summary, price_summary)

    print("Business analysis completed successfully.")
    print(f"Selected item groups: {RESULTS_DIR / 'selected_item_groups.csv'}")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Images directory: {IMAGES_DIR}")


if __name__ == "__main__":
    main()
