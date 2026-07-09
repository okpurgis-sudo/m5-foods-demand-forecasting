from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR_CANDIDATES = [
    PROJECT_ROOT / "data",
    PROJECT_ROOT / "data" / "raw",
    PROJECT_ROOT / "input",
    PROJECT_ROOT / "m5-forecasting-accuracy",
]

RESULTS_DIR = PROJECT_ROOT / "results"
IMAGES_DIR = PROJECT_ROOT / "images"

RESULTS_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)


GROUP_LABELS = {
    "top": "top（よく売れる商品）",
    "middle": "middle（平均的な商品）",
    "bottom": "bottom（あまり売れない商品）",
}

GROUP_ORDER = {
    "top": 0,
    "middle": 1,
    "bottom": 2,
}

PRICE_BIN_LABELS = {
    "low": "低価格",
    "mid_low": "やや低価格",
    "mid_high": "やや高価格",
    "high": "高価格",
}


def setup_japanese_font() -> None:
    """
    Matplotlibで日本語が文字化けしにくいように、利用可能な日本語フォントを設定します。
    Windowsなら Yu Gothic / Meiryo、Linuxなら Noto Sans CJK JP などを優先します。
    """
    preferred_fonts = [
        "Yu Gothic",
        "Yu Gothic UI",
        "Meiryo",
        "MS Gothic",
        "Noto Sans CJK JP",
        "Noto Sans JP",
        "IPAexGothic",
        "IPAGothic",
        "TakaoGothic",
        "DejaVu Sans",
    ]

    installed_fonts = {font.name for font in font_manager.fontManager.ttflist}

    for font_name in preferred_fonts:
        if font_name in installed_fonts:
            plt.rcParams["font.family"] = font_name
            break

    # マイナス記号の文字化け対策
    plt.rcParams["axes.unicode_minus"] = False


def find_data_file(filename: str) -> Path:
    for data_dir in DATA_DIR_CANDIDATES:
        path = data_dir / filename
        if path.exists():
            return path

    raise FileNotFoundError(
        f"{filename} was not found. "
        "Please place Kaggle M5 files under data/, data/raw/, input/, "
        "or m5-forecasting-accuracy/."
    )


def load_m5_data():
    sales_path = find_data_file("sales_train_validation.csv")
    calendar_path = find_data_file("calendar.csv")
    prices_path = find_data_file("sell_prices.csv")

    print(f"Loading sales data: {sales_path}")
    sales = pd.read_csv(sales_path)

    print(f"Loading calendar data: {calendar_path}")
    calendar = pd.read_csv(calendar_path)

    print(f"Loading price data: {prices_path}")
    prices = pd.read_csv(prices_path)

    return sales, calendar, prices


def assign_sales_group(food_sales: pd.DataFrame, day_cols: list[str]) -> pd.DataFrame:
    """
    各商品・店舗行の累計販売数をもとに top / middle / bottom を付与します。
    """
    food_sales = food_sales.copy()
    food_sales["total_sales"] = food_sales[day_cols].sum(axis=1)

    # 同じ販売数が多い場合でもqcutが失敗しにくいようにrankを使います。
    ranked_total = food_sales["total_sales"].rank(method="first")

    food_sales["sales_group"] = pd.qcut(
        ranked_total,
        q=3,
        labels=["bottom", "middle", "top"],
    )

    food_sales["sales_group"] = food_sales["sales_group"].astype(str)
    return food_sales


def create_group_summary(food_sales: pd.DataFrame, day_cols: list[str]) -> pd.DataFrame:
    summary_rows = []

    for group_name, group_df in food_sales.groupby("sales_group"):
        values = group_df[day_cols].to_numpy()

        summary_rows.append(
            {
                "sales_group": group_name,
                "販売グループ": GROUP_LABELS.get(group_name, group_name),
                "商品_店舗数": len(group_df),
                "累計販売数": float(values.sum()),
                "1日あたり平均販売数": float(values.mean()),
                "1日あたり中央値": float(np.median(values)),
                "販売数が0だった日の割合": float((values == 0).mean()),
                "最大日次販売数": float(values.max()),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary["sort_order"] = summary["sales_group"].map(GROUP_ORDER)
    summary = summary.sort_values("sort_order").drop(columns=["sort_order", "sales_group"])

    output_path = RESULTS_DIR / "business_analysis_summary.csv"
    summary.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {output_path}")

    return summary


def plot_sales_trend_by_group(food_sales: pd.DataFrame, calendar: pd.DataFrame, day_cols: list[str]):
    trend_rows = []

    for group_name, group_df in food_sales.groupby("sales_group"):
        daily_sales = group_df[day_cols].sum(axis=0)
        temp = pd.DataFrame(
            {
                "d": daily_sales.index,
                "販売数": daily_sales.values,
                "sales_group": group_name,
                "販売グループ": GROUP_LABELS.get(group_name, group_name),
            }
        )
        trend_rows.append(temp)

    trend = pd.concat(trend_rows, ignore_index=True)
    trend = trend.merge(calendar[["d", "date"]], on="d", how="left")
    trend["date"] = pd.to_datetime(trend["date"])

    plt.figure(figsize=(12, 6))
    for _, group_df in trend.groupby("販売グループ"):
        group_df = group_df.sort_values("date")
        rolling_sales = group_df["販売数"].rolling(28, min_periods=1).mean()
        plt.plot(group_df["date"], rolling_sales, label=group_df["販売グループ"].iloc[0])

    plt.title("販売規模別の販売数推移（28日移動平均）")
    plt.xlabel("日付")
    plt.ylabel("販売数合計")
    plt.legend()
    plt.tight_layout()

    output_path = IMAGES_DIR / "business_sales_trend_by_group.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_zero_rate_by_group(summary: pd.DataFrame):
    plot_df = summary.copy()

    plt.figure(figsize=(9, 5))
    plt.bar(plot_df["販売グループ"], plot_df["販売数が0だった日の割合"])
    plt.title("商品グループ別：販売数が0だった日の割合")
    plt.xlabel("商品グループ")
    plt.ylabel("販売数が0だった日の割合")
    plt.ylim(0, 1)
    plt.xticks(rotation=15)
    plt.tight_layout()

    output_path = IMAGES_DIR / "business_zero_rate_by_group.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def create_state_group_daily_sales(food_sales: pd.DataFrame, calendar: pd.DataFrame, day_cols: list[str]):
    """
    SNAPは州ごとに対象日が異なるため、sales_group × state_id 単位で日別販売数を集計します。
    """
    rows = []

    for (group_name, state_id), group_df in food_sales.groupby(["sales_group", "state_id"]):
        daily_sales = group_df[day_cols].sum(axis=0)

        temp = pd.DataFrame(
            {
                "d": daily_sales.index,
                "販売数": daily_sales.values,
                "sales_group": group_name,
                "販売グループ": GROUP_LABELS.get(group_name, group_name),
                "state_id": state_id,
            }
        )
        rows.append(temp)

    daily = pd.concat(rows, ignore_index=True)

    calendar_cols = [
        "d",
        "date",
        "event_name_1",
        "event_name_2",
        "snap_CA",
        "snap_TX",
        "snap_WI",
    ]
    available_cols = [col for col in calendar_cols if col in calendar.columns]
    daily = daily.merge(calendar[available_cols], on="d", how="left")
    daily["date"] = pd.to_datetime(daily["date"])

    event_1 = daily["event_name_1"].notna() if "event_name_1" in daily.columns else False
    event_2 = daily["event_name_2"].notna() if "event_name_2" in daily.columns else False
    daily["イベント日"] = (event_1 | event_2).astype(int)

    def get_snap(row):
        snap_col = f"snap_{row['state_id']}"
        if snap_col in row.index:
            return row[snap_col]
        return 0

    daily["SNAP対象日"] = daily.apply(get_snap, axis=1).fillna(0).astype(int)

    return daily


def plot_event_snap_impact(daily: pd.DataFrame):
    event_summary = (
        daily.groupby(["販売グループ", "イベント日"])["販売数"]
        .mean()
        .reset_index()
    )

    snap_summary = (
        daily.groupby(["販売グループ", "SNAP対象日"])["販売数"]
        .mean()
        .reset_index()
    )

    event_pivot = event_summary.pivot(
        index="販売グループ",
        columns="イベント日",
        values="販売数",
    ).fillna(0)

    snap_pivot = snap_summary.pivot(
        index="販売グループ",
        columns="SNAP対象日",
        values="販売数",
    ).fillna(0)

    event_pivot = event_pivot.rename(columns={0: "通常日", 1: "イベント日"})
    snap_pivot = snap_pivot.rename(columns={0: "SNAP対象外", 1: "SNAP対象日"})

    event_output = RESULTS_DIR / "business_event_impact_summary.csv"
    snap_output = RESULTS_DIR / "business_snap_impact_summary.csv"

    event_pivot.to_csv(event_output, encoding="utf-8-sig")
    snap_pivot.to_csv(snap_output, encoding="utf-8-sig")

    print(f"Saved: {event_output}")
    print(f"Saved: {snap_output}")

    plt.figure(figsize=(9, 5))
    event_pivot.plot(kind="bar", ax=plt.gca())
    plt.title("イベント日と通常日の平均販売数比較")
    plt.xlabel("販売グループ")
    plt.ylabel("平均販売数")
    plt.xticks(rotation=15)
    plt.tight_layout()

    output_path = IMAGES_DIR / "business_event_impact.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")

    plt.figure(figsize=(9, 5))
    snap_pivot.plot(kind="bar", ax=plt.gca())
    plt.title("SNAP対象日と非対象日の平均販売数比較")
    plt.xlabel("販売グループ")
    plt.ylabel("平均販売数")
    plt.xticks(rotation=15)
    plt.tight_layout()

    output_path = IMAGES_DIR / "business_snap_impact.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def analyze_price_vs_sales(
    food_sales: pd.DataFrame,
    calendar: pd.DataFrame,
    prices: pd.DataFrame,
    day_cols: list[str],
    recent_days: int = 180,
):
    """
    価格分析はデータ量が大きくなりやすいため、直近日数だけを使います。
    """
    recent_day_cols = day_cols[-recent_days:]

    id_cols = [
        "id",
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id",
        "sales_group",
    ]

    print(f"Creating price analysis data for recent {recent_days} days...")

    long_sales = food_sales[id_cols + recent_day_cols].melt(
        id_vars=id_cols,
        value_vars=recent_day_cols,
        var_name="d",
        value_name="sales",
    )

    calendar_small = calendar[["d", "wm_yr_wk"]]
    long_sales = long_sales.merge(calendar_small, on="d", how="left")

    long_sales = long_sales.merge(
        prices[["store_id", "item_id", "wm_yr_wk", "sell_price"]],
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left",
    )

    long_sales = long_sales.dropna(subset=["sell_price"])

    if long_sales.empty:
        print("Price analysis skipped because no matching price data was found.")
        return

    long_sales["price_bin"] = pd.qcut(
        long_sales["sell_price"].rank(method="first"),
        q=4,
        labels=["low", "mid_low", "mid_high", "high"],
    )

    price_summary = (
        long_sales.groupby(["sales_group", "price_bin"], observed=True)
        .agg(
            平均販売数=("sales", "mean"),
            平均価格=("sell_price", "mean"),
            データ件数=("sales", "size"),
        )
        .reset_index()
    )

    price_summary["販売グループ"] = price_summary["sales_group"].map(GROUP_LABELS)
    price_summary["価格帯"] = price_summary["price_bin"].astype(str).map(PRICE_BIN_LABELS)

    output_df = price_summary[
        ["販売グループ", "価格帯", "平均販売数", "平均価格", "データ件数"]
    ]

    output_path = RESULTS_DIR / "business_price_vs_sales_summary.csv"
    output_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {output_path}")

    pivot = output_df.pivot(
        index="価格帯",
        columns="販売グループ",
        values="平均販売数",
    )

    price_order = ["低価格", "やや低価格", "やや高価格", "高価格"]
    pivot = pivot.reindex(price_order)

    plt.figure(figsize=(9, 5))
    pivot.plot(kind="bar", ax=plt.gca())
    plt.title("価格帯ごとの平均販売数")
    plt.xlabel("価格帯")
    plt.ylabel("平均販売数")
    plt.xticks(rotation=0)
    plt.tight_layout()

    output_path = IMAGES_DIR / "business_price_vs_sales.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    setup_japanese_font()

    sales, calendar, prices = load_m5_data()

    day_cols = [col for col in sales.columns if col.startswith("d_")]

    print("Filtering FOODS category...")
    food_sales = sales[sales["cat_id"] == "FOODS"].copy()

    print(f"FOODS item-store rows: {len(food_sales):,}")
    print(f"Number of day columns: {len(day_cols):,}")

    print("Assigning sales groups...")
    food_sales = assign_sales_group(food_sales, day_cols)

    print("Creating group summary...")
    summary = create_group_summary(food_sales, day_cols)

    print("Creating sales trend chart...")
    plot_sales_trend_by_group(food_sales, calendar, day_cols)

    print("Creating zero rate chart...")
    plot_zero_rate_by_group(summary)

    print("Creating event / SNAP analysis...")
    daily = create_state_group_daily_sales(food_sales, calendar, day_cols)
    plot_event_snap_impact(daily)

    print("Creating price analysis...")
    analyze_price_vs_sales(food_sales, calendar, prices, day_cols)

    print("Business analysis completed.")


if __name__ == "__main__":
    main()
