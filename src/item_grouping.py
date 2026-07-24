"""M5ポートフォリオで共通利用する商品グループ選定処理。

実務のABC分析の考え方を参考にした、販売数量ベースの固定サイズ層化です。
評価開始日より前の履歴だけを使い、top / middle / bottom を選定します。
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd

PathLike = Union[str, Path]

DEFAULT_GROUPING_CUTOFF_DATE = pd.Timestamp("2015-10-01")
DEFAULT_ITEMS_PER_GROUP = 30
GROUP_ORDER = ("top", "middle", "bottom")


def _history_day_columns(
    food_sales: pd.DataFrame,
    calendar: pd.DataFrame,
    cutoff_date: pd.Timestamp,
) -> list[str]:
    required = {"d", "date"}
    missing = required.difference(calendar.columns)
    if missing:
        raise ValueError(f"calendar is missing required columns: {sorted(missing)}")

    calendar_for_selection = calendar[["d", "date"]].copy()
    calendar_for_selection["date"] = pd.to_datetime(
        calendar_for_selection["date"], errors="raise"
    )
    day_columns = calendar_for_selection.loc[
        calendar_for_selection["date"] < cutoff_date, "d"
    ].tolist()
    day_columns = [column for column in day_columns if column in food_sales.columns]

    if not day_columns:
        raise ValueError(
            "No sales day columns were found before the grouping cutoff date: "
            f"{cutoff_date.date()}"
        )
    return day_columns


def select_item_groups(
    food_sales: pd.DataFrame,
    calendar: pd.DataFrame,
    cutoff_date: pd.Timestamp | str = DEFAULT_GROUPING_CUTOFF_DATE,
    items_per_group: int = DEFAULT_ITEMS_PER_GROUP,
    output_path: PathLike | None = None,
) -> pd.DataFrame:
    """評価開始前の販売数だけで固定商品群を選定する。

    * top: 累計販売数が大きい商品
    * middle: 非ゼロ商品の順位中央付近
    * bottom: 累計販売数が小さい商品（累計0は除外）

    同率時は item_id で決定し、毎回同じ結果になるようにします。
    """
    if items_per_group <= 0:
        raise ValueError("items_per_group must be greater than zero")
    if "item_id" not in food_sales.columns:
        raise ValueError("food_sales is missing required column: item_id")

    cutoff = pd.Timestamp(cutoff_date)
    history_days = _history_day_columns(food_sales, calendar, cutoff)

    metadata_columns = [
        column
        for column in ("item_id", "dept_id", "cat_id")
        if column in food_sales.columns
    ]
    item_metadata = (
        food_sales[metadata_columns]
        .drop_duplicates(subset=["item_id"])
        .set_index("item_id")
    )

    item_totals = (
        food_sales.groupby("item_id", sort=False)[history_days]
        .sum()
        .sum(axis=1)
        .rename("total_sales_before_cutoff")
        .to_frame()
        .join(item_metadata, how="left")
        .reset_index()
    )

    ranked = (
        item_totals[item_totals["total_sales_before_cutoff"] > 0]
        .sort_values(
            ["total_sales_before_cutoff", "item_id"],
            ascending=[False, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    minimum_required = items_per_group * 3
    if len(ranked) < minimum_required:
        raise ValueError(
            "Not enough non-zero-sales items to create three groups. "
            f"required={minimum_required}, available={len(ranked)}"
        )

    top = ranked.head(items_per_group).copy()
    middle_start = (len(ranked) - items_per_group) // 2
    middle = ranked.iloc[middle_start : middle_start + items_per_group].copy()
    bottom = ranked.tail(items_per_group).copy()

    parts: list[pd.DataFrame] = []
    for name, frame in (("top", top), ("middle", middle), ("bottom", bottom)):
        part = frame.copy()
        part["sales_group"] = name
        parts.append(part)

    selected = pd.concat(parts, ignore_index=True)
    if selected["item_id"].duplicated().any():
        duplicates = selected.loc[
            selected["item_id"].duplicated(keep=False), "item_id"
        ].tolist()
        raise RuntimeError(f"The selected groups overlap: {duplicates}")

    selected["grouping_cutoff_date"] = cutoff.date().isoformat()
    selected["selection_method"] = "sales_volume_stratified_abc_inspired"
    selected["items_per_group"] = items_per_group
    selected["_group_order"] = selected["sales_group"].map(
        {name: index for index, name in enumerate(GROUP_ORDER)}
    )
    selected = (
        selected.sort_values(
            ["_group_order", "total_sales_before_cutoff", "item_id"],
            ascending=[True, False, True],
            kind="mergesort",
        )
        .drop(columns="_group_order")
        .reset_index(drop=True)
    )

    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        selected.to_csv(destination, index=False, encoding="utf-8-sig")
    return selected


def apply_item_groups(
    food_sales: pd.DataFrame,
    selected_item_groups: pd.DataFrame,
) -> pd.DataFrame:
    """選定商品だけを残し、固定した sales_group を付与する。"""
    required = {"item_id", "sales_group"}
    missing = required.difference(selected_item_groups.columns)
    if missing:
        raise ValueError(
            f"selected_item_groups is missing required columns: {sorted(missing)}"
        )

    group_map = selected_item_groups[["item_id", "sales_group"]].copy()
    if group_map["item_id"].duplicated().any():
        raise ValueError("selected_item_groups contains duplicate item_id values")

    grouped = food_sales.drop(columns=["sales_group"], errors="ignore").merge(
        group_map,
        on="item_id",
        how="inner",
        validate="many_to_one",
    )

    expected = group_map["item_id"].nunique()
    actual = grouped["item_id"].nunique()
    if actual != expected:
        missing_items = sorted(set(group_map["item_id"]) - set(grouped["item_id"]))
        raise RuntimeError(f"Selected items were not found in food_sales: {missing_items}")
    return grouped
