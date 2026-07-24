from __future__ import annotations

import unittest

import pandas as pd

from src.item_grouping import apply_item_groups, select_item_groups


class ItemGroupingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.calendar = pd.DataFrame(
            {
                "d": ["d_1", "d_2", "d_3", "d_4"],
                "date": [
                    "2015-09-28",
                    "2015-09-29",
                    "2015-10-01",
                    "2015-10-02",
                ],
            }
        )
        rows = []
        for index in range(1, 10):
            rows.append(
                {
                    "item_id": f"item_{index:02d}",
                    "dept_id": "FOODS_1",
                    "cat_id": "FOODS",
                    "store_id": "CA_1",
                    "d_1": index,
                    "d_2": index,
                    # cutoff後の値を逆転・巨大化しても選定へ影響しないことを確認する
                    "d_3": (10 - index) * 1000,
                    "d_4": (10 - index) * 1000,
                }
            )
        self.sales = pd.DataFrame(rows)

    def test_selection_uses_only_pre_cutoff_history(self) -> None:
        selected = select_item_groups(
            self.sales,
            self.calendar,
            cutoff_date="2015-10-01",
            items_per_group=2,
        )
        top_items = selected.loc[selected["sales_group"] == "top", "item_id"].tolist()
        bottom_items = selected.loc[
            selected["sales_group"] == "bottom", "item_id"
        ].tolist()
        self.assertEqual(top_items, ["item_09", "item_08"])
        self.assertEqual(bottom_items, ["item_02", "item_01"])

    def test_groups_are_non_overlapping_and_fixed_size(self) -> None:
        selected = select_item_groups(
            self.sales,
            self.calendar,
            cutoff_date="2015-10-01",
            items_per_group=2,
        )
        self.assertFalse(selected["item_id"].duplicated().any())
        self.assertEqual(
            selected.groupby("sales_group")["item_id"].nunique().to_dict(),
            {"bottom": 2, "middle": 2, "top": 2},
        )

    def test_apply_item_groups_keeps_only_selected_items(self) -> None:
        selected = select_item_groups(
            self.sales,
            self.calendar,
            cutoff_date="2015-10-01",
            items_per_group=2,
        )
        grouped = apply_item_groups(self.sales, selected)
        self.assertEqual(grouped["item_id"].nunique(), 6)
        self.assertEqual(set(grouped["sales_group"]), {"top", "middle", "bottom"})


if __name__ == "__main__":
    unittest.main()
