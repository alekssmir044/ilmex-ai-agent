"""Проверка инструментов без модели: прямые вызовы на фиксированном CSV."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from main import _DATASETS, describe_columns, filter_rows, group_by, load_csv  # noqa: E402

CSV = "\n".join(
    [
        "region;product;quantity;price;date",
        "north;apple;10;1.50;2026-01-05",
        "north;pear;5;2.20;2026-01-06",
        "south;apple;7;1.20;2026-01-05",
        "south;apple;3;1.25;2026-01-07",
        "south;pear;8;2.10;2026-01-08",
        "west;apple;12;1.40;2026-01-09",
    ]
)


async def run() -> None:
    _DATASETS.pop("local", None)

    loaded = await load_csv(csv_content=CSV, file_name="sales.csv")
    assert loaded["rows_loaded"] == 6, loaded
    assert set(loaded["columns_loaded"]) == {"region", "product", "quantity", "price", "date"}, loaded

    described = await describe_columns()
    by_name = {c["name"]: c for c in described["columns"]}
    assert by_name["quantity"]["type"] == "numeric", by_name
    assert by_name["quantity"]["min"] == 3, by_name
    assert by_name["region"]["unique"] == 3, by_name

    filtered = await filter_rows(column="region", op="=", value="south")
    assert filtered["matched_rows"] == 3, filtered

    contains = await filter_rows(column="product", op="contains", value="app", columns=["region", "quantity"])
    assert contains["matched_rows"] == 4, contains

    grouped = await group_by(
        group_by=["region"],
        aggregations=[{"column": "quantity", "agg": "sum"}, {"column": "price", "agg": "mean"}],
        order_by="sum(quantity)",
        order_desc=True,
    )
    rows = grouped["rows"]
    assert grouped["groups_total"] == 3, grouped
    assert rows[0]["region"] == "south" and rows[0]["sum(quantity)"] == 18, rows

    count_only = await group_by(group_by=["product"], aggregations=[{"agg": "count"}])
    assert count_only["groups_total"] == 2, count_only

    _DATASETS.pop("local", None)

    again = await describe_columns()
    assert again["rows"] == 6, again
    assert again["file_name"] == "sales.csv", again

    print("OK: все проверки инструментов пройдены")


if __name__ == "__main__":
    import asyncio

    asyncio.run(run())
