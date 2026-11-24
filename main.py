"""Агент-аналитик данных: CSV из Google Drive и фиксированные инструменты анализа.

Нет песочницы и произвольного кода: модель может только вызывать коннектор
Google Drive и четыре фиксированных инструмента (load, describe, filter, group_by).
"""

from __future__ import annotations

import io
import json
import math
import re
import tempfile
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
import pandas as pd
from pydantic import Field

from mistralai import agents

_DRIVE_CONNECTOR = "google_drive_mcp"
_MAX_PREVIEW_ROWS = 100
_FILTER_OPS = ("=", "!=", ">", ">=", "<", "<=", "in", "not_in", "contains", "startswith", "endswith", "isnull", "notnull")
_AGGS = ("count", "nunique", "sum", "mean", "median", "min", "max", "std")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_jsonable(item) for item in value]
    if value is None or value is pd.NaT or (isinstance(value, (float, np.floating)) and math.isnan(float(value))) or pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _session_key(session_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", session_id) or "local"


def _cache_dir(session_id: str) -> Path:
    cache = Path(tempfile.gettempdir()) / "mistral-data-analyst" / _session_key(session_id)
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _cache_meta(session_id: str) -> Path:
    return _cache_dir(session_id) / "meta.json"


def _write_cache(session_id: str, csv_content: str, file_name: str) -> None:
    (_cache_dir(session_id) / "current.csv").write_text(csv_content, encoding="utf-8")
    _cache_meta(session_id).write_text(json.dumps({"file_name": file_name}), encoding="utf-8")


_DATASETS: dict[str, pd.DataFrame] = {}


def _restore(session_id: str) -> None:
    if session_id in _DATASETS:
        return
    path = _cache_dir(session_id) / "current.csv"
    if path.exists():
        _DATASETS[session_id] = pd.read_csv(path, sep=None, engine="python")

def _get_df(session_id: str) -> tuple[pd.DataFrame, str]:
    _restore(session_id)
    df = _DATASETS.get(session_id)
    if df is None:
        raise agents.ToolError(
            "Набор данных не загружен. Сначала найди файл в Google Drive коннектором "
            f"{_DRIVE_CONNECTOR}, прочитай его содержимое инструментом чтения файлов коннектора, "
            "затем вызови data.load_csv, передав содержимое целиком."
        )
    meta = _cache_meta(session_id)
    file_name = json.loads(meta.read_text(encoding="utf-8")).get("file_name", "data.csv") if meta.exists() else "data.csv"
    return df, file_name


def _resolve_column(df: pd.DataFrame, column: str) -> str:
    if column in df.columns:
        return column
    lowered = {str(col).lower(): col for col in df.columns}
    if column.lower() in lowered:
        return lowered[column.lower()]
    raise agents.ToolError(
        f"Столбца {column!r} нет в данных. Доступные столбцы: {', '.join(map(str, df.columns))}."
    )


def _profile(df: pd.DataFrame) -> list[dict[str, Any]]:
    columns: list[dict[str, Any]] = []
    for name in df.columns:
        series = df[name]
        column: dict[str, Any] = {
            "name": str(name),
            "type": str(series.dtype),
            "missing": int(series.isna().sum()),
            "unique": int(series.nunique(dropna=True)),
            "sample_values": [_jsonable(v) for v in series.dropna().unique()[:5]],
        }
        numeric = pd.to_numeric(series, errors="coerce")
        non_null = series.notna().sum()
        if non_null and (numeric.notna().sum() / non_null) >= 0.9:
            clean = numeric.dropna()
            column["type"] = "numeric"
            column["min"] = _jsonable(clean.min())
            column["max"] = _jsonable(clean.max())
            column["mean"] = _jsonable(clean.mean())
        columns.append(column)
    return columns


def _check_names(df: pd.DataFrame, names: list[str], what: str) -> None:
    known = {str(col).lower() for col in df.columns}
    unknown = [name for name in names if str(name).lower() not in known]
    if unknown:
        raise agents.ToolError(
            f"Неизвестные столбцы в параметре {what!r}: {', '.join(unknown)}. "
            f"Доступны: {', '.join(map(str, df.columns))}."
        )


def _coerce_value(series: pd.Series, value: str) -> Any:
    stripped = value.strip()
    if stripped.lower() in ("true", "false") and series.dtype == "bool":
        return stripped.lower() == "true"
    if series.dtype.kind in "if" or (pd.to_numeric(series, errors="coerce").notna().any() and re.fullmatch(r"-?\d+([.,]\d+)?", stripped)):
        try:
            return float(stripped.replace(",", ".")) if "." in stripped or "," in stripped else int(stripped)
        except ValueError:
            return value
    if series.dtype.kind == "M":
        ts = pd.to_datetime(stripped, errors="coerce")
        if not pd.isna(ts):
            return ts
    return value


def _mask(df: pd.DataFrame, column: str, op: str, value: Any) -> pd.Series:
    series = df[column]
    match op:
        case "=":
            return series == value
        case "!=":
            return series != value
        case ">":
            return series > value
        case ">=":
            return series >= value
        case "<":
            return series < value
        case "<=":
            return series <= value
        case "in" | "not_in":
            items = [item.strip() for item in str(value).split(",") if item.strip() != ""]
            coerced = [type(series.dropna().iloc[0])(item) if len(series.dropna()) else item for item in items]
            mask = series.isin(coerced)
            return ~mask if op == "not_in" else mask
        case "contains":
            return series.astype(str).str.contains(str(value), case=False, regex=False, na=False)
        case "startswith":
            return series.astype(str).str.startswith(str(value), na=False)
        case "endswith":
            return series.astype(str).str.endswith(str(value), na=False)
        case "isnull":
            return series.isna()
        case "notnull":
            return series.notna()
        case _:
            raise agents.ToolError(
                f"Операция {op!r} не поддерживается. Разрешены: {', '.join(_FILTER_OPS)}."
            )


@agents.tool(namespace="data")
async def load_csv(
    ctx: agents.ToolCallbackContext,
    /,
    csv_content: Annotated[str, Field(description="Полное содержимое CSV-файла, прочитанного из Google Drive.")],
    file_name: Annotated[str, Field(description="Имя файла, как оно записано в Google Drive.")],
) -> dict[str, Any]:
    """Загрузить CSV-файл в набор данных сессии для последующего анализа.

    Сначала найди файл в Google Drive коннектором google_drive_mcp, прочитай его
    инструментом чтения файлов коннектора, затем передай прочитанное сюда целиком,
    не пересказывая: содержимое уходит в анализ, а не в ответ.
    """
    if not csv_content.strip():
        raise agents.ToolError("Содержимое файла пустое. Проверь, что в Drive лежит именно CSV.")
    try:
        df = pd.read_csv(io.StringIO(csv_content), sep=None, engine="python", nrows=500_000)
    except Exception as error:
        raise agents.ToolError(
            f"Не удалось разобрать CSV ({error}). Возможно, это не CSV: файл в Drive может быть в формате XLSX, JSON и т.п."
        ) from None
    if df.empty:
        raise agents.ToolError("CSV разобран, но строк в нём нет.")
    truncated = csv_content.count("\n") + 1 > len(df) + 1
    _DATASETS[ctx.session.id] = df
    _write_cache(ctx.session.id, csv_content, file_name)
    return {
        "file_name": file_name,
        "rows_loaded": int(len(df)),
        "columns_loaded": [str(col) for col in df.columns],
        "truncated_to_500k_rows": truncated or None,
        "next_step": "Данные в сессии готовы: описывай их через data.describe_columns, фильтруй через data.filter_rows, агрегируй через data.group_by.",
    }


@agents.tool(namespace="data", annotations=agents.ToolAnnotations(read_only_hint=True))
async def describe_columns(ctx: agents.ToolCallbackContext, /) -> dict[str, Any]:
    """Описать загруженный набор данных: столбцы, типы, пропуски, уникальные значения, min/max/mean."""
    df, file_name = _get_df(ctx.session.id)
    return {
        "file_name": file_name,
        "rows": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": _profile(df),
    }


@agents.tool(namespace="data", annotations=agents.ToolAnnotations(read_only_hint=True))
async def filter_rows(
    ctx: agents.ToolCallbackContext,
    /,
    column: Annotated[str, Field(description="Столбец для условия.")],
    op: Annotated[Literal[(*_FILTER_OPS,)], Field(description="Операция сравнения.")],
    value: Annotated[str, Field(description="Значение условия; для in и not_in — значения через запятую; для isnull и notnull — не учитывается.")],
    columns: Annotated[list[str] | None, Field(description="Ограничить вывод этими столбцами.")] = None,
) -> dict[str, Any]:
    """Отфильтровать строки загруженного набора и вернуть подходящие строки (не более 100) и их количество."""
    df, file_name = _get_df(ctx.session.id)
    column = _resolve_column(df, column)
    mask = _mask(df, column, op, value)
    matched = df[mask]
    if columns:
        kept = [_resolve_column(df, col) for col in columns]
        matched = matched[kept]
    preview = matched.head(_MAX_PREVIEW_ROWS)
    return {
        "file_name": file_name,
        "op": f"{column} {op} {value!r}",
        "matched_rows": int(len(matched)),
        "returned_rows": int(len(preview)),
        "rows": [_jsonable(row) for row in preview.to_dict(orient="records")],
    }


@agents.tool(namespace="data", annotations=agents.ToolAnnotations(read_only_hint=True))
async def group_by(
    ctx: agents.ToolCallbackContext,
    /,
    group_by: Annotated[list[str], Field(description="Столбцы группировки (1-3 столбца).")],
    aggregations: Annotated[list[dict[str, str]], Field(description='Список агрегаций вида {"column": "...", "agg": "count|nunique|sum|mean|median|min|max|std"}.')],
    order_by: Annotated[str | None, Field(description="Столбец результата для сортировки.")] = None,
    order_desc: bool = True,
) -> dict[str, Any]:
    """Сгруппировать строки по столбцам и вычислить агрегаты (sum, mean, count и т.п.); вернуть таблицу агрегатов."""
    df, file_name = _get_df(ctx.session.id)
    groups = [_resolve_column(df, col) for col in group_by[:3]] or None
    if not aggregations:
        raise agents.ToolError("Передай хотя бы одну агрегацию, например {\"column\": \"price\", \"agg\": \"sum\"}.")
    result = df.groupby(groups, dropna=False)
    table = pd.DataFrame(index=result.size().index)
    for spec in aggregations:
        column = str(spec.get("column", ""))
        agg = str(spec.get("agg", "")).lower()
        if agg not in _AGGS:
            raise agents.ToolError(
                f"Агрегация {agg!r} не поддерживается. Разрешены: {', '.join(_AGGS)}."
            )
        if agg == "count":
            label = f"{agg}({column})" if column else "count(*)"
            table[label] = result[column].count() if column else result.size()
        else:
            column = _resolve_column(df, column)
            label = f"{agg}({column})"
            table[label] = getattr(result[column], agg)()
    if order_by:
        if order_by not in table.columns and order_by not in table.index.names:
            raise agents.ToolError(
                f"Столбца сортировки {order_by!r} нет в результате. Варианты: {', '.join(map(str, table.reset_index().columns))}."
            )
        table = table.sort_values(order_by, ascending=not order_desc)
    table = table.head(_MAX_PREVIEW_ROWS)
    return {
        "file_name": file_name,
        "groups_returned": int(len(table)),
        "groups_total": int(result.ngroups),
        "rows": [_jsonable(row) for row in table.reset_index().to_dict(orient="records")],
    }


agent = agents.Agent(
    model="mistral-large-4",
    name="data-analyst",
    description="Аналитик данных: загружает CSV из Google Drive и отвечает на вопросы по нему с реальными числами.",
    instructions=(
        "Ты — аналитик данных. Работай только на русском языке.\n"
        "\n"
        "Файл приходит ссылкой на Google Drive или названием. План работы:\n"
        f"1. Найди файл коннектором {_DRIVE_CONNECTOR} (поиск по имени или разбор ссылки).\n"
        f"2. Прочитай его содержимое инструментом чтения файлов этого коннектора.\n"
        "3. Вызови data.load_csv, передав содержимое целиком, и кратко подтверди пользователю загрузку.\n"
        "4. Отвечай на вопросы, вызывая data.describe_columns, data.filter_rows и data.group_by — каждый ответ строй строго на числах из результатов инструментов.\n"
        "\n"
        "Правила:\n"
        "- Не выдумывай числа и не оценивай данные «на глаз»: если данных не хватает для точного ответа, так и скажи.\n"
        "- Никогда не показывай пользователю сырые куски CSV без необходимости: пересказывай результаты инструментов.\n"
        "- Считай набор данных загруженным до конца сессии, а файл и все обсуждённые договорённости помни сам; не проси пользователя присылать файл или объяснять заново.\n"
        "- Ссылайся на столбцы по точным именам из data.describe_columns.\n"
        "- Сложные вопросы раскладывай на несколько вызовов инструментов (описать → отфильтровать → сгруппировать).\n"
        "- Если файл не CSV (например, XLSX), скажи об этом и попроси CSV-версию.\n"
        "- Произвольный код не исполняется: доступных фиксированных инструментов достаточно."
    ),
    tools=[load_csv, describe_columns, filter_rows, group_by],
    connectors=[agents.connector(_DRIVE_CONNECTOR)],
)


if __name__ == "__main__":
    import asyncio

    from tests.check import run

    asyncio.run(run())
