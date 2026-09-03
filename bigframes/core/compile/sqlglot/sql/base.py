# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import datetime
import typing

import bigframes_vendored.sqlglot as sg
import bigframes_vendored.sqlglot.expressions as sge
import numpy as np
import pandas as pd
import pandas.api.types
import pyarrow as pa
from dateutil.relativedelta import relativedelta
from google.cloud import bigquery

import bigframes.core.compile.sqlglot.sqlglot_types as sgt
from bigframes import dtypes
from bigframes.core import utils
from bigframes.core.compile.sqlglot.expressions import constants

# shapely.wkt.dumps was moved to shapely.io.to_wkt in 2.0.
try:
    from shapely.io import to_wkt  # type: ignore
except ImportError:
    from shapely.wkt import dumps  # type: ignore

    to_wkt = dumps


QUOTED: bool = True
"""Whether to quote identifiers in the generated SQL."""

PRETTY: bool = True
"""Whether to pretty-print the generated SQL."""

DIALECT = sg.dialects.bigquery.BigQuery
"""The SQL dialect used for generation."""


def to_sql(expr: sge.Expression) -> str:
    """Generate SQL string from the given expression."""
    return expr.sql(dialect=DIALECT, pretty=PRETTY)


def identifier(id: str) -> sge.Identifier:
    """Return a string representing column reference in a SQL."""
    return sge.to_identifier(id, quoted=QUOTED)


def _unwrap_scalar(value: typing.Any) -> typing.Any:
    """Unwraps PyArrow scalars and NumPy generic types to native Python objects."""
    if isinstance(value, pa.Scalar):
        return value.as_py()
    if isinstance(value, np.datetime64):
        # Avoid calling .item() on np.datetime64 because sub-microsecond np.datetime64 objects
        # are converted to integer epoch.
        return pd.Timestamp(value).to_pydatetime()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _relativedelta_to_sqlglot(rd: relativedelta) -> sge.Expression:
    field_units = [
        ("YEAR", rd.years),
        ("MONTH", rd.months),
        ("DAY", rd.days),
        ("HOUR", rd.hours),
        ("MINUTE", rd.minutes),
        ("SECOND", rd.seconds),
        ("MICROSECOND", rd.microseconds),
    ]
    non_zero = [(unit, val) for unit, val in field_units if val != 0]
    if not non_zero:
        return sge.Interval(this=sge.convert(0), unit=sge.var("DAY"))
    if len(non_zero) == 1:
        unit, val = non_zero[0]
        return sge.Interval(this=sge.convert(val), unit=sge.var(unit))
    expr: sge.Expression | None = None
    for unit, val in non_zero:
        interval_expr = sge.Interval(this=sge.convert(val), unit=sge.var(unit))
        if expr is None:
            expr = interval_expr
        else:
            expr = sge.Add(this=expr, expression=interval_expr)
    assert expr is not None
    return expr


def _month_day_nano_to_sqlglot(mdn: pa.MonthDayNano) -> sge.Expression:
    field_units = [
        ("MONTH", mdn.months),
        ("DAY", mdn.days),
    ]
    ns = mdn.nanoseconds
    if ns != 0:
        if ns % 1_000_000_000 == 0:
            field_units.append(("SECOND", ns // 1_000_000_000))
        elif ns % 1_000_000 == 0:
            field_units.append(("MILLISECOND", ns // 1_000_000))
        elif ns % 1_000 == 0:
            field_units.append(("MICROSECOND", ns // 1_000))
        else:
            field_units.append(("MICROSECOND", ns // 1_000))
    non_zero = [(unit, val) for unit, val in field_units if val != 0]
    if not non_zero:
        return sge.Interval(this=sge.convert(0), unit=sge.var("DAY"))
    if len(non_zero) == 1:
        unit, val = non_zero[0]
        return sge.Interval(this=sge.convert(val), unit=sge.var(unit))
    expr: sge.Expression | None = None
    for unit, val in non_zero:
        interval_expr = sge.Interval(this=sge.convert(val), unit=sge.var(unit))
        if expr is None:
            expr = interval_expr
        else:
            expr = sge.Add(this=expr, expression=interval_expr)
    assert expr is not None
    return expr


def _literal_with_known_dtype(
    value: typing.Any,
    *,
    dtype: dtypes.Dtype | str,
) -> sge.Expression:
    """Return a string representing a literal value in SQL."""
    if isinstance(dtype, str):
        if dtype == "date_part":
            if isinstance(value, str):
                return sge.var(value.upper())
            raise TypeError(
                f"Expected str for date_part, got {type(value).__name__}: {value}"
            )
        elif dtype == "interval":
            dtype = dtypes.INTERVAL_DTYPE
        else:
            raise TypeError(f"got unexpeced dtype {dtype}")

    sqlglot_type = sgt.from_bigframes_dtype(dtype) if dtype else None
    if sqlglot_type is None:
        if not pd.isna(value):
            raise ValueError(f"Cannot infer SQLGlot type from None dtype: {value}")
        return sge.Null()

    # month_day_nano not currently supported in pandas, so we need to check this
    # before calling any pandas api types functions.
    if dtype == dtypes.INTERVAL_DTYPE:
        if isinstance(value, pa.Scalar) and hasattr(value, "as_py"):
            value = value.as_py()
        if isinstance(value, pa.MonthDayNano):
            return _month_day_nano_to_sqlglot(value)
        if isinstance(value, relativedelta):
            return _relativedelta_to_sqlglot(value)
        if isinstance(value, sge.Interval):
            return value
        raise TypeError(
            f"Expected relativedelta or MonthDayNano for interval, got {type(value).__name__}: {value}"
        )
    elif dtypes.is_struct_like(dtype):
        items = [
            literal(value=value[field_name], dtype=field_dtype).as_(
                field_name, quoted=True
            )
            for field_name, field_dtype in dtypes.get_struct_fields(dtype).items()
        ]
        return sge.Struct.from_arg_list(items)
    elif dtypes.is_array_like(dtype):
        value_type = dtypes.get_array_inner_type(dtype)
        values = sge.Array(
            expressions=[literal(value=v, dtype=value_type) for v in value]
        )
        return values if len(value) > 0 else cast(values, sqlglot_type)
    elif dtype == dtypes.FLOAT_DTYPE:
        if pd.isna(value):
            if isinstance(value, (float, np.floating)) and np.isnan(value):
                return constants._NAN
            return cast(sge.Null(), sqlglot_type)
        if np.isinf(value):
            return constants._INF if value > 0 else constants._NEG_INF
        return sge.convert(value)
    elif dtype == dtypes.JSON_DTYPE:
        return sge.ParseJSON(this=sge.convert(str(value)))
    elif dtype == dtypes.BYTES_DTYPE:
        return cast(str(value), sqlglot_type)
    elif dtypes.is_time_like(dtype):
        value = _unwrap_scalar(value)
        if isinstance(value, str):
            return cast(sge.convert(value), sqlglot_type)
        elif isinstance(value, pd.Timestamp):
            value = value.to_pydatetime()
        return cast(sge.convert(value.isoformat()), sqlglot_type)
    elif dtypes.is_date_like(dtype):
        value = _unwrap_scalar(value)
        if isinstance(value, str):
            return cast(sge.convert(value), sqlglot_type)
        elif isinstance(value, datetime.datetime):
            # pd.Timestamp is a subclass of datatime.datetime.
            value = value.date()
        return cast(sge.convert(value.isoformat()), sqlglot_type)
    elif dtype in (dtypes.NUMERIC_DTYPE, dtypes.BIGNUMERIC_DTYPE):
        return cast(sge.convert(value), sqlglot_type)
    elif dtypes.is_geo_like(dtype):
        wkt = value if isinstance(value, str) else to_wkt(value)
        return sge.func("ST_GEOGFROMTEXT", sge.convert(wkt))
    elif dtype == dtypes.TIMEDELTA_DTYPE:
        return sge.convert(utils.timedelta_to_micros(value))
    elif dtype == dtypes.STRING_DTYPE:
        return sge.convert(str(_unwrap_scalar(value)))
    else:
        return sge.convert(_unwrap_scalar(value))


def _null_literal(*, dtype: dtypes.Dtype | None):
    """Return a NULL literal, optionally cast to a dtype."""
    sqlglot_type = sgt.from_bigframes_dtype(dtype) if dtype else None
    if sqlglot_type is None:
        return sge.Null()
    if str(sqlglot_type).upper() == "NULL":
        return sge.Null()
    return cast(sge.Null(), sqlglot_type)


def _is_null_value(
    value: typing.Any,
    dtype: dtypes.Dtype | str | None,
) -> bool:
    """Checks if the given literal value should be treated as SQL NULL."""
    if value is None:
        return True

    is_invalid_arrow_scalar = isinstance(value, pa.Scalar) and not value.is_valid
    if is_invalid_arrow_scalar:
        return True

    # month_day_nano not currently supported in pandas, so we need to
    # check this before calling any pandas api types functions.
    is_interval_dtype = dtype == dtypes.INTERVAL_DTYPE
    if is_interval_dtype:
        return False

    # If we get NaN for floats, then don't assume it's NULL.
    is_float_dtype = dtype is not None and pandas.api.types.is_float_dtype(dtype)
    is_list_like = pandas.api.types.is_list_like(value)
    if is_float_dtype or is_list_like:
        return False

    return bool(pd.isna(value))


def literal(
    value: typing.Any,
    dtype: dtypes.Dtype | str | None = None,
) -> sge.Expression:
    """Return a string representing a literal value in SQL."""
    if dtype is None:
        dtype = dtypes.infer_literal_type(value)

    if _is_null_value(value, dtype):
        return _null_literal(
            dtype=dtype if not isinstance(dtype, str) else None,
        )

    if dtype is not None:
        return _literal_with_known_dtype(value, dtype=dtype)

    if isinstance(value, pa.MonthDayNano):
        return _month_day_nano_to_sqlglot(value)
    elif isinstance(value, relativedelta):
        return _relativedelta_to_sqlglot(value)
    else:
        return sge.convert(_unwrap_scalar(value))


def cast(arg: typing.Any, to: str, safe: bool = False) -> sge.Cast | sge.TryCast:
    """Return a SQL expression that casts the given argument to the specified type."""
    if safe:
        return sge.TryCast(this=arg, to=to)
    else:
        return sge.Cast(this=arg, to=to)


def table(table: bigquery.TableReference) -> sge.Table:
    """Return a SQLGlot Table expression representing the given BigQuery table reference."""
    return sge.Table(
        this=sge.to_identifier(table.table_id, quoted=True),
        db=sge.to_identifier(table.dataset_id, quoted=True),
        catalog=sge.to_identifier(table.project, quoted=True),
    )


def is_null_literal(expr: sge.Expression) -> bool:
    """Checks if the given expression is a NULL literal."""
    if isinstance(expr, sge.Null):
        return True
    if isinstance(expr, sge.Cast) and isinstance(expr.this, sge.Null):
        return True
    return False
