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

import datetime as dt

import pandas as pd
import pyarrow as pa
import pytest
from dateutil.relativedelta import relativedelta

import bigframes.bigquery as bbq
import bigframes.core.col
import bigframes.core.compile.sqlglot.testing
import bigframes.core.expression as ex
import bigframes.dtypes
from bigframes.core.compile.sqlglot import sql


def test_date_trunc_with_string():
    # User can pass string
    res_str = bbq.date_trunc(dt.date(2025, 1, 15), "MONTH")
    assert isinstance(res_str, bigframes.core.col.Expression)
    sql_str = bigframes.core.compile.sqlglot.testing.to_sql_snippet(res_str)
    assert sql_str == "DATE_TRUNC(CAST('2025-01-15' AS DATE), MONTH)"

    # Lowercase string converts to uppercase date part identifier
    res_lower = bbq.date_trunc(dt.date(2025, 1, 15), "month")  # type: ignore[arg-type]
    assert isinstance(res_lower, bigframes.core.col.Expression)
    sql_lower = bigframes.core.compile.sqlglot.testing.to_sql_snippet(res_lower)
    assert sql_lower == "DATE_TRUNC(CAST('2025-01-15' AS DATE), MONTH)"


def test_date_diff_with_string():
    res_str = bbq.date_diff(dt.date(2025, 2, 1), dt.date(2025, 1, 1), "DAY")
    assert isinstance(res_str, bigframes.core.col.Expression)
    sql_str = bigframes.core.compile.sqlglot.testing.to_sql_snippet(res_str)
    assert (
        sql_str
        == "DATE_DIFF(CAST('2025-02-01' AS DATE), CAST('2025-01-01' AS DATE), DAY)"
    )


def test_date_add_with_relativedelta():
    res = bbq.date_add(dt.date(2025, 1, 1), relativedelta(days=5))
    assert isinstance(res, bigframes.core.col.Expression)
    sql = bigframes.core.compile.sqlglot.testing.to_sql_snippet(res)
    assert sql == "DATE_ADD(CAST('2025-01-01' AS DATE), INTERVAL 5 DAY)"

    res_months = bbq.date_add(dt.date(2025, 1, 1), relativedelta(months=2))
    assert isinstance(res_months, bigframes.core.col.Expression)
    sql_months = bigframes.core.compile.sqlglot.testing.to_sql_snippet(res_months)
    assert sql_months == "DATE_ADD(CAST('2025-01-01' AS DATE), INTERVAL 2 MONTH)"


def test_generate_date_array_with_interval():
    res = bbq.generate_date_array(
        dt.date(2025, 1, 1),
        dt.date(2025, 1, 10),
        relativedelta(days=2),
    )
    assert isinstance(res, bigframes.core.col.Expression)
    sql = bigframes.core.compile.sqlglot.testing.to_sql_snippet(res)
    assert (
        sql
        == "GENERATE_DATE_ARRAY(CAST('2025-01-01' AS DATE), CAST('2025-01-10' AS DATE), INTERVAL 2 DAY)"
    )


def test_extract_with_date_part():
    res = bbq.extract(dt.date(2025, 1, 1), "DAY")
    assert isinstance(res, bigframes.core.col.Expression)
    sql = bigframes.core.compile.sqlglot.testing.to_sql_snippet(res)
    assert sql == "EXTRACT(DAY FROM CAST('2025-01-01' AS DATE))"


def test_extract_with_timezone():
    res = bbq.extract(dt.datetime(2025, 1, 1, 12, 0), "HOUR", "UTC")  # noqa: DTZ001
    assert isinstance(res, bigframes.core.col.Expression)
    sql = bigframes.core.compile.sqlglot.testing.to_sql_snippet(res)
    assert (
        sql
        == "EXTRACT(HOUR FROM TIMESTAMP(DATETIME(CAST('2025-01-01T12:00:00' AS DATETIME), 'UTC')))"
    )


def test_extract_rewrite_rule():
    import bigframes.operations as ops
    from bigframes.core import rewrite

    res = bbq.extract(dt.date(2025, 1, 1), "DAY")
    assert isinstance(res, bigframes.core.col.Expression)
    assert isinstance(res._value, ex.OpExpression)
    # Before rewrite, op is the GoogleSqlScalarOp
    assert isinstance(res._value.op, ops.GoogleSqlScalarOp)
    assert res._value.op.sql_name == "EXTRACT"

    # After rewrite, op is ExtractOp
    lowered = rewrite.lower_googlesql_expr(res._value)
    assert isinstance(lowered, ex.OpExpression)
    assert isinstance(lowered.op, ops.ExtractOp)


def test_extract_op_direct_compilation():
    import bigframes.core.col
    import bigframes.core.expression as ex
    import bigframes.operations as ops
    from bigframes import dtypes

    date_expr = ex.const(dt.date(2025, 1, 1), dtypes.DATE_DTYPE)
    # ExtractOp with part attribute
    op_expr = ops.ExtractOp(part="YEAR").as_expr(date_expr)
    sql = bigframes.core.compile.sqlglot.testing.to_sql_snippet(
        bigframes.core.col.Expression(op_expr)
    )
    assert sql == "EXTRACT(YEAR FROM CAST('2025-01-01' AS DATE))"


def test_interval_dtype():
    # TODO(b/299294296): Add full INTERVAL support.
    assert bigframes.dtypes.INTERVAL_DTYPE == pd.ArrowDtype(
        pa.month_day_nano_interval()
    )
    assert bigframes.dtypes.is_interval_like(bigframes.dtypes.INTERVAL_DTYPE)
    assert not bigframes.dtypes.is_interval_like(bigframes.dtypes.INT_DTYPE)


def test_literal_with_date_part_dtype():
    # Note that these aren't quoted like strings in the SQL output.
    assert sql.to_sql(sql.literal("HOUR", dtype="date_part")) == "HOUR"
    assert sql.to_sql(sql.literal("DAY", dtype="date_part")) == "DAY"
    assert sql.to_sql(sql.literal("day", dtype="date_part")) == "DAY"


def test_literal_with_interval_dtype():
    res_rd = sql.literal(relativedelta(days=5), dtype=bigframes.dtypes.INTERVAL_DTYPE)
    assert sql.to_sql(res_rd) == "INTERVAL 5 DAY"

    res_rd_str = sql.literal(relativedelta(days=5), dtype="interval")
    assert sql.to_sql(res_rd_str) == "INTERVAL 5 DAY"

    res_rd_nodtype = sql.literal(relativedelta(days=5))
    assert sql.to_sql(res_rd_nodtype) == "INTERVAL 5 DAY"

    mdn = pa.scalar((2, 5, 1000), type=pa.month_day_nano_interval())
    res_mdn = sql.literal(mdn, dtype=bigframes.dtypes.INTERVAL_DTYPE)
    assert (
        sql.to_sql(res_mdn)
        == "INTERVAL 2 MONTH + INTERVAL 5 DAY + INTERVAL 1 MICROSECOND"
    )

    res_mdn_nodtype = sql.literal(mdn)
    assert (
        sql.to_sql(res_mdn_nodtype)
        == "INTERVAL 2 MONTH + INTERVAL 5 DAY + INTERVAL 1 MICROSECOND"
    )


def test_literal_invalid_dtype_raises():
    with pytest.raises(TypeError, match="Expected str for date_part"):
        sql.literal(123, dtype="date_part")

    with pytest.raises(TypeError, match="Expected relativedelta or MonthDayNano"):
        sql.literal(123, dtype=bigframes.dtypes.INTERVAL_DTYPE)
