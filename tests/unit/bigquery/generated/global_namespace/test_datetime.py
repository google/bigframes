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
#
# DO NOT MODIFY THIS FILE DIRECTLY.
# This file was generated from: scripts/data/sql-functions/global_namespace/datetime.yaml
# by the script: scripts/generate_bigframes_bigquery.py

import datetime as dt

import pytest
from dateutil import relativedelta

import bigframes.bigquery as bbq
import bigframes.core.col
import bigframes.core.compile.sqlglot.testing
import bigframes.core.expression as ex
import bigframes.operations.googlesql.global_namespace.datetime as datetime_op

pytest.importorskip("pytest_snapshot")


def test_current_datetime_expression(snapshot):
    # Call the function with literals or col() expressions
    result = bbq.current_datetime(
        "UTC",
    )

    # Verify result is a col Expression
    assert isinstance(result, bigframes.core.col.Expression)

    # Verify the internal expression structure
    expr = result._value
    assert isinstance(expr, ex.OpExpression)
    assert expr.op == datetime_op._CURRENT_DATETIME_OP

    # Verify arguments
    assert len(expr.inputs) == 1
    assert isinstance(expr.inputs[0], ex.ScalarConstantExpression)

    snippet = bigframes.core.compile.sqlglot.testing.to_sql_snippet(result)
    snapshot.assert_match(snippet + "\n", "out.sql")


def test_datetime_expression(snapshot):
    # Call the function with literals or col() expressions
    result = bbq.datetime(
        dt.date(2025, 1, 1),
        "UTC",
        dt.time(12, 0, 0),
        2025,
        1,
        1,
        12,
        0,
        0,
    )

    # Verify result is a col Expression
    assert isinstance(result, bigframes.core.col.Expression)

    # Verify the internal expression structure
    expr = result._value
    assert isinstance(expr, ex.OpExpression)
    assert expr.op == datetime_op._DATETIME_OP

    # Verify arguments
    assert len(expr.inputs) == 9
    assert isinstance(expr.inputs[0], ex.ScalarConstantExpression)
    assert isinstance(expr.inputs[1], ex.ScalarConstantExpression)
    assert isinstance(expr.inputs[2], ex.ScalarConstantExpression)
    assert isinstance(expr.inputs[3], ex.ScalarConstantExpression)
    assert isinstance(expr.inputs[4], ex.ScalarConstantExpression)
    assert isinstance(expr.inputs[5], ex.ScalarConstantExpression)
    assert isinstance(expr.inputs[6], ex.ScalarConstantExpression)
    assert isinstance(expr.inputs[7], ex.ScalarConstantExpression)
    assert isinstance(expr.inputs[8], ex.ScalarConstantExpression)

    snippet = bigframes.core.compile.sqlglot.testing.to_sql_snippet(result)
    snapshot.assert_match(snippet + "\n", "out.sql")


def test_datetime_add_expression(snapshot):
    # Call the function with literals or col() expressions
    result = bbq.datetime_add(
        dt.datetime(2025, 1, 1, 12, 0, 0),
        relativedelta.relativedelta(days=1),
    )

    # Verify result is a col Expression
    assert isinstance(result, bigframes.core.col.Expression)

    # Verify the internal expression structure
    expr = result._value
    assert isinstance(expr, ex.OpExpression)
    assert expr.op == datetime_op._DATETIME_ADD_OP

    # Verify arguments
    assert len(expr.inputs) == 2
    assert isinstance(expr.inputs[0], ex.ScalarConstantExpression)
    assert isinstance(expr.inputs[1], ex.ScalarConstantExpression)

    snippet = bigframes.core.compile.sqlglot.testing.to_sql_snippet(result)
    snapshot.assert_match(snippet + "\n", "out.sql")


def test_datetime_diff_expression(snapshot):
    # Call the function with literals or col() expressions
    result = bbq.datetime_diff(
        dt.datetime(2025, 1, 1, 12, 0, 0),
        dt.datetime(2025, 1, 1, 12, 0, 0),
        "DAY",
    )

    # Verify result is a col Expression
    assert isinstance(result, bigframes.core.col.Expression)

    # Verify the internal expression structure
    expr = result._value
    assert isinstance(expr, ex.OpExpression)
    assert expr.op == datetime_op._DATETIME_DIFF_OP

    # Verify arguments
    assert len(expr.inputs) == 3
    assert isinstance(expr.inputs[0], ex.ScalarConstantExpression)
    assert isinstance(expr.inputs[1], ex.ScalarConstantExpression)
    assert isinstance(expr.inputs[2], ex.ScalarConstantExpression)

    snippet = bigframes.core.compile.sqlglot.testing.to_sql_snippet(result)
    snapshot.assert_match(snippet + "\n", "out.sql")


def test_datetime_sub_expression(snapshot):
    # Call the function with literals or col() expressions
    result = bbq.datetime_sub(
        dt.datetime(2025, 1, 1, 12, 0, 0),
        relativedelta.relativedelta(days=1),
    )

    # Verify result is a col Expression
    assert isinstance(result, bigframes.core.col.Expression)

    # Verify the internal expression structure
    expr = result._value
    assert isinstance(expr, ex.OpExpression)
    assert expr.op == datetime_op._DATETIME_SUB_OP

    # Verify arguments
    assert len(expr.inputs) == 2
    assert isinstance(expr.inputs[0], ex.ScalarConstantExpression)
    assert isinstance(expr.inputs[1], ex.ScalarConstantExpression)

    snippet = bigframes.core.compile.sqlglot.testing.to_sql_snippet(result)
    snapshot.assert_match(snippet + "\n", "out.sql")


def test_datetime_trunc_expression(snapshot):
    # Call the function with literals or col() expressions
    result = bbq.datetime_trunc(
        dt.datetime(2025, 1, 1, 12, 0, 0, tzinfo=dt.timezone.utc),
        "DAY",
        "UTC",
    )

    # Verify result is a col Expression
    assert isinstance(result, bigframes.core.col.Expression)

    # Verify the internal expression structure
    expr = result._value
    assert isinstance(expr, ex.OpExpression)
    assert expr.op == datetime_op._DATETIME_TRUNC_OP

    # Verify arguments
    assert len(expr.inputs) == 3
    assert isinstance(expr.inputs[0], ex.ScalarConstantExpression)
    assert isinstance(expr.inputs[1], ex.ScalarConstantExpression)
    assert isinstance(expr.inputs[2], ex.ScalarConstantExpression)

    snippet = bigframes.core.compile.sqlglot.testing.to_sql_snippet(result)
    snapshot.assert_match(snippet + "\n", "out.sql")


def test_format_datetime_expression(snapshot):
    # Call the function with literals or col() expressions
    result = bbq.format_datetime(
        "%Y-%m-%d %H:%M:%S",
        dt.datetime(2025, 1, 1, 12, 0, 0),
    )

    # Verify result is a col Expression
    assert isinstance(result, bigframes.core.col.Expression)

    # Verify the internal expression structure
    expr = result._value
    assert isinstance(expr, ex.OpExpression)
    assert expr.op == datetime_op._FORMAT_DATETIME_OP

    # Verify arguments
    assert len(expr.inputs) == 2
    assert isinstance(expr.inputs[0], ex.ScalarConstantExpression)
    assert isinstance(expr.inputs[1], ex.ScalarConstantExpression)

    snippet = bigframes.core.compile.sqlglot.testing.to_sql_snippet(result)
    snapshot.assert_match(snippet + "\n", "out.sql")


def test_parse_datetime_expression(snapshot):
    # Call the function with literals or col() expressions
    result = bbq.parse_datetime(
        "%Y-%m-%d %H:%M:%S",
        "2025-01-01 12:00:00",
    )

    # Verify result is a col Expression
    assert isinstance(result, bigframes.core.col.Expression)

    # Verify the internal expression structure
    expr = result._value
    assert isinstance(expr, ex.OpExpression)
    assert expr.op == datetime_op._PARSE_DATETIME_OP

    # Verify arguments
    assert len(expr.inputs) == 2
    assert isinstance(expr.inputs[0], ex.ScalarConstantExpression)
    assert isinstance(expr.inputs[1], ex.ScalarConstantExpression)

    snippet = bigframes.core.compile.sqlglot.testing.to_sql_snippet(result)
    snapshot.assert_match(snippet + "\n", "out.sql")
