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
# This file was generated from: scripts/data/sql-functions/global_namespace/date.yaml
# by the script: scripts/generate_bigframes_bigquery.py

import datetime as dt

import pytest

import bigframes.bigquery as bbq
import bigframes.core.col
import bigframes.core.compile.sqlglot.testing
import bigframes.core.expression as ex
import bigframes.operations.googlesql.global_namespace.date as date_op

pytest.importorskip("pytest_snapshot")


def test_current_date_expression(snapshot):
    # Call the function with literals or col() expressions
    result = bbq.current_date(
        "UTC",
    )

    # Verify result is a col Expression
    assert isinstance(result, bigframes.core.col.Expression)

    # Verify the internal expression structure
    expr = result._value
    assert isinstance(expr, ex.OpExpression)
    assert expr.op == date_op._CURRENT_DATE_OP

    # Verify arguments
    assert len(expr.inputs) == 1
    assert isinstance(expr.inputs[0], ex.ScalarConstantExpression)

    snippet = bigframes.core.compile.sqlglot.testing.to_sql_snippet(result)
    snapshot.assert_match(snippet + "\n", "out.sql")


def test_date_expression(snapshot):
    # Call the function with literals or col() expressions
    result = bbq.date(
        dt.date(2025, 1, 1),
        "UTC",
        2025,
        1,
        1,
    )

    # Verify result is a col Expression
    assert isinstance(result, bigframes.core.col.Expression)

    # Verify the internal expression structure
    expr = result._value
    assert isinstance(expr, ex.OpExpression)
    assert expr.op == date_op._DATE_OP

    # Verify arguments
    assert len(expr.inputs) == 5
    assert isinstance(expr.inputs[0], ex.ScalarConstantExpression)
    assert isinstance(expr.inputs[1], ex.ScalarConstantExpression)
    assert isinstance(expr.inputs[2], ex.ScalarConstantExpression)
    assert isinstance(expr.inputs[3], ex.ScalarConstantExpression)
    assert isinstance(expr.inputs[4], ex.ScalarConstantExpression)

    snippet = bigframes.core.compile.sqlglot.testing.to_sql_snippet(result)
    snapshot.assert_match(snippet + "\n", "out.sql")


def test_date_add_expression(snapshot):
    # Call the function with literals or col() expressions
    result = bbq.date_add(
        dt.date(2025, 1, 1),
        1,
        "DAY",
    )

    # Verify result is a col Expression
    assert isinstance(result, bigframes.core.col.Expression)

    # Verify the internal expression structure
    expr = result._value
    assert isinstance(expr, ex.OpExpression)
    assert expr.op == date_op._DATE_ADD_OP

    # Verify arguments
    assert len(expr.inputs) == 3
    assert isinstance(expr.inputs[0], ex.ScalarConstantExpression)
    assert isinstance(expr.inputs[1], ex.ScalarConstantExpression)
    assert isinstance(expr.inputs[2], ex.ScalarConstantExpression)

    snippet = bigframes.core.compile.sqlglot.testing.to_sql_snippet(result)
    snapshot.assert_match(snippet + "\n", "out.sql")


def test_date_diff_expression(snapshot):
    # Call the function with literals or col() expressions
    result = bbq.date_diff(
        dt.date(2025, 1, 1),
        dt.date(2025, 1, 1),
        "DAY",
    )

    # Verify result is a col Expression
    assert isinstance(result, bigframes.core.col.Expression)

    # Verify the internal expression structure
    expr = result._value
    assert isinstance(expr, ex.OpExpression)
    assert expr.op == date_op._DATE_DIFF_OP

    # Verify arguments
    assert len(expr.inputs) == 3
    assert isinstance(expr.inputs[0], ex.ScalarConstantExpression)
    assert isinstance(expr.inputs[1], ex.ScalarConstantExpression)
    assert isinstance(expr.inputs[2], ex.ScalarConstantExpression)

    snippet = bigframes.core.compile.sqlglot.testing.to_sql_snippet(result)
    snapshot.assert_match(snippet + "\n", "out.sql")


def test_date_from_unix_date_expression(snapshot):
    # Call the function with literals or col() expressions
    result = bbq.date_from_unix_date(
        1,
    )

    # Verify result is a col Expression
    assert isinstance(result, bigframes.core.col.Expression)

    # Verify the internal expression structure
    expr = result._value
    assert isinstance(expr, ex.OpExpression)
    assert expr.op == date_op._DATE_FROM_UNIX_DATE_OP

    # Verify arguments
    assert len(expr.inputs) == 1
    assert isinstance(expr.inputs[0], ex.ScalarConstantExpression)

    snippet = bigframes.core.compile.sqlglot.testing.to_sql_snippet(result)
    snapshot.assert_match(snippet + "\n", "out.sql")


def test_date_sub_expression(snapshot):
    # Call the function with literals or col() expressions
    result = bbq.date_sub(
        dt.date(2025, 1, 1),
        1,
        "DAY",
    )

    # Verify result is a col Expression
    assert isinstance(result, bigframes.core.col.Expression)

    # Verify the internal expression structure
    expr = result._value
    assert isinstance(expr, ex.OpExpression)
    assert expr.op == date_op._DATE_SUB_OP

    # Verify arguments
    assert len(expr.inputs) == 3
    assert isinstance(expr.inputs[0], ex.ScalarConstantExpression)
    assert isinstance(expr.inputs[1], ex.ScalarConstantExpression)
    assert isinstance(expr.inputs[2], ex.ScalarConstantExpression)

    snippet = bigframes.core.compile.sqlglot.testing.to_sql_snippet(result)
    snapshot.assert_match(snippet + "\n", "out.sql")


def test_date_trunc_expression(snapshot):
    # Call the function with literals or col() expressions
    result = bbq.date_trunc(
        dt.date(2025, 1, 1),
        "DAY",
    )

    # Verify result is a col Expression
    assert isinstance(result, bigframes.core.col.Expression)

    # Verify the internal expression structure
    expr = result._value
    assert isinstance(expr, ex.OpExpression)
    assert expr.op == date_op._DATE_TRUNC_OP

    # Verify arguments
    assert len(expr.inputs) == 2
    assert isinstance(expr.inputs[0], ex.ScalarConstantExpression)
    assert isinstance(expr.inputs[1], ex.ScalarConstantExpression)

    snippet = bigframes.core.compile.sqlglot.testing.to_sql_snippet(result)
    snapshot.assert_match(snippet + "\n", "out.sql")


def test_format_date_expression(snapshot):
    # Call the function with literals or col() expressions
    result = bbq.format_date(
        "%Y-%m-%d",
        dt.date(2025, 1, 1),
    )

    # Verify result is a col Expression
    assert isinstance(result, bigframes.core.col.Expression)

    # Verify the internal expression structure
    expr = result._value
    assert isinstance(expr, ex.OpExpression)
    assert expr.op == date_op._FORMAT_DATE_OP

    # Verify arguments
    assert len(expr.inputs) == 2
    assert isinstance(expr.inputs[0], ex.ScalarConstantExpression)
    assert isinstance(expr.inputs[1], ex.ScalarConstantExpression)

    snippet = bigframes.core.compile.sqlglot.testing.to_sql_snippet(result)
    snapshot.assert_match(snippet + "\n", "out.sql")


def test_last_day_expression(snapshot):
    # Call the function with literals or col() expressions
    result = bbq.last_day(
        dt.date(2025, 1, 1),
        "DAY",
    )

    # Verify result is a col Expression
    assert isinstance(result, bigframes.core.col.Expression)

    # Verify the internal expression structure
    expr = result._value
    assert isinstance(expr, ex.OpExpression)
    assert expr.op == date_op._LAST_DAY_OP

    # Verify arguments
    assert len(expr.inputs) == 2
    assert isinstance(expr.inputs[0], ex.ScalarConstantExpression)
    assert isinstance(expr.inputs[1], ex.ScalarConstantExpression)

    snippet = bigframes.core.compile.sqlglot.testing.to_sql_snippet(result)
    snapshot.assert_match(snippet + "\n", "out.sql")


def test_parse_date_expression(snapshot):
    # Call the function with literals or col() expressions
    result = bbq.parse_date(
        "%Y-%m-%d",
        "2025-01-01",
    )

    # Verify result is a col Expression
    assert isinstance(result, bigframes.core.col.Expression)

    # Verify the internal expression structure
    expr = result._value
    assert isinstance(expr, ex.OpExpression)
    assert expr.op == date_op._PARSE_DATE_OP

    # Verify arguments
    assert len(expr.inputs) == 2
    assert isinstance(expr.inputs[0], ex.ScalarConstantExpression)
    assert isinstance(expr.inputs[1], ex.ScalarConstantExpression)

    snippet = bigframes.core.compile.sqlglot.testing.to_sql_snippet(result)
    snapshot.assert_match(snippet + "\n", "out.sql")


def test_unix_date_expression(snapshot):
    # Call the function with literals or col() expressions
    result = bbq.unix_date(
        dt.date(2025, 1, 1),
    )

    # Verify result is a col Expression
    assert isinstance(result, bigframes.core.col.Expression)

    # Verify the internal expression structure
    expr = result._value
    assert isinstance(expr, ex.OpExpression)
    assert expr.op == date_op._UNIX_DATE_OP

    # Verify arguments
    assert len(expr.inputs) == 1
    assert isinstance(expr.inputs[0], ex.ScalarConstantExpression)

    snippet = bigframes.core.compile.sqlglot.testing.to_sql_snippet(result)
    snapshot.assert_match(snippet + "\n", "out.sql")
