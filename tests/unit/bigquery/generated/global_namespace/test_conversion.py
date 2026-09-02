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
# This file was generated from: scripts/data/sql-functions/global_namespace/conversion.yaml
# by the script: scripts/generate_bigframes_bigquery.py

import datetime as dt

import pytest

import bigframes.bigquery as bbq
import bigframes.core.col
import bigframes.core.compile.sqlglot.testing
import bigframes.core.expression as ex
import bigframes.operations.googlesql.global_namespace.conversion as conversion_op

pytest.importorskip("pytest_snapshot")


def test_bool__expression(snapshot):
    # Call the function with literals or col() expressions
    result = bbq.bool_(
        "test",
    )

    # Verify result is a col Expression
    assert isinstance(result, bigframes.core.col.Expression)

    # Verify the internal expression structure
    expr = result._value
    assert isinstance(expr, ex.OpExpression)
    assert expr.op == conversion_op._BOOL_OP

    # Verify arguments
    assert len(expr.inputs) == 1
    assert isinstance(expr.inputs[0], ex.ScalarConstantExpression)

    snippet = bigframes.core.compile.sqlglot.testing.to_sql_snippet(result)
    snapshot.assert_match(snippet + "\n", "out.sql")


def test_double_expression(snapshot):
    # Call the function with literals or col() expressions
    result = bbq.double(
        "test",
        "test",
    )

    # Verify result is a col Expression
    assert isinstance(result, bigframes.core.col.Expression)

    # Verify the internal expression structure
    expr = result._value
    assert isinstance(expr, ex.OpExpression)
    assert expr.op == conversion_op._DOUBLE_OP

    # Verify arguments
    assert len(expr.inputs) == 2
    assert isinstance(expr.inputs[0], ex.ScalarConstantExpression)
    assert isinstance(expr.inputs[1], ex.ScalarConstantExpression)

    snippet = bigframes.core.compile.sqlglot.testing.to_sql_snippet(result)
    snapshot.assert_match(snippet + "\n", "out.sql")


def test_float64_expression(snapshot):
    # Call the function with literals or col() expressions
    result = bbq.float64(
        "test",
        "test",
    )

    # Verify result is a col Expression
    assert isinstance(result, bigframes.core.col.Expression)

    # Verify the internal expression structure
    expr = result._value
    assert isinstance(expr, ex.OpExpression)
    assert expr.op == conversion_op._FLOAT64_OP

    # Verify arguments
    assert len(expr.inputs) == 2
    assert isinstance(expr.inputs[0], ex.ScalarConstantExpression)
    assert isinstance(expr.inputs[1], ex.ScalarConstantExpression)

    snippet = bigframes.core.compile.sqlglot.testing.to_sql_snippet(result)
    snapshot.assert_match(snippet + "\n", "out.sql")


def test_int64_expression(snapshot):
    # Call the function with literals or col() expressions
    result = bbq.int64(
        "test",
    )

    # Verify result is a col Expression
    assert isinstance(result, bigframes.core.col.Expression)

    # Verify the internal expression structure
    expr = result._value
    assert isinstance(expr, ex.OpExpression)
    assert expr.op == conversion_op._INT64_OP

    # Verify arguments
    assert len(expr.inputs) == 1
    assert isinstance(expr.inputs[0], ex.ScalarConstantExpression)

    snippet = bigframes.core.compile.sqlglot.testing.to_sql_snippet(result)
    snapshot.assert_match(snippet + "\n", "out.sql")


def test_parse_bignumeric_expression(snapshot):
    # Call the function with literals or col() expressions
    result = bbq.parse_bignumeric(
        "test",
    )

    # Verify result is a col Expression
    assert isinstance(result, bigframes.core.col.Expression)

    # Verify the internal expression structure
    expr = result._value
    assert isinstance(expr, ex.OpExpression)
    assert expr.op == conversion_op._PARSE_BIGNUMERIC_OP

    # Verify arguments
    assert len(expr.inputs) == 1
    assert isinstance(expr.inputs[0], ex.ScalarConstantExpression)

    snippet = bigframes.core.compile.sqlglot.testing.to_sql_snippet(result)
    snapshot.assert_match(snippet + "\n", "out.sql")


def test_parse_numeric_expression(snapshot):
    # Call the function with literals or col() expressions
    result = bbq.parse_numeric(
        "test",
    )

    # Verify result is a col Expression
    assert isinstance(result, bigframes.core.col.Expression)

    # Verify the internal expression structure
    expr = result._value
    assert isinstance(expr, ex.OpExpression)
    assert expr.op == conversion_op._PARSE_NUMERIC_OP

    # Verify arguments
    assert len(expr.inputs) == 1
    assert isinstance(expr.inputs[0], ex.ScalarConstantExpression)

    snippet = bigframes.core.compile.sqlglot.testing.to_sql_snippet(result)
    snapshot.assert_match(snippet + "\n", "out.sql")


def test_string_expression(snapshot):
    # Call the function with literals or col() expressions
    result = bbq.string(
        dt.date(2025, 1, 1),
        "test",
    )

    # Verify result is a col Expression
    assert isinstance(result, bigframes.core.col.Expression)

    # Verify the internal expression structure
    expr = result._value
    assert isinstance(expr, ex.OpExpression)
    assert expr.op == conversion_op._STRING_OP

    # Verify arguments
    assert len(expr.inputs) == 2
    assert isinstance(expr.inputs[0], ex.ScalarConstantExpression)
    assert isinstance(expr.inputs[1], ex.ScalarConstantExpression)

    snippet = bigframes.core.compile.sqlglot.testing.to_sql_snippet(result)
    snapshot.assert_match(snippet + "\n", "out.sql")
