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
# This file was generated from: scripts/data/sql-functions/global_namespace/array.yaml
# by the script: scripts/generate_bigframes_bigquery.py

import pytest

import bigframes.bigquery as bbq
import bigframes.core.col
import bigframes.core.compile.sqlglot.testing
import bigframes.core.expression as ex
import bigframes.operations.googlesql.global_namespace.array as array_op

pytest.importorskip("pytest_snapshot")


def test_array_concat_expression(snapshot):
    # Call the function with literals or col() expressions
    result = bbq.array_concat(
        [1, 2],
        [1, 2],
    )

    # Verify result is a col Expression
    assert isinstance(result, bigframes.core.col.Expression)

    # Verify the internal expression structure
    expr = result._value
    assert isinstance(expr, ex.OpExpression)
    assert expr.op == array_op._ARRAY_CONCAT_OP

    # Verify arguments
    assert len(expr.inputs) == 2
    assert isinstance(expr.inputs[0], ex.ScalarConstantExpression)
    assert isinstance(expr.inputs[1], ex.ScalarConstantExpression)

    snippet = bigframes.core.compile.sqlglot.testing.to_sql_snippet(result)
    snapshot.assert_match(snippet + "\n", "out.sql")


def test_array_first_expression(snapshot):
    # Call the function with literals or col() expressions
    result = bbq.array_first(
        [1, 2],
    )

    # Verify result is a col Expression
    assert isinstance(result, bigframes.core.col.Expression)

    # Verify the internal expression structure
    expr = result._value
    assert isinstance(expr, ex.OpExpression)
    assert expr.op == array_op._ARRAY_FIRST_OP

    # Verify arguments
    assert len(expr.inputs) == 1
    assert isinstance(expr.inputs[0], ex.ScalarConstantExpression)

    snippet = bigframes.core.compile.sqlglot.testing.to_sql_snippet(result)
    snapshot.assert_match(snippet + "\n", "out.sql")


def test_array_first_n_expression(snapshot):
    # Call the function with literals or col() expressions
    result = bbq.array_first_n(
        [1, 2],
        1,
    )

    # Verify result is a col Expression
    assert isinstance(result, bigframes.core.col.Expression)

    # Verify the internal expression structure
    expr = result._value
    assert isinstance(expr, ex.OpExpression)
    assert expr.op == array_op._ARRAY_FIRST_N_OP

    # Verify arguments
    assert len(expr.inputs) == 2
    assert isinstance(expr.inputs[0], ex.ScalarConstantExpression)
    assert isinstance(expr.inputs[1], ex.ScalarConstantExpression)

    snippet = bigframes.core.compile.sqlglot.testing.to_sql_snippet(result)
    snapshot.assert_match(snippet + "\n", "out.sql")


def test_array_includes_expression(snapshot):
    # Call the function with literals or col() expressions
    result = bbq.array_includes(
        [1, 2],
        1,
    )

    # Verify result is a col Expression
    assert isinstance(result, bigframes.core.col.Expression)

    # Verify the internal expression structure
    expr = result._value
    assert isinstance(expr, ex.OpExpression)
    assert expr.op == array_op._ARRAY_INCLUDES_OP

    # Verify arguments
    assert len(expr.inputs) == 2
    assert isinstance(expr.inputs[0], ex.ScalarConstantExpression)
    assert isinstance(expr.inputs[1], ex.ScalarConstantExpression)

    snippet = bigframes.core.compile.sqlglot.testing.to_sql_snippet(result)
    snapshot.assert_match(snippet + "\n", "out.sql")


def test_array_includes_all_expression(snapshot):
    # Call the function with literals or col() expressions
    result = bbq.array_includes_all(
        [1, 2],
        [1, 2],
    )

    # Verify result is a col Expression
    assert isinstance(result, bigframes.core.col.Expression)

    # Verify the internal expression structure
    expr = result._value
    assert isinstance(expr, ex.OpExpression)
    assert expr.op == array_op._ARRAY_INCLUDES_ALL_OP

    # Verify arguments
    assert len(expr.inputs) == 2
    assert isinstance(expr.inputs[0], ex.ScalarConstantExpression)
    assert isinstance(expr.inputs[1], ex.ScalarConstantExpression)

    snippet = bigframes.core.compile.sqlglot.testing.to_sql_snippet(result)
    snapshot.assert_match(snippet + "\n", "out.sql")


def test_array_includes_any_expression(snapshot):
    # Call the function with literals or col() expressions
    result = bbq.array_includes_any(
        [1, 2],
        [1, 2],
    )

    # Verify result is a col Expression
    assert isinstance(result, bigframes.core.col.Expression)

    # Verify the internal expression structure
    expr = result._value
    assert isinstance(expr, ex.OpExpression)
    assert expr.op == array_op._ARRAY_INCLUDES_ANY_OP

    # Verify arguments
    assert len(expr.inputs) == 2
    assert isinstance(expr.inputs[0], ex.ScalarConstantExpression)
    assert isinstance(expr.inputs[1], ex.ScalarConstantExpression)

    snippet = bigframes.core.compile.sqlglot.testing.to_sql_snippet(result)
    snapshot.assert_match(snippet + "\n", "out.sql")


def test_array_is_distinct_expression(snapshot):
    # Call the function with literals or col() expressions
    result = bbq.array_is_distinct(
        [1, 2],
    )

    # Verify result is a col Expression
    assert isinstance(result, bigframes.core.col.Expression)

    # Verify the internal expression structure
    expr = result._value
    assert isinstance(expr, ex.OpExpression)
    assert expr.op == array_op._ARRAY_IS_DISTINCT_OP

    # Verify arguments
    assert len(expr.inputs) == 1
    assert isinstance(expr.inputs[0], ex.ScalarConstantExpression)

    snippet = bigframes.core.compile.sqlglot.testing.to_sql_snippet(result)
    snapshot.assert_match(snippet + "\n", "out.sql")


def test_array_last_expression(snapshot):
    # Call the function with literals or col() expressions
    result = bbq.array_last(
        [1, 2],
    )

    # Verify result is a col Expression
    assert isinstance(result, bigframes.core.col.Expression)

    # Verify the internal expression structure
    expr = result._value
    assert isinstance(expr, ex.OpExpression)
    assert expr.op == array_op._ARRAY_LAST_OP

    # Verify arguments
    assert len(expr.inputs) == 1
    assert isinstance(expr.inputs[0], ex.ScalarConstantExpression)

    snippet = bigframes.core.compile.sqlglot.testing.to_sql_snippet(result)
    snapshot.assert_match(snippet + "\n", "out.sql")


def test_array_length_expression(snapshot):
    # Call the function with literals or col() expressions
    result = bbq.array_length(
        [1, 2],
    )

    # Verify result is a col Expression
    assert isinstance(result, bigframes.core.col.Expression)

    # Verify the internal expression structure
    expr = result._value
    assert isinstance(expr, ex.OpExpression)
    assert expr.op == array_op._ARRAY_LENGTH_OP

    # Verify arguments
    assert len(expr.inputs) == 1
    assert isinstance(expr.inputs[0], ex.ScalarConstantExpression)

    snippet = bigframes.core.compile.sqlglot.testing.to_sql_snippet(result)
    snapshot.assert_match(snippet + "\n", "out.sql")


def test_array_reverse_expression(snapshot):
    # Call the function with literals or col() expressions
    result = bbq.array_reverse(
        [1, 2],
    )

    # Verify result is a col Expression
    assert isinstance(result, bigframes.core.col.Expression)

    # Verify the internal expression structure
    expr = result._value
    assert isinstance(expr, ex.OpExpression)
    assert expr.op == array_op._ARRAY_REVERSE_OP

    # Verify arguments
    assert len(expr.inputs) == 1
    assert isinstance(expr.inputs[0], ex.ScalarConstantExpression)

    snippet = bigframes.core.compile.sqlglot.testing.to_sql_snippet(result)
    snapshot.assert_match(snippet + "\n", "out.sql")


def test_array_slice_expression(snapshot):
    # Call the function with literals or col() expressions
    result = bbq.array_slice(
        [1, 2],
        1,
        1,
    )

    # Verify result is a col Expression
    assert isinstance(result, bigframes.core.col.Expression)

    # Verify the internal expression structure
    expr = result._value
    assert isinstance(expr, ex.OpExpression)
    assert expr.op == array_op._ARRAY_SLICE_OP

    # Verify arguments
    assert len(expr.inputs) == 3
    assert isinstance(expr.inputs[0], ex.ScalarConstantExpression)
    assert isinstance(expr.inputs[1], ex.ScalarConstantExpression)
    assert isinstance(expr.inputs[2], ex.ScalarConstantExpression)

    snippet = bigframes.core.compile.sqlglot.testing.to_sql_snippet(result)
    snapshot.assert_match(snippet + "\n", "out.sql")


def test_array_to_string_expression(snapshot):
    # Call the function with literals or col() expressions
    result = bbq.array_to_string(
        ["a", "b"],
        ", ",
        "test",
    )

    # Verify result is a col Expression
    assert isinstance(result, bigframes.core.col.Expression)

    # Verify the internal expression structure
    expr = result._value
    assert isinstance(expr, ex.OpExpression)
    assert expr.op == array_op._ARRAY_TO_STRING_OP

    # Verify arguments
    assert len(expr.inputs) == 3
    assert isinstance(expr.inputs[0], ex.ScalarConstantExpression)
    assert isinstance(expr.inputs[1], ex.ScalarConstantExpression)
    assert isinstance(expr.inputs[2], ex.ScalarConstantExpression)

    snippet = bigframes.core.compile.sqlglot.testing.to_sql_snippet(result)
    snapshot.assert_match(snippet + "\n", "out.sql")


def test_flatten_expression(snapshot):
    # Call the function with literals or col() expressions
    result = bbq.flatten(
        [[1, 2], [3, 4]],
        1,
    )

    # Verify result is a col Expression
    assert isinstance(result, bigframes.core.col.Expression)

    # Verify the internal expression structure
    expr = result._value
    assert isinstance(expr, ex.OpExpression)
    assert expr.op == array_op._FLATTEN_OP

    # Verify arguments
    assert len(expr.inputs) == 2
    assert isinstance(expr.inputs[0], ex.ScalarConstantExpression)
    assert isinstance(expr.inputs[1], ex.ScalarConstantExpression)

    snippet = bigframes.core.compile.sqlglot.testing.to_sql_snippet(result)
    snapshot.assert_match(snippet + "\n", "out.sql")


def test_generate_array_expression(snapshot):
    # Call the function with literals or col() expressions
    result = bbq.generate_array(
        1,
        1,
        1,
    )

    # Verify result is a col Expression
    assert isinstance(result, bigframes.core.col.Expression)

    # Verify the internal expression structure
    expr = result._value
    assert isinstance(expr, ex.OpExpression)
    assert expr.op == array_op._GENERATE_ARRAY_OP

    # Verify arguments
    assert len(expr.inputs) == 3
    assert isinstance(expr.inputs[0], ex.ScalarConstantExpression)
    assert isinstance(expr.inputs[1], ex.ScalarConstantExpression)
    assert isinstance(expr.inputs[2], ex.ScalarConstantExpression)

    snippet = bigframes.core.compile.sqlglot.testing.to_sql_snippet(result)
    snapshot.assert_match(snippet + "\n", "out.sql")
