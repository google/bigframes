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
# This file was generated from: scripts/data/sql-functions/aead.yaml
# by the script: scripts/generate_bigframes_bigquery.py

import pytest

import bigframes.bigquery as bbq
import bigframes.core.col
import bigframes.core.compile.sqlglot.testing
import bigframes.core.expression as ex
import bigframes.operations.googlesql.aead as aead_op

pytest.importorskip("pytest_snapshot")


def test_decrypt_bytes_expression(snapshot):
    # Call the function with literals or col() expressions
    result = bbq.aead.decrypt_bytes(
        b"keyset",
        b"ciphertext",
        b"additional_data",
    )

    # Verify result is a col Expression
    assert isinstance(result, bigframes.core.col.Expression)

    # Verify the internal expression structure
    expr = result._value
    assert isinstance(expr, ex.OpExpression)
    assert expr.op == aead_op._DECRYPT_BYTES_OP

    # Verify arguments
    assert len(expr.inputs) == 3
    assert isinstance(expr.inputs[0], ex.ScalarConstantExpression)
    assert isinstance(expr.inputs[1], ex.ScalarConstantExpression)
    assert isinstance(expr.inputs[2], ex.ScalarConstantExpression)

    snippet = bigframes.core.compile.sqlglot.testing.to_sql_snippet(result)
    snapshot.assert_match(snippet + "\n", "out.sql")


def test_decrypt_string_expression(snapshot):
    # Call the function with literals or col() expressions
    result = bbq.aead.decrypt_string(
        b"keyset",
        b"ciphertext",
        "additional_data",
    )

    # Verify result is a col Expression
    assert isinstance(result, bigframes.core.col.Expression)

    # Verify the internal expression structure
    expr = result._value
    assert isinstance(expr, ex.OpExpression)
    assert expr.op == aead_op._DECRYPT_STRING_OP

    # Verify arguments
    assert len(expr.inputs) == 3
    assert isinstance(expr.inputs[0], ex.ScalarConstantExpression)
    assert isinstance(expr.inputs[1], ex.ScalarConstantExpression)
    assert isinstance(expr.inputs[2], ex.ScalarConstantExpression)

    snippet = bigframes.core.compile.sqlglot.testing.to_sql_snippet(result)
    snapshot.assert_match(snippet + "\n", "out.sql")


def test_encrypt_expression(snapshot):
    # Call the function with literals or col() expressions
    result = bbq.aead.encrypt(
        b"keyset",
        "plaintext",
        "additional_data",
    )

    # Verify result is a col Expression
    assert isinstance(result, bigframes.core.col.Expression)

    # Verify the internal expression structure
    expr = result._value
    assert isinstance(expr, ex.OpExpression)
    assert expr.op == aead_op._ENCRYPT_OP

    # Verify arguments
    assert len(expr.inputs) == 3
    assert isinstance(expr.inputs[0], ex.ScalarConstantExpression)
    assert isinstance(expr.inputs[1], ex.ScalarConstantExpression)
    assert isinstance(expr.inputs[2], ex.ScalarConstantExpression)

    snippet = bigframes.core.compile.sqlglot.testing.to_sql_snippet(result)
    snapshot.assert_match(snippet + "\n", "out.sql")
