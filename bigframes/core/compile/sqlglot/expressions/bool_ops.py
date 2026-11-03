# Copyright 2025 Google LLC
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

import bigframes_vendored.sqlglot.expressions as sge

import bigframes.core.compile.sqlglot.expression_compiler as expression_compiler
from bigframes import dtypes
from bigframes import operations as ops
from bigframes.core.compile.sqlglot import sql
from bigframes.core.compile.sqlglot.expressions.typed_expr import TypedExpr

register_binary_op = expression_compiler.expression_compiler.register_binary_op


@register_binary_op(ops.and_op)
def _(left: TypedExpr, right: TypedExpr) -> sge.Expression:
    # For AND, when we encounter a NULL value, we only know when the result is FALSE,
    # otherwise the result is unknown (NULL). See: truth table at
    # https://en.wikibooks.org/wiki/Structured_Query_Language/NULLs_and_the_Three_Valued_Logic#AND,_OR
    if sql.is_null_literal(left.expr) and sql.is_null_literal(right.expr):
        return sge.null()
    if sql.is_null_literal(left.expr):
        condition = sge.EQ(this=right.expr, expression=sge.convert(False))
        return sge.If(this=condition, true=right.expr, false=sge.null())
    if sql.is_null_literal(right.expr):
        condition = sge.EQ(this=left.expr, expression=sge.convert(False))
        return sge.If(this=condition, true=left.expr, false=sge.null())

    if left.dtype == dtypes.BOOL_DTYPE and right.dtype == dtypes.BOOL_DTYPE:
        return sge.And(this=left.expr, expression=right.expr)
    return sge.BitwiseAnd(this=left.expr, expression=right.expr)


@register_binary_op(ops.or_op)
def _(left: TypedExpr, right: TypedExpr) -> sge.Expression:
    # For OR, when we encounter a NULL value, we only know when the result is TRUE,
    # otherwise the result is unknown (NULL). See: truth table at
    # https://en.wikibooks.org/wiki/Structured_Query_Language/NULLs_and_the_Three_Valued_Logic#AND,_OR
    if sql.is_null_literal(left.expr) and sql.is_null_literal(right.expr):
        return sge.null()
    if sql.is_null_literal(left.expr):
        condition = sge.EQ(this=right.expr, expression=sge.convert(True))
        return sge.If(this=condition, true=right.expr, false=sge.null())
    if sql.is_null_literal(right.expr):
        condition = sge.EQ(this=left.expr, expression=sge.convert(True))
        return sge.If(this=condition, true=left.expr, false=sge.null())

    if left.dtype == dtypes.BOOL_DTYPE and right.dtype == dtypes.BOOL_DTYPE:
        return sge.Or(this=left.expr, expression=right.expr)
    return sge.BitwiseOr(this=left.expr, expression=right.expr)


@register_binary_op(ops.xor_op)
def _(left: TypedExpr, right: TypedExpr) -> sge.Expression:
    if sql.is_null_literal(left.expr) or sql.is_null_literal(right.expr):
        return sge.null()

    if left.dtype == dtypes.BOOL_DTYPE and right.dtype == dtypes.BOOL_DTYPE:
        return sge.Or(
            this=sge.paren(
                sge.And(this=left.expr, expression=sge.Not(this=right.expr))
            ),
            expression=sge.paren(
                sge.And(this=sge.Not(this=left.expr), expression=right.expr)
            ),
        )
    return sge.BitwiseXor(this=left.expr, expression=right.expr)
