# Copyright 2023 Google LLC
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

"""
Utility functions for SQL construction.
"""

from __future__ import annotations

import json
import re
from typing import (
    TYPE_CHECKING,
    Any,
    Collection,
    Iterable,
    Mapping,
    Optional,
    Union,
    cast,
)

import bigframes_vendored.sqlglot as sg
import bigframes_vendored.sqlglot.expressions as sge

from bigframes.core.compile.sqlglot import sql

if TYPE_CHECKING:
    import google.cloud.bigquery as bigquery

    import bigframes.core.ordering


# shapely.wkt.dumps was moved to shapely.io.to_wkt in 2.0.
try:
    from shapely.io import to_wkt  # type: ignore
except ImportError:
    from shapely.wkt import dumps  # type: ignore

    to_wkt = dumps


def identifier(name: str) -> str:
    return f"`{escape_chars(name)}`"


def escape_chars(value: str):
    """Escapes all special characters"""
    # TODO: Reuse literal's escaping logic instead of re-implementing it here.
    # https://cloud.google.com/bigquery/docs/reference/standard-sql/lexical#string_and_bytes_literals
    trans_table = str.maketrans(
        {
            "\a": r"\a",
            "\b": r"\b",
            "\f": r"\f",
            "\n": r"\n",
            "\r": r"\r",
            "\t": r"\t",
            "\v": r"\v",
            "\\": r"\\",
            "?": r"\?",
            '"': r"\"",
            "'": r"\'",
            "`": r"\`",
        }
    )
    return value.translate(trans_table)


# GoogleSQL's rules for unquoted and quoted identifiers:
# https://cloud.google.com/bigquery/docs/reference/standard-sql/lexical#identifiers
_UNQUOTED_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

_QUOTED_IDENTIFIER_RE = re.compile(r"`(?:[^`\\\r\n]|\\[^\r\n])+`")

_ALLOWED_DATA_TYPE_NODES = (
    sge.DataType,
    sge.ColumnDef,
    sge.Identifier,
    sge.DataTypeParam,
    sge.Literal,
    sge.Var,
)


def is_quoted_identifier(text: str) -> bool:
    """Returns whether text is one backtick-quoted identifier."""
    return _QUOTED_IDENTIFIER_RE.fullmatch(text) is not None


def normalize_option_name(name: object) -> str:
    """Returns name without surrounding whitespace if it is one identifier.

    Raises:
        ValueError: If name isn't a plain or backtick-quoted identifier.
    """
    name_sql = str(name).strip()
    if name is None or not (
        _UNQUOTED_IDENTIFIER_RE.fullmatch(name_sql) or is_quoted_identifier(name_sql)
    ):
        raise ValueError(
            f"Invalid option name {name!r}. Expected an identifier, such as "
            "'model_type'."
        )
    return name_sql


def normalize_data_type(data_type: object) -> str:
    """Returns data_type normalized by sqlglot if it is one BigQuery data type.

    Raises:
        ValueError: If data_type isn't a single BigQuery data type.
    """
    type_sql = str(data_type)
    # sqlglot's parse_one splits on ';' and returns only the first chunk.
    if data_type is None or ";" in type_sql:
        raise ValueError(
            f"Invalid data type {data_type!r}. Expected a BigQuery data type, "
            "such as 'STRING' or 'ARRAY<FLOAT64>'."
        )
    try:
        parsed = sg.parse_one(
            type_sql,
            read=sql.base.DIALECT,
            into=sge.DataType,
            error_level=sg.ErrorLevel.RAISE,
        )
    except (sg.ParseError, sg.TokenError) as exc:
        raise ValueError(
            f"Invalid data type {data_type!r}. Expected a BigQuery data type, "
            "such as 'STRING' or 'ARRAY<FLOAT64>'."
        ) from exc
    if not isinstance(parsed, sge.DataType) or not _is_allowed_data_type(parsed):
        raise ValueError(
            f"Invalid data type {data_type!r}. Expected a BigQuery data type, "
            "such as 'STRING' or 'ARRAY<FLOAT64>'."
        )
    return parsed.sql(dialect=sql.base.DIALECT, comments=False)


def _is_allowed_data_type(parsed: sge.DataType) -> bool:
    for node in parsed.walk():
        if not isinstance(node, _ALLOWED_DATA_TYPE_NODES):
            return False
        if isinstance(node, sge.Identifier) and not node.name:
            return False
        if isinstance(node, sge.Literal) and not node.is_int:
            return False
        if isinstance(node, sge.Var) and node.name.upper() != "MAX":
            return False
    return True


def multi_literal(*values: Any):
    literal_strings = [sql.to_sql(sql.literal(i)) for i in values]
    return "(" + ", ".join(literal_strings) + ")"


def cast_as_string(column_name: str) -> str:
    """Return a string representing string casting of a column."""

    return sge.Cast(this=sge.to_identifier(column_name, quoted=True), to="STRING").sql(
        dialect="bigquery"
    )


def to_json_string(column_name: str) -> str:
    """Return a string representing JSON version of a column."""

    return f"TO_JSON_STRING({sql.to_sql(sql.identifier(column_name))})"


def csv(values: Iterable[str]) -> str:
    """Return a string of comma separated values."""
    return ", ".join(values)


def infix_op(opname: str, left_arg: str, right_arg: str):
    # Maybe should add parentheses??
    return f"{left_arg} {opname} {right_arg}"


def is_distinct_sql(columns: Iterable[str], table_ref: bigquery.TableReference) -> str:
    table_expr = sge.Table(
        this=sge.Identifier(this=table_ref.table_id, quoted=True),
        db=sge.Identifier(this=table_ref.dataset_id, quoted=True),
        catalog=sge.Identifier(this=table_ref.project, quoted=True),
    )
    to_select = [sge.to_identifier(col, quoted=True) for col in columns]

    full_table_sql = (
        sge.Select().select(*to_select).from_(table_expr).sql(dialect="bigquery")
    )
    distinct_table_sql = (
        sge.Select()
        .select(*to_select)
        .distinct()
        .from_(table_expr)
        .sql(dialect="bigquery")
    )

    is_unique_sql = f"""WITH full_table AS (
        {full_table_sql}
    ),
    distinct_table AS (
        {distinct_table_sql}
    )

    SELECT (SELECT COUNT(*) FROM full_table) AS `total_count`,
    (SELECT COUNT(*) FROM distinct_table) AS `distinct_count`
    """
    return is_unique_sql


def ordering_clause(
    ordering: Iterable[bigframes.core.ordering.OrderingExpression],
) -> str:
    import bigframes.core.expression

    parts = []
    for col_ref in ordering:
        asc_desc = "ASC" if col_ref.direction.is_ascending else "DESC"
        null_clause = "NULLS LAST" if col_ref.na_last else "NULLS FIRST"
        ordering_expr = col_ref.scalar_expression
        # We don't know how to compile scalar expressions in isolation
        if ordering_expr.is_const:
            # Probably shouldn't have constants in ordering definition, but best to ignore if somehow they end up here.
            continue
        assert isinstance(ordering_expr, bigframes.core.expression.DerefOp)
        part = f"{sql.to_sql(sql.identifier(ordering_expr.id.sql))} {asc_desc} {null_clause}"
        parts.append(part)
    return f"ORDER BY {' ,'.join(parts)}"


def create_vector_index_ddl(
    *,
    replace: bool,
    index_name: str,
    table_name: str,
    column_name: str,
    stored_column_names: Collection[str],
    options: Mapping[str, Union[str | int | bool | float]] = {},
) -> str:
    """Encode the VECTOR INDEX statement for BigQuery Vector Search."""

    if replace:
        create = "CREATE OR REPLACE VECTOR INDEX "
    else:
        create = "CREATE VECTOR INDEX IF NOT EXISTS "

    if len(stored_column_names) > 0:
        escaped_stored = [
            f"{sql.to_sql(sql.identifier(name))}" for name in stored_column_names
        ]
        storing = f"STORING({', '.join(escaped_stored)}) "
    else:
        storing = ""

    rendered_options = ", ".join(
        [
            f"{normalize_option_name(option_name)} = {sql.to_sql(sql.literal(option_value))}"
            for option_name, option_value in options.items()
        ]
    )

    return f"""
    {create} {sql.to_sql(sql.identifier(index_name))}
    ON {sql.to_sql(sql.identifier(table_name))}({sql.to_sql(sql.identifier(column_name))})
    {storing}
    OPTIONS({rendered_options});
    """


def create_vector_search_sql(
    sql_string: str,
    *,
    base_table: str,
    column_to_search: str,
    query_column_to_search: Optional[str] = None,
    top_k: Optional[int] = None,
    distance_type: Optional[str] = None,
    options: Optional[Mapping[str, Union[str | int | bool | float]]] = None,
) -> str:
    """Encode the VECTOR SEARCH statement for BigQuery Vector Search."""

    vector_search_args = [
        f"TABLE {sql.to_sql(sql.identifier(cast(str, base_table)))}",
        f"{sql.to_sql(sql.literal(column_to_search))}",
        f"({sql_string})",
    ]

    if query_column_to_search is not None:
        vector_search_args.append(
            f"query_column_to_search => {sql.to_sql(sql.literal(query_column_to_search))}"
        )

    if top_k is not None:
        vector_search_args.append(f"top_k=> {sql.to_sql(sql.literal(top_k))}")

    if distance_type is not None:
        vector_search_args.append(
            f"distance_type => {sql.to_sql(sql.literal(distance_type))}"
        )

    if options is not None:
        vector_search_args.append(
            f"options => {sql.to_sql(sql.literal(json.dumps(options, indent=None)))}"
        )

    args_str = ",\n".join(vector_search_args)
    return f"""
    SELECT
        query.*,
        base.*,
        distance,
    FROM VECTOR_SEARCH({args_str})
    """
