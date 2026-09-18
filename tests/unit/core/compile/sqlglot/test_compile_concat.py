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

import google.cloud.bigquery as bigquery
import pytest

import bigframes.pandas as bpd
from bigframes import dtypes
from bigframes.core import (
    array_value,
    blocks,
    bq_data,
    identifiers,
    nodes,
    ordering,
    schema,
)

pytest.importorskip("pytest_snapshot")


def test_compile_concat(scalar_types_df: bpd.DataFrame, snapshot):
    # TODO: concat two same dataframes, which SQL does not get reused.
    df1 = scalar_types_df[["rowindex", "int64_col", "string_col"]]
    concat_df = bpd.concat([df1, df1])
    snapshot.assert_match(concat_df.sql, "out.sql")


def test_compile_concat_filter_sorted(scalar_types_df: bpd.DataFrame, snapshot):
    scalars_array_value = scalar_types_df._block.expr
    input_1 = scalars_array_value.select_columns(["float64_col", "int64_col"]).order_by(
        [ordering.ascending_over("int64_col")]
    )
    input_2 = scalars_array_value.filter_by_id("bool_col").select_columns(
        ["float64_col", "int64_too"]
    )

    result = input_1.concat([input_2, input_1, input_2])

    new_names = ["float64_col", "int64_col"]
    col_ids = {
        old_name: new_name for old_name, new_name in zip(result.column_ids, new_names)
    }
    result = result.rename_columns(col_ids).select_columns(new_names)

    sql = result.session._executor.to_sql(result, enable_cache=False)
    snapshot.assert_match(sql, "out.sql")


def test_compile_concat_w_json_string_workaround(
    compiler_session_w_json_types, json_types_table_schema, snapshot
):
    table_ref = bigquery.TableReference(
        bigquery.DatasetReference("bigframes-dev", "sqlglot_test"),
        "json_types",
    )
    table = bigquery.Table(table_ref, tuple(json_types_table_schema))
    table._properties["location"] = compiler_session_w_json_types._location
    native_table = bq_data.GbqNativeTable.from_table(table)

    logical_schema = schema.ArraySchema(
        (
            schema.SchemaItem("rowindex", dtypes.INT_DTYPE),
            schema.SchemaItem("json_col", dtypes.JSON_DTYPE),
            schema.SchemaItem("json_string_col", dtypes.JSON_DTYPE),
        )
    )
    source = bq_data.BigqueryDataSource(
        table=native_table,
        schema=logical_schema,
    )
    scan_list = nodes.ScanList(
        (
            nodes.ScanItem(identifiers.ColumnId("rowindex"), "rowindex"),
            nodes.ScanItem(identifiers.ColumnId("json_col"), "json_col"),
            nodes.ScanItem(identifiers.ColumnId("json_string_col"), "json_string_col"),
        )
    )
    read_node = nodes.ReadTableNode(
        source=source,
        scan_list=scan_list,
        table_session=compiler_session_w_json_types,
    )
    block = blocks.Block(
        array_value.ArrayValue(read_node),
        index_columns=(),
        column_labels=("rowindex", "json_col", "json_string_col"),
    )
    df = bpd.DataFrame(block)
    s1 = df["json_col"]
    s2 = df["json_string_col"]

    concat_series = bpd.concat([s1, s2])

    snapshot.assert_match(concat_series.sql, "out.sql")
