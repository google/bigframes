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

from google.cloud import bigquery

from bigframes import dtypes
from bigframes.core import bq_data, schema, sql_nodes


def test_is_star_selection_matching_schema():
    table_ref = bigquery.TableReference(
        bigquery.DatasetReference("project", "dataset"), "table"
    )
    physical_schema = (
        bigquery.SchemaField("col1", "INTEGER"),
        bigquery.SchemaField("col2", "STRING"),
    )
    table = bigquery.Table(table_ref, physical_schema)
    native_table = bq_data.GbqNativeTable.from_table(table)
    logical_schema = schema.ArraySchema(
        (
            schema.SchemaItem("col1", dtypes.INT_DTYPE),
            schema.SchemaItem("col2", dtypes.STRING_DTYPE),
        )
    )
    source = bq_data.BigqueryDataSource(
        table=native_table,
        schema=logical_schema,
    )

    node = sql_nodes.SqlDataSource(source=source)

    assert node.is_star_selection is True


def test_is_star_selection_mismatched_types():
    table_ref = bigquery.TableReference(
        bigquery.DatasetReference("project", "dataset"), "table"
    )
    physical_schema = (
        bigquery.SchemaField("col1", "INTEGER"),
        bigquery.SchemaField("col2", "STRING"),
    )
    table = bigquery.Table(table_ref, physical_schema)
    native_table = bq_data.GbqNativeTable.from_table(table)
    # col2 logical type is JSON_DTYPE while physical is STRING
    logical_schema = schema.ArraySchema(
        (
            schema.SchemaItem("col1", dtypes.INT_DTYPE),
            schema.SchemaItem("col2", dtypes.JSON_DTYPE),
        )
    )
    source = bq_data.BigqueryDataSource(
        table=native_table,
        schema=logical_schema,
    )

    node = sql_nodes.SqlDataSource(source=source)

    assert node.is_star_selection is False


def test_is_star_selection_mismatched_names():
    table_ref = bigquery.TableReference(
        bigquery.DatasetReference("project", "dataset"), "table"
    )
    physical_schema = (
        bigquery.SchemaField("col1", "INTEGER"),
        bigquery.SchemaField("col2", "STRING"),
    )
    table = bigquery.Table(table_ref, physical_schema)
    native_table = bq_data.GbqNativeTable.from_table(table)
    logical_schema = schema.ArraySchema((schema.SchemaItem("col1", dtypes.INT_DTYPE),))
    source = bq_data.BigqueryDataSource(
        table=native_table,
        schema=logical_schema,
    )

    node = sql_nodes.SqlDataSource(source=source)

    assert node.is_star_selection is False
