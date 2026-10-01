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
from __future__ import annotations

import asyncio
import datetime
from typing import Optional
from unittest import mock

import pytest
from google.cloud import bigquery, bigquery_storage_v1

from bigframes import dtypes
from bigframes.core import bq_data, identifiers, nodes, schema
from bigframes.session import execution_spec, executor, read_api_execution
from bigframes.testing import mocks

SPEC = execution_spec.ExecutionSpec(ordered=False)
AT_TIME = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)


@pytest.fixture
def object_under_test():
    return read_api_execution.ReadApiSemiExecutor(
        bqstoragereadclient=mock.create_autospec(
            bigquery_storage_v1.BigQueryReadClient, instance=True
        ),
        project="my-project",
    )


def create_read_table_node(
    dataset_id: str, at_time: Optional[datetime.datetime]
) -> nodes.ReadTableNode:
    table = bq_data.GbqNativeTable(
        project_id="my-project",
        dataset_id=dataset_id,
        table_id="my_table",
        physical_schema=(bigquery.SchemaField("col", "INTEGER"),),
        metadata=bq_data.TableMetadata(
            location=bq_data.BigQueryRegion("us-central1"),
            type="TABLE",
        ),
    )
    source = bq_data.BigqueryDataSource(
        table=table,
        schema=schema.ArraySchema((schema.SchemaItem("col", dtypes.INT_DTYPE),)),
        at_time=at_time,
    )
    return nodes.ReadTableNode(
        source=source,
        scan_list=nodes.ScanList((nodes.ScanItem(identifiers.ColumnId("col"), "col"),)),
        table_session=mocks.create_bigquery_session(),
    )


def test_read_api_semi_executor_lakehouse_table_with_time_travel_uses_sql(
    object_under_test,
):
    # The Read API returns current data for Lakehouse tables even when
    # snapshot_time is set (b/568786565), so these reads must be left to the SQL path.
    plan = create_read_table_node("my_catalog.my_namespace", at_time=AT_TIME)

    assert asyncio.run(object_under_test.execute(plan, SPEC)) is None


@pytest.mark.parametrize(
    ("dataset_id", "at_time"),
    (
        # Public read APIs always pin Lakehouse tables; at_time is None only
        # when snapshots are disabled, e.g. read_gbq_table_streaming.
        pytest.param("my_catalog.my_namespace", None, id="lakehouse-no-time-travel"),
        pytest.param("my_dataset", AT_TIME, id="standard-time-travel"),
        pytest.param("my_dataset", None, id="standard-no-time-travel"),
    ),
)
def test_read_api_semi_executor_uses_read_api(dataset_id, at_time, object_under_test):
    plan = create_read_table_node(dataset_id, at_time=at_time)

    result = asyncio.run(object_under_test.execute(plan, SPEC))

    assert isinstance(result, executor.BQTableExecuteResult)
