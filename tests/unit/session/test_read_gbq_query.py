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

"""Unit tests for read_gbq_query functions."""

import google.cloud.bigquery as bigquery

from bigframes.testing import mocks


def test_read_gbq_query_sets_destination_table():
    """Workaround the 10 GB query results limitation by setting a destination table.

    See internal issue b/303057336.
    """
    # Use partial ordering mode to skip column uniqueness checks.
    session = mocks.create_bigquery_session(ordering_mode="partial")

    _ = session.read_gbq_query("SELECT 'my-test-query';", allow_large_results=True)
    queries = session._queries  # type: ignore
    configs = session._job_configs  # type: ignore

    for query, config in zip(queries, configs):
        if query == "SELECT 'my-test-query';" and not config.dry_run:
            break

    assert query == "SELECT 'my-test-query';"
    assert config.destination is not None
    session.close()


def test_read_gbq_query_escapes_columns_and_index_col():
    schema = (
        bigquery.SchemaField("idx`col", "INT64"),
        bigquery.SchemaField("val`\\col", "STRING"),
    )
    session = mocks.create_bigquery_session(
        ordering_mode="partial", table_schema=schema
    )

    _ = session.read_gbq_query(
        "SELECT 1 AS a",
        index_col="idx`col",
        columns=["val`\\col"],
        max_results=10,
        dry_run=True,
    )
    queries = session._queries  # type: ignore
    assert (
        r"SELECT `_bf_source`.`idx\`col`, `_bf_source`.`val\`\\col` FROM (SELECT 1 AS a) AS _bf_source LIMIT 10"
        in queries
    )
    session.close()
