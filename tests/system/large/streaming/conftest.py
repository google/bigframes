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

import pytest

import bigframes


@pytest.fixture(scope="session", autouse=True)
def refresh_streaming_table(session_load: bigframes.Session):
    """Refreshes the streaming source table once per test session.

    BigQuery continuous queries use the `APPENDS` table-valued function (TVF)
    to perform Change Data Capture (CDC) streaming. `APPENDS` only streams rows
    that have been appended to the table within the change history window (up to
    7 days) on or after `start_timestamp`.

    If the table is not replaced, the historical change events expire and no
    recent append events exist in the table's change stream. As a result, the
    continuous query will find 0 appended rows to process, causing downstream
    destinations (Bigtable / Pub/Sub) to remain empty and tests asserting on
    received rows to fail. Replacing the table also prevents unbounded row
    growth across test runs.
    """
    df = session_load.read_gbq("bigquery-public-data.ml_datasets.penguins")
    df.to_gbq("birds.penguins_bigtable_streaming", if_exists="replace")
