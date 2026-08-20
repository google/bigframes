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

import bigframes.pandas as bpd

from . import delete_table


def test_delete_table(project_id: str, dataset_id: str) -> None:
    table_id = f"{project_id}.{dataset_id}.test_delete_table"

    # Create a table to delete
    df = bpd.DataFrame({"col": [1, 2, 3]})
    df.to_gbq(table_id, if_exists="replace")

    # Delete the table
    delete_table.delete_table(table_id=table_id)

    # Verify table is deleted
    session = bpd.get_global_session()
    tables = [
        t.table_id for t in session.bqclient.list_tables(f"{project_id}.{dataset_id}")
    ]
    assert "test_delete_table" not in tables
