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

# [START bigquery_bigframes_delete_table]
import bigframes.pandas as bpd

# Set partial ordering mode for BigQuery DataFrames.
bpd.options.bigquery.ordering_mode = "partial"


def delete_table(
    table_id: str = "your-project.your_dataset.your_table_name",
) -> None:
    session = bpd.get_global_session()

    # Delete the table if it exists.
    session.bqclient.delete_table(table_id, not_found_ok=True)
    print(f"Deleted table '{table_id}'.")


# [END bigquery_bigframes_delete_table]


if __name__ == "__main__":
    import os

    table_id = os.environ.get("TABLE_ID", "your-project.your_dataset.your_table_name")
    delete_table(table_id=table_id)
