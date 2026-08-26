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

# [START bigquery_create_table]
import bigframes.pandas as bpd
from google.cloud import bigquery
import pandas as pd
import pandas_gbq

# Set partial ordering mode for BigQuery DataFrames.
bpd.options.bigquery.ordering_mode = "partial"


def create_table_bigframes(
    table_id: str = "your-project.your_dataset.your_table_name",
) -> None:
    """Creates and populates a BigQuery table using BigQuery DataFrames."""
    df_bigframes = bpd.DataFrame(
        {
            "full_name": ["Barney Rubble", "Betty Rubble"],
            "age": [30, 28],
        }
    )
    df_bigframes.to_gbq(table_id, if_exists="replace")
    print(f"Created/populated table {table_id} using BigQuery DataFrames.")


def create_table_client(
    table_id: str = "your-project.your_dataset.your_table_name",
) -> None:
    """Creates an empty table with an explicit schema using google-cloud-bigquery."""
    client = bigquery.Client()
    schema = [
        bigquery.SchemaField("full_name", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("age", "INTEGER", mode="REQUIRED"),
    ]
    table = bigquery.Table(table_id, schema=schema)
    table = client.create_table(table)
    print(
        f"Created table {table.project}.{table.dataset_id}.{table.table_id} with schema."
    )


def create_table_pandas_gbq(
    table_id: str = "your-project.your_dataset.your_table_name",
) -> None:
    """Ingests an in-memory pandas DataFrame into a BigQuery table using pandas-gbq."""
    df_pandas = pd.DataFrame(
        {
            "full_name": ["Phred Phlyntstone", "Wylma Phlyntstone"],
            "age": [32, 29],
        }
    )
    pandas_gbq.to_gbq(df_pandas, table_id, if_exists="replace")
    print(f"Created/populated table {table_id} using pandas-gbq.")


# [Preferred] Run using BigQuery DataFrames:
# create_table_bigframes("your-project.your_dataset.your_table_name")

# Alternatively, run using google-cloud-bigquery client library:
# create_table_client("your-project.your_dataset.your_table_name")

# Alternatively, run using pandas-gbq:
# create_table_pandas_gbq("your-project.your_dataset.your_table_name")
# [END bigquery_create_table]
