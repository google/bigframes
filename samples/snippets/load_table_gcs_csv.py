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

# [START bigquery_load_table_gcs_csv]
import bigframes.pandas as bpd
from google.cloud import bigquery

# Set partial ordering mode for BigQuery DataFrames.
bpd.options.bigquery.ordering_mode = "partial"


def load_table_gcs_csv_bigframes(
    table_id: str = "your-project.your_dataset.your_table_name",
) -> None:
    """Loads a CSV file from Cloud Storage into a BigQuery table using BigQuery DataFrames."""
    uri = "gs://cloud-samples-data/bigquery/us-states/us-states.csv"

    df = bpd.read_csv(uri)
    df.to_gbq(table_id, if_exists="replace")
    print(f"Loaded {len(df)} rows using BigQuery DataFrames.")


def load_table_gcs_csv_client(
    table_id: str = "your-project.your_dataset.your_table_name",
) -> None:
    """Loads a CSV file from Cloud Storage into a BigQuery table using google-cloud-bigquery."""
    uri = "gs://cloud-samples-data/bigquery/us-states/us-states.csv"

    client = bigquery.Client()
    job_config = bigquery.LoadJobConfig(
        schema=[
            bigquery.SchemaField("name", "STRING"),
            bigquery.SchemaField("post_abbr", "STRING"),
        ],
        skip_leading_rows=1,
        source_format=bigquery.SourceFormat.CSV,
    )

    load_job = client.load_table_from_uri(uri, table_id, job_config=job_config)
    load_job.result()  # Waits for the job to complete.

    destination_table = client.get_table(table_id)
    print(f"Loaded {destination_table.num_rows} rows using client library.")


# [Preferred] Run using BigQuery DataFrames:
# load_table_gcs_csv_bigframes("your-project.your_dataset.your_table_name")

# Alternatively, run using google-cloud-bigquery client library:
# load_table_gcs_csv_client("your-project.your_dataset.your_table_name")
# [END bigquery_load_table_gcs_csv]
