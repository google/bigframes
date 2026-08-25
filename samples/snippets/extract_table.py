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

# [START bigquery_extract_table]
import bigframes.pandas as bpd
from google.cloud import bigquery

# Set partial ordering mode for BigQuery DataFrames.
bpd.options.bigquery.ordering_mode = "partial"


def extract_table_bigframes(bucket_name: str = "my-bucket") -> None:
    """Exports a BigQuery table to Cloud Storage in CSV format using BigQuery DataFrames."""
    project = "bigquery-public-data"
    dataset_id = "samples"
    table_id = "shakespeare"
    destination_uri = f"gs://{bucket_name}/shakespeare-*.csv"

    full_table_id = f"{project}.{dataset_id}.{table_id}"
    df = bpd.read_gbq(full_table_id)
    df.to_csv(destination_uri)
    print(f"Exported {full_table_id} to {destination_uri} using BigQuery DataFrames.")


def extract_table_client(bucket_name: str = "my-bucket") -> None:
    """Exports a BigQuery table to Cloud Storage in CSV format using google-cloud-bigquery."""
    project = "bigquery-public-data"
    dataset_id = "samples"
    table_id = "shakespeare"
    destination_uri = f"gs://{bucket_name}/shakespeare-*.csv"

    client = bigquery.Client()
    dataset_ref = bigquery.DatasetReference(project, dataset_id)
    table_ref = dataset_ref.table(table_id)

    extract_job = client.extract_table(
        table_ref,
        destination_uri,
        # Location must match that of the source table.
        location="US",
    )  # API request
    extract_job.result()  # Waits for job to complete.

    print(
        f"Exported {project}:{dataset_id}.{table_id} to {destination_uri} using client library."
    )


# [Preferred] Run using BigQuery DataFrames:
# extract_table_bigframes("my-bucket")

# Alternatively, run using google-cloud-bigquery client library:
# extract_table_client("my-bucket")
# [END bigquery_extract_table]
