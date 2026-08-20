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

# [START bigquery_bigframes_client_json_credentials]
import bigframes.pandas as bpd
from google.oauth2 import service_account

# Set partial ordering mode for BigQuery DataFrames.
bpd.options.bigquery.ordering_mode = "partial"


def client_json_credentials(
    key_path: str = "path/to/your_service_account_key.json",
) -> bpd.DataFrame:
    # Explicitly use service account credentials from a JSON file.
    credentials = service_account.Credentials.from_service_account_file(key_path)
    bpd.options.bigquery.credentials = credentials

    sql = """
    SELECT name, SUM(number) as total_people
    FROM `bigquery-public-data.usa_names.usa_1910_2013`
    WHERE state = 'TX'
    GROUP BY name
    ORDER BY total_people DESC
    LIMIT 10
    """
    df = bpd.read_gbq(sql)
    return df


# [END bigquery_bigframes_client_json_credentials]


if __name__ == "__main__":
    import os

    key_path = os.environ.get(
        "GOOGLE_APPLICATION_CREDENTIALS",
        "path/to/your_service_account_key.json",
    )
    print(client_json_credentials(key_path=key_path))
