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

# [START bigquery_pandas_gbq_to_gbq_simple]
import bigframes.pandas as bpd
import pandas as pd
import pandas_gbq

# Set partial ordering mode for BigQuery DataFrames.
bpd.options.bigquery.ordering_mode = "partial"


def pandas_to_gbq_simple_pandas_gbq(
    destination_table: str = "your-project.your_dataset.your_table_name",
) -> None:
    """Uploads a pandas DataFrame to BigQuery using pandas-gbq directly."""
    df = pd.DataFrame(
        {
            "my_string": ["a", "b", "c"],
            "my_int64": [1, 2, 3],
            "my_float64": [4.0, 5.0, 6.0],
            "my_bool1": [True, False, True],
            "my_bool2": [False, True, False],
            "my_dates": pd.date_range("now", periods=3),
        }
    )
    pandas_gbq.to_gbq(df, destination_table, if_exists="replace")
    print(f"Uploaded DataFrame to {destination_table} using pandas-gbq.")


def pandas_to_gbq_simple_bigframes(
    destination_table: str = "your-project.your_dataset.your_table_name",
) -> None:
    """Uploads a pandas DataFrame to BigQuery using BigQuery DataFrames."""
    df = pd.DataFrame(
        {
            "my_string": ["a", "b", "c"],
            "my_int64": [1, 2, 3],
            "my_float64": [4.0, 5.0, 6.0],
            "my_bool1": [True, False, True],
            "my_bool2": [False, True, False],
            "my_dates": pd.date_range("now", periods=3),
        }
    )
    bq_df = bpd.read_pandas(df)
    bq_df.to_gbq(destination_table, if_exists="replace")
    print(f"Uploaded DataFrame to {destination_table} using BigQuery DataFrames.")


# [Preferred] Run using pandas-gbq:
# pandas_to_gbq_simple_pandas_gbq("your-project.your_dataset.your_table_name")

# Alternatively, run using BigQuery DataFrames:
# pandas_to_gbq_simple_bigframes("your-project.your_dataset.your_table_name")
# [END bigquery_pandas_gbq_to_gbq_simple]
