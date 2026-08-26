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

# [START bigquery_query_results_dataframe]
import bigframes.pandas as bpd
import pandas as pd
import pandas_gbq

# Set partial ordering mode for BigQuery DataFrames.
bpd.options.bigquery.ordering_mode = "partial"


def query_results_dataframe_bigframes() -> bpd.DataFrame:
    """Queries BigQuery using BigQuery DataFrames and returns a distributed DataFrame."""
    sql = """
    SELECT name, SUM(number) as total_people
    FROM `bigquery-public-data.usa_names.usa_1910_2013`
    WHERE state = 'TX'
    GROUP BY name
    ORDER BY total_people DESC
    LIMIT 20
    """
    df_bigframes = bpd.read_gbq(sql)
    print("Retrieved query results using BigQuery DataFrames.")
    return df_bigframes


def query_results_dataframe_pandas_gbq() -> pd.DataFrame:
    """Queries BigQuery using pandas-gbq directly and returns an in-memory DataFrame."""
    sql = """
    SELECT name, SUM(number) as total_people
    FROM `bigquery-public-data.usa_names.usa_1910_2013`
    WHERE state = 'TX'
    GROUP BY name
    ORDER BY total_people DESC
    LIMIT 20
    """
    df_pandas = pandas_gbq.read_gbq(sql)
    print("Retrieved query results using pandas-gbq.")
    return df_pandas


# [Preferred] Run using BigQuery DataFrames:
# df_bigframes = query_results_dataframe_bigframes()
# print(df_bigframes.head())

# Alternatively, run using pandas-gbq:
# df_pandas = query_results_dataframe_pandas_gbq()
# print(df_pandas.head())
# [END bigquery_query_results_dataframe]
