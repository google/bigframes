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

from . import query_results_dataframe


def test_query_results_dataframe_bigframes() -> None:
    df_bf = query_results_dataframe.query_results_dataframe_bigframes()

    assert df_bf is not None
    assert len(df_bf) == 20
    assert "total_people" in df_bf.columns


def test_query_results_dataframe_pandas_gbq() -> None:
    df_pandas = query_results_dataframe.query_results_dataframe_pandas_gbq()

    assert df_pandas is not None
    assert len(df_pandas) == 20
    assert "total_people" in df_pandas.columns
