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

from . import pandas_to_gbq_simple


def test_pandas_to_gbq_simple_pandas_gbq(project_id: str, dataset_id: str) -> None:
    table_id = f"{project_id}.{dataset_id}.test_pandas_to_gbq_pandas"

    pandas_to_gbq_simple.pandas_to_gbq_simple_pandas_gbq(destination_table=table_id)

    df_loaded = bpd.read_gbq(table_id)
    assert len(df_loaded) == 3


def test_pandas_to_gbq_simple_bigframes(project_id: str, dataset_id: str) -> None:
    table_id = f"{project_id}.{dataset_id}.test_pandas_to_gbq_bigframes"

    pandas_to_gbq_simple.pandas_to_gbq_simple_bigframes(destination_table=table_id)

    df_loaded = bpd.read_gbq(table_id)
    assert len(df_loaded) == 3
