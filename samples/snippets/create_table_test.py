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
from google.cloud import bigquery

from . import create_table


def test_create_table_bigframes(project_id: str, dataset_id: str) -> None:
    table_id = f"{project_id}.{dataset_id}.test_create_table_bigframes"

    create_table.create_table_bigframes(table_id=table_id)

    df_bf = bpd.read_gbq(table_id)
    assert len(df_bf) == 2


def test_create_table_client(
    project_id: str, dataset_id: str, bigquery_client: bigquery.Client
) -> None:
    table_id = f"{project_id}.{dataset_id}.test_create_table_client"

    create_table.create_table_client(table_id=table_id)

    table = bigquery_client.get_table(table_id)
    assert table is not None
    assert len(table.schema) == 2


def test_create_table_pandas_gbq(project_id: str, dataset_id: str) -> None:
    table_id = f"{project_id}.{dataset_id}.test_create_table_pandas"

    create_table.create_table_pandas_gbq(table_id=table_id)

    df_pandas = bpd.read_gbq(table_id)
    assert len(df_pandas) == 2
