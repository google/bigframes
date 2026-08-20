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

from . import load_table_gcs_csv


def test_load_table_gcs_csv_bigframes(project_id: str, dataset_id: str) -> None:
    table_id = f"{project_id}.{dataset_id}.test_load_csv_bigframes"

    load_table_gcs_csv.load_table_gcs_csv_bigframes(table_id=table_id)

    df_loaded = bpd.read_gbq(table_id)
    assert len(df_loaded) == 50


def test_load_table_gcs_csv_client(project_id: str, dataset_id: str) -> None:
    table_id = f"{project_id}.{dataset_id}.test_load_csv_client"

    load_table_gcs_csv.load_table_gcs_csv_client(table_id=table_id)

    df_loaded = bpd.read_gbq(table_id)
    assert len(df_loaded) == 50
