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

from typing import Generator

from google.cloud import storage
import pytest

from . import extract_table_json


@pytest.fixture(autouse=True)
def cleanup_exported_blobs(
    gcs_bucket: str, storage_client: storage.Client
) -> Generator[None, None, None]:
    bucket = storage_client.get_bucket(gcs_bucket)
    for blob in bucket.list_blobs(prefix="shakespeare-"):
        blob.delete()
    try:
        yield
    finally:
        for blob in bucket.list_blobs(prefix="shakespeare-"):
            blob.delete()


def test_extract_table_json_bigframes(
    gcs_bucket: str, storage_client: storage.Client
) -> None:
    extract_table_json.extract_table_json_bigframes(bucket_name=gcs_bucket)

    bucket = storage_client.get_bucket(gcs_bucket)
    blobs = list(bucket.list_blobs(prefix="shakespeare-"))
    assert len(blobs) > 0


def test_extract_table_json_client(
    gcs_bucket: str, storage_client: storage.Client
) -> None:
    extract_table_json.extract_table_json_client(bucket_name=gcs_bucket)

    bucket = storage_client.get_bucket(gcs_bucket)
    blobs = list(bucket.list_blobs(prefix="shakespeare-"))
    assert len(blobs) > 0
