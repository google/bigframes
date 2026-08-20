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

import unittest.mock as mock

from google.oauth2 import service_account
import pytest

from . import client_json_credentials


def test_client_json_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_creds = mock.create_autospec(service_account.Credentials, instance=True)
    monkeypatch.setattr(
        service_account.Credentials,
        "from_service_account_file",
        lambda path: mock_creds,
    )
    with mock.patch("bigframes.pandas.read_gbq") as mock_read_gbq:
        mock_read_gbq.return_value = mock.MagicMock()
        res = client_json_credentials.client_json_credentials(key_path="fake/path.json")
        assert res is not None
        mock_read_gbq.assert_called_once()
