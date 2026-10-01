# Copyright 2024 Google LLC
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

from unittest import mock

import google.cloud.bigquery
import pytest

import bigframes._config.auth
import bigframes.dataframe
import bigframes.pandas
import bigframes.pandas.io.api as bf_io_api
import bigframes.session
import bigframes.session.clients

# _read_gbq_colab requires the polars engine.
pytest.importorskip("polars")


@mock.patch(
    "bigframes.pandas.io.api._set_default_session_location_if_possible_deferred_query"
)
@mock.patch("bigframes.core.global_session.with_default_session")
def test_read_gbq_colab_dry_run_doesnt_call_set_location(
    mock_with_default_session, mock_set_location
):
    """
    Ensure that we don't bind to a location too early. If it's a dry run, the
    user might not be done typing.
    """
    mock_df = mock.create_autospec(bigframes.dataframe.DataFrame)
    mock_with_default_session.return_value = mock_df

    query_or_table = "SELECT {param1} AS param1"
    sample_pyformat_args = {"param1": "value1"}
    bf_io_api._read_gbq_colab(
        query_or_table, pyformat_args=sample_pyformat_args, dry_run=True
    )

    mock_set_location.assert_not_called()


@mock.patch("bigframes._config.auth.pydata_google_auth.default")
@mock.patch("bigframes.core.global_session.with_default_session")
def test_read_gbq_colab_dry_run_doesnt_authenticate_multiple_times(
    mock_with_default_session, mock_get_credentials, monkeypatch
):
    """
    Ensure that we authenticate too often, which is an expensive operation,
    performance-wise (2+ seconds).
    """
    bigframes.pandas.close_session()

    mock_get_credentials.return_value = (mock.Mock(), "unit-test-project")
    mock_create_bq_client = mock.Mock()
    mock_bq_client = mock.create_autospec(google.cloud.bigquery.Client, instance=True)
    mock_create_bq_client.return_value = mock_bq_client
    mock_query_job = mock.create_autospec(google.cloud.bigquery.QueryJob, instance=True)
    type(mock_query_job).schema = mock.PropertyMock(return_value=[])
    mock_query_job._properties = {}
    mock_bq_client.query.return_value = mock_query_job
    monkeypatch.setattr(
        bigframes.session.clients.ClientsProvider,
        "_create_bigquery_client",
        mock_create_bq_client,
    )
    mock_df = mock.create_autospec(bigframes.dataframe.DataFrame)
    mock_with_default_session.return_value = mock_df

    bigframes._config.auth._cached_credentials = None
    query_or_table = "SELECT {param1} AS param1"
    sample_pyformat_args = {"param1": "value1"}
    bf_io_api._read_gbq_colab(
        query_or_table, pyformat_args=sample_pyformat_args, dry_run=True
    )

    mock_get_credentials.assert_called()
    mock_with_default_session.assert_not_called()
    mock_get_credentials.reset_mock()

    # Repeat the operation so that the credentials would have have been cached.
    bf_io_api._read_gbq_colab(
        query_or_table, pyformat_args=sample_pyformat_args, dry_run=True
    )
    mock_get_credentials.assert_not_called()


@mock.patch(
    "bigframes.pandas.io.api._set_default_session_location_if_possible_deferred_query"
)
@mock.patch("bigframes.core.global_session.with_default_session")
def test_read_gbq_colab_calls_set_location(
    mock_with_default_session, mock_set_location
):
    # Configure the mock for with_default_session to return a DataFrame mock
    mock_df = mock.create_autospec(bigframes.dataframe.DataFrame)
    mock_with_default_session.return_value = mock_df

    query_or_table = "SELECT {param1} AS param1"
    sample_pyformat_args = {"param1": "'value1'"}
    result = bf_io_api._read_gbq_colab(
        query_or_table, pyformat_args=sample_pyformat_args, dry_run=False
    )

    # Make sure that we format the SQL first to prevent syntax errors.
    formatted_query = "SELECT 'value1' AS param1"
    mock_set_location.assert_called_once()
    args, _ = mock_set_location.call_args
    assert formatted_query == args[0]()
    mock_with_default_session.assert_called_once()

    # Check the actual arguments passed to with_default_session
    args, kwargs = mock_with_default_session.call_args
    assert args[0] == bigframes.session.Session._read_gbq_colab
    assert args[1] == query_or_table
    assert kwargs["pyformat_args"] == sample_pyformat_args
    assert not kwargs["dry_run"]
    assert isinstance(result, bigframes.dataframe.DataFrame)


@pytest.mark.parametrize(
    "table_id",
    [
        # 4 parts, the same shape as an Iceberg REST catalog table ID.
        "my-project.region-us.INFORMATION_SCHEMA.SCHEMATA",
        "my-project.my_dataset.INFORMATION_SCHEMA.TABLES",
        # 3 parts, which tables.get would parse as project.dataset.table.
        "region-us.INFORMATION_SCHEMA.SCHEMATA",
        "my_dataset.INFORMATION_SCHEMA.TABLES",
    ],
)
@mock.patch("bigframes.pandas.io.api._get_bqclient_and_project")
def test_set_default_session_location_information_schema_uses_dry_run(
    mock_get_bqclient_and_project, table_id
):
    bigframes.pandas.close_session()
    bigframes.pandas.options.bigquery.location = None
    mock_bqclient = mock.create_autospec(google.cloud.bigquery.Client, instance=True)
    mock_query_job = mock.create_autospec(google.cloud.bigquery.QueryJob, instance=True)
    mock_query_job.location = "us-east4"
    type(mock_query_job).schema = mock.PropertyMock(return_value=[])
    mock_bqclient.query.return_value = mock_query_job
    mock_get_bqclient_and_project.return_value = (mock_bqclient, "default-project")

    try:
        bf_io_api._set_default_session_location_if_possible(table_id)

        mock_bqclient.query.assert_called_once()
        args, kwargs = mock_bqclient.query.call_args
        assert table_id in args[0]
        assert kwargs["job_config"].dry_run
        mock_bqclient.get_table.assert_not_called()
        assert bigframes.pandas.options.bigquery.location == "us-east4"
    finally:
        bigframes.pandas.options.bigquery.location = None


@pytest.mark.parametrize(
    "table_id",
    [
        "my-project.my_dataset.my_table",
        # Names that contain INFORMATION_SCHEMA but aren't INFORMATION_SCHEMA
        # views.
        "my-project.MY_INFORMATION_SCHEMA.TABLES",
        "my-project.my_dataset.INFORMATION_SCHEMA",
    ],
)
@mock.patch("bigframes.pandas.io.api._get_bqclient_and_project")
def test_set_default_session_location_table_uses_get_table(
    mock_get_bqclient_and_project, table_id
):
    bigframes.pandas.close_session()
    bigframes.pandas.options.bigquery.location = None
    mock_bqclient = mock.create_autospec(google.cloud.bigquery.Client, instance=True)
    mock_table = mock.create_autospec(google.cloud.bigquery.Table, instance=True)
    mock_table.location = "asia-northeast1"
    mock_bqclient.get_table.return_value = mock_table
    mock_get_bqclient_and_project.return_value = (mock_bqclient, "default-project")

    try:
        bf_io_api._set_default_session_location_if_possible(table_id)

        mock_bqclient.get_table.assert_called_once_with(table_id)
        mock_bqclient.query.assert_not_called()
        assert bigframes.pandas.options.bigquery.location == "asia-northeast1"
    finally:
        bigframes.pandas.options.bigquery.location = None


@mock.patch("bigframes.pandas.io.api._get_bqclient_and_project")
def test_set_default_session_location_four_part_lakehouse_table(
    mock_get_bqclient_and_project,
):
    bigframes.pandas.close_session()
    bigframes.pandas.options.bigquery.location = None
    mock_bqclient = mock.create_autospec(google.cloud.bigquery.Client, instance=True)
    mock_table = mock.create_autospec(google.cloud.bigquery.Table, instance=True)
    mock_table.location = "us-central1"
    mock_bqclient.get_table.return_value = mock_table
    mock_get_bqclient_and_project.return_value = (mock_bqclient, "default-project")

    try:
        bf_io_api._set_default_session_location_if_possible(
            "my-project.my_catalog.my_namespace.my_table"
        )

        mock_bqclient.get_table.assert_called_once_with(
            "my-project.my_catalog.my_namespace.my_table"
        )
        assert bigframes.pandas.options.bigquery.location == "us-central1"
    finally:
        bigframes.pandas.options.bigquery.location = None
