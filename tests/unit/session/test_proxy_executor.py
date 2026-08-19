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

from unittest import mock

import google.cloud.bigquery as bigquery
import google.cloud.exceptions
import pytest

import bigframes
from bigframes.session.proxy_executor import DualCompilerProxyExecutor


@pytest.fixture
def proxy_executor():
    """Creates a DualCompilerProxyExecutor instance with mocked dependencies."""
    bqclient = mock.create_autospec(bigquery.Client)
    bqclient.project = "test-project"
    storage_manager = mock.Mock()
    bqstoragereadclient = mock.Mock()
    loader = mock.Mock()
    publisher = mock.Mock()
    function_manager = mock.Mock()
    return DualCompilerProxyExecutor(
        bqclient,
        storage_manager,
        bqstoragereadclient,
        loader,
        publisher=publisher,
        function_manager=function_manager,
    )


def test_execute_legacy_routes_to_ibis(proxy_executor, monkeypatch):
    array_value = mock.Mock(spec=bigframes.core.ArrayValue)
    execution_spec = mock.Mock(spec=bigframes.session.execution_spec.ExecutionSpec)
    execution_spec.with_bq_labels.return_value = execution_spec

    proxy_executor._ibis_executor = mock.Mock()
    proxy_executor._sqlglot_executor = mock.Mock()

    monkeypatch.setattr(bigframes.options.experiments, "sql_compiler", "legacy")
    proxy_executor.execute(array_value, execution_spec)

    execution_spec.with_bq_labels.assert_called_once_with(
        {"bigframes-compiler": "ibis"}
    )
    proxy_executor._ibis_executor.execute.assert_called_once_with(
        array_value, execution_spec
    )
    proxy_executor._sqlglot_executor.execute.assert_not_called()


def test_execute_experimental_routes_to_sqlglot(proxy_executor, monkeypatch):
    array_value = mock.Mock(spec=bigframes.core.ArrayValue)
    execution_spec = mock.Mock(spec=bigframes.session.execution_spec.ExecutionSpec)
    execution_spec.with_bq_labels.return_value = execution_spec

    proxy_executor._ibis_executor = mock.Mock()
    proxy_executor._sqlglot_executor = mock.Mock()

    monkeypatch.setattr(bigframes.options.experiments, "sql_compiler", "experimental")
    proxy_executor.execute(array_value, execution_spec)

    execution_spec.with_bq_labels.assert_called_once_with(
        {"bigframes-compiler": "sqlglot"}
    )
    proxy_executor._sqlglot_executor.execute.assert_called_once_with(
        array_value, execution_spec
    )
    proxy_executor._ibis_executor.execute.assert_not_called()


def test_execute_stable_routes_to_sqlglot_success(proxy_executor, monkeypatch):
    array_value = mock.Mock(spec=bigframes.core.ArrayValue)
    execution_spec = mock.Mock(spec=bigframes.session.execution_spec.ExecutionSpec)
    execution_spec.with_bq_labels.return_value = execution_spec

    proxy_executor._ibis_executor = mock.Mock()
    proxy_executor._sqlglot_executor = mock.Mock()

    monkeypatch.setattr(bigframes.options.experiments, "sql_compiler", "stable")
    with mock.patch("uuid.uuid1") as mock_uuid:
        mock_uuid.return_value.hex = "1234567890123456"
        proxy_executor.execute(array_value, execution_spec)

    execution_spec.with_bq_labels.assert_called_once_with(
        {"bigframes-compiler": "sqlglot-123456789012"}
    )
    proxy_executor._sqlglot_executor.execute.assert_called_once_with(
        array_value, execution_spec
    )
    proxy_executor._ibis_executor.execute.assert_not_called()


def test_execute_stable_routes_to_sqlglot_fallback_to_ibis(proxy_executor, monkeypatch):
    array_value = mock.Mock(spec=bigframes.core.ArrayValue)
    execution_spec = mock.Mock(spec=bigframes.session.execution_spec.ExecutionSpec)

    spec_sqlglot = mock.Mock(spec=bigframes.session.execution_spec.ExecutionSpec)
    spec_ibis = mock.Mock(spec=bigframes.session.execution_spec.ExecutionSpec)
    execution_spec.with_bq_labels.side_effect = [spec_sqlglot, spec_ibis]

    proxy_executor._ibis_executor = mock.Mock()
    proxy_executor._sqlglot_executor = mock.Mock()

    proxy_executor._sqlglot_executor.execute.side_effect = (
        google.cloud.exceptions.BadRequest("test error")
    )

    monkeypatch.setattr(bigframes.options.experiments, "sql_compiler", "stable")
    with mock.patch("uuid.uuid1") as mock_uuid:
        mock_uuid.return_value.hex = "1234567890123456"
        with pytest.warns(
            UserWarning, match="Compiler ID 123456789012: Exception on sqlglot"
        ):
            proxy_executor.execute(array_value, execution_spec)

    execution_spec.with_bq_labels.assert_has_calls(
        [
            mock.call({"bigframes-compiler": "sqlglot-123456789012"}),
            mock.call({"bigframes-compiler": "ibis-123456789012"}),
        ]
    )

    proxy_executor._sqlglot_executor.execute.assert_called_once_with(
        array_value, spec_sqlglot
    )
    proxy_executor._ibis_executor.execute.assert_called_once_with(
        array_value, spec_ibis
    )


def test_cached_legacy_routes_to_ibis(proxy_executor, monkeypatch):
    array_value = mock.Mock(spec=bigframes.core.ArrayValue)
    config = mock.Mock()

    proxy_executor._ibis_executor = mock.Mock()
    proxy_executor._sqlglot_executor = mock.Mock()

    monkeypatch.setattr(bigframes.options.experiments, "sql_compiler", "legacy")
    proxy_executor.cached(array_value, config=config)

    proxy_executor._ibis_executor.cached.assert_called_once_with(
        array_value, config=config
    )
    proxy_executor._sqlglot_executor.cached.assert_not_called()


def test_cached_experimental_routes_to_sqlglot(proxy_executor, monkeypatch):
    array_value = mock.Mock(spec=bigframes.core.ArrayValue)
    config = mock.Mock()

    proxy_executor._ibis_executor = mock.Mock()
    proxy_executor._sqlglot_executor = mock.Mock()

    monkeypatch.setattr(bigframes.options.experiments, "sql_compiler", "experimental")
    proxy_executor.cached(array_value, config=config)

    proxy_executor._sqlglot_executor.cached.assert_called_once_with(
        array_value, config=config
    )
    proxy_executor._ibis_executor.cached.assert_not_called()


def test_cached_stable_routes_to_sqlglot_fallback_to_ibis(proxy_executor, monkeypatch):
    array_value = mock.Mock(spec=bigframes.core.ArrayValue)
    config = mock.Mock()

    proxy_executor._ibis_executor = mock.Mock()
    proxy_executor._sqlglot_executor = mock.Mock()

    proxy_executor._sqlglot_executor.cached.side_effect = (
        google.cloud.exceptions.BadRequest("test error")
    )

    monkeypatch.setattr(bigframes.options.experiments, "sql_compiler", "stable")
    with mock.patch("uuid.uuid1") as mock_uuid:
        mock_uuid.return_value.hex = "1234567890123456"
        with pytest.warns(
            UserWarning, match="Compiler ID 123456789012: Exception on sqlglot"
        ):
            proxy_executor.cached(array_value, config=config)

    proxy_executor._sqlglot_executor.cached.assert_called_once_with(
        array_value, config=config
    )
    proxy_executor._ibis_executor.cached.assert_called_once_with(
        array_value, config=config
    )


def test_dry_run_legacy_routes_to_ibis(proxy_executor, monkeypatch):
    array_value = mock.Mock(spec=bigframes.core.ArrayValue)

    proxy_executor._ibis_executor = mock.Mock()
    proxy_executor._sqlglot_executor = mock.Mock()

    monkeypatch.setattr(bigframes.options.experiments, "sql_compiler", "legacy")
    proxy_executor.dry_run(array_value, ordered=True)

    proxy_executor._ibis_executor.dry_run.assert_called_once_with(
        array_value, ordered=True, labels={"bigframes-compiler": "ibis"}
    )
    proxy_executor._sqlglot_executor.dry_run.assert_not_called()


def test_dry_run_experimental_routes_to_sqlglot(proxy_executor, monkeypatch):
    array_value = mock.Mock(spec=bigframes.core.ArrayValue)

    proxy_executor._ibis_executor = mock.Mock()
    proxy_executor._sqlglot_executor = mock.Mock()

    monkeypatch.setattr(bigframes.options.experiments, "sql_compiler", "experimental")
    proxy_executor.dry_run(array_value, ordered=False)

    proxy_executor._sqlglot_executor.dry_run.assert_called_once_with(
        array_value, ordered=False, labels={"bigframes-compiler": "sqlglot"}
    )
    proxy_executor._ibis_executor.dry_run.assert_not_called()


def test_dry_run_stable_routes_to_sqlglot_success(proxy_executor, monkeypatch):
    array_value = mock.Mock(spec=bigframes.core.ArrayValue)

    proxy_executor._ibis_executor = mock.Mock()
    proxy_executor._sqlglot_executor = mock.Mock()

    monkeypatch.setattr(bigframes.options.experiments, "sql_compiler", "stable")
    with mock.patch("uuid.uuid1") as mock_uuid:
        mock_uuid.return_value.hex = "1234567890123456"
        proxy_executor.dry_run(array_value)

    proxy_executor._sqlglot_executor.dry_run.assert_called_once_with(
        array_value,
        ordered=True,
        labels={"bigframes-compiler": "sqlglot-123456789012"},
    )
    proxy_executor._ibis_executor.dry_run.assert_not_called()


def test_dry_run_stable_routes_to_sqlglot_fallback_to_ibis(proxy_executor, monkeypatch):
    array_value = mock.Mock(spec=bigframes.core.ArrayValue)

    proxy_executor._ibis_executor = mock.Mock()
    proxy_executor._sqlglot_executor = mock.Mock()

    proxy_executor._sqlglot_executor.dry_run.side_effect = (
        google.cloud.exceptions.BadRequest("test error")
    )

    monkeypatch.setattr(bigframes.options.experiments, "sql_compiler", "stable")
    with mock.patch("uuid.uuid1") as mock_uuid:
        mock_uuid.return_value.hex = "1234567890123456"
        with pytest.warns(
            UserWarning, match="Compiler ID 123456789012: Exception on sqlglot"
        ):
            proxy_executor.dry_run(array_value)

    proxy_executor._sqlglot_executor.dry_run.assert_called_once_with(
        array_value,
        ordered=True,
        labels={"bigframes-compiler": "sqlglot-123456789012"},
    )
    proxy_executor._ibis_executor.dry_run.assert_called_once_with(
        array_value,
        ordered=True,
        labels={"bigframes-compiler": "ibis-123456789012"},
    )
