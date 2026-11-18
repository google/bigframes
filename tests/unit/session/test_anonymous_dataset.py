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

import datetime
import gc
import unittest.mock as mock
import warnings
import weakref

import google.cloud.bigquery as bigquery
import pytest

import bigframes.core.events
import bigframes.exceptions
from bigframes.session import anonymous_dataset
from bigframes.testing import mocks


def _create_manager(
    bqclient: mock.Mock,
) -> anonymous_dataset.AnonymousDatasetManager:
    return anonymous_dataset.AnonymousDatasetManager(
        bqclient=bqclient,
        location="US",
        session_id="session123",
        publisher=bigframes.core.events.Publisher(),
    )


def _mock_bqclient_with_dataset(
    dataset_ref: bigquery.DatasetReference,
) -> mock.Mock:
    bqclient = mock.create_autospec(bigquery.Client, instance=True)
    query_job = mock.create_autospec(bigquery.QueryJob, instance=True)
    query_job._properties = {}
    type(query_job).destination = mock.PropertyMock(
        return_value=dataset_ref.table("anon_result")
    )
    bqclient.query.return_value = query_job
    return bqclient


def test_close_skips_udf_cleanup_when_dataset_unused():
    bqclient = mock.create_autospec(bigquery.Client, instance=True)
    manager = _create_manager(bqclient)

    manager.close()

    bqclient.query.assert_not_called()
    bqclient.list_routines.assert_not_called()
    bqclient.delete_table.assert_not_called()


def test_close_runs_udf_cleanup_once_when_dataset_used():
    dataset_ref = bigquery.DatasetReference("test-project", "_anon_dataset")
    bqclient = _mock_bqclient_with_dataset(dataset_ref)

    now = datetime.datetime.now(datetime.timezone.utc)

    old_scalar = mock.MagicMock(spec=bigquery.Routine)
    old_scalar.created = now - datetime.timedelta(days=10)
    old_scalar.reference = dataset_ref.routine("old_scalar")
    old_scalar._properties = {"routineType": "SCALAR_FUNCTION"}

    recent_scalar = mock.MagicMock(spec=bigquery.Routine)
    recent_scalar.created = now - datetime.timedelta(days=1)
    recent_scalar.reference = dataset_ref.routine("recent_scalar")
    recent_scalar._properties = {"routineType": "SCALAR_FUNCTION"}

    old_procedure = mock.MagicMock(spec=bigquery.Routine)
    old_procedure.created = now - datetime.timedelta(days=10)
    old_procedure.reference = dataset_ref.routine("old_procedure")
    old_procedure._properties = {"routineType": "PROCEDURE"}

    bqclient.list_routines.return_value = [old_scalar, recent_scalar, old_procedure]

    manager = _create_manager(bqclient)
    assert manager.dataset == dataset_ref

    manager.close()

    bqclient.list_routines.assert_called_once_with(dataset_ref)
    bqclient.delete_routine.assert_called_once()
    assert bqclient.delete_routine.call_args[0][0] == old_scalar.reference

    # Repeated close() (e.g. explicit close followed by __del__) should not
    # re-run list_routines / delete_routine.
    manager.close()
    bqclient.list_routines.assert_called_once_with(dataset_ref)
    bqclient.delete_routine.assert_called_once()


def test_close_warns_once_on_udf_cleanup_failure():
    dataset_ref = bigquery.DatasetReference("test-project", "_anon_dataset")
    bqclient = _mock_bqclient_with_dataset(dataset_ref)
    bqclient.list_routines.side_effect = RuntimeError("boom")

    manager = _create_manager(bqclient)
    _ = manager.dataset

    with pytest.warns(bigframes.exceptions.CleanupFailedWarning, match="boom"):
        manager.close()

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        manager.close()

    assert recorded == []
    bqclient.list_routines.assert_called_once_with(dataset_ref)


def test_gc_unclosed_unused_session_does_not_call_start_query_with_job():
    def _create_and_drop_session():
        session = mocks.create_bigquery_session()
        return weakref.ref(session)

    session_ref = _create_and_drop_session()

    with mock.patch(
        "bigframes.session._io.bigquery.start_query_with_job"
    ) as mock_start_query:
        gc.collect()

    assert session_ref() is None
    mock_start_query.assert_not_called()
