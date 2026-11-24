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

"""Unit tests for the SQL that the streaming exports send to BigQuery."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import bigframes
from bigframes.streaming import dataframe as streaming_dataframe
from bigframes.testing import mocks

pytestmark = pytest.mark.filterwarnings("ignore::bigframes.exceptions.PreviewWarning")

TABLE_ID = "test-project.test_dataset.test_table"
BOOL_OPTIONS = ("truncate", "overwrite", "auto_create_column_families")

# Values that can never be a Bigtable or Pub/Sub resource ID.
INVALID_RESOURCE_IDS = (
    "../../victim-project/instances/victim-inst",
    't") AS (SELECT session_user() AS rowkey); -- ',
    'x", format = "CLOUD_BIGTABLE',
    None,
    "",
    ".",
    "..",
    "a/b",
    "a\\b",
    "a'b",
    'a"b',
    "a`b",
    "a?b",
    "a#b",
    "a b",
    "a\tb",
    "a\nb",
    "a\u3000b",
    "a\x00b",
    "a\x1bb",
    "a\x7fb",
    "a\x85b",
    "a\x9bb",
)


@pytest.fixture
def session() -> bigframes.Session:
    return mocks.create_bigquery_session()


def _export_queries(session: bigframes.Session) -> list[str]:
    queries = session._queries  # type: ignore
    return [query for query in queries if query.startswith("EXPORT DATA")]


def _export_query(session: bigframes.Session) -> str:
    (query,) = _export_queries(session)
    return query


def _string_literal(text: str) -> str:
    """Quotes printable ASCII text as a GoogleSQL string literal."""
    return "'" + text.replace("\\", "\\\\").replace("'", "\\'") + "'"


def test_to_bigtable_sql(session):
    streaming_dataframe._to_bigtable(
        "SELECT 1",
        instance="my-instance",
        table="my-table",
        session=session,
        app_profile="my-profile",
        truncate=True,
        overwrite=False,
        auto_create_column_families=True,
        bigtable_options={"columnFamilies": [{"familyId": "f"}]},
    )

    assert _export_query(session) == (
        "EXPORT DATA\n"
        "OPTIONS (\n"
        "format = 'CLOUD_BIGTABLE',\n"
        """bigtable_options = '{"columnFamilies": [{"familyId": "f"}]}',\n"""
        "truncate = TRUE,\n"
        "overwrite = FALSE,\n"
        "auto_create_column_families = TRUE,\n"
        "uri = 'https://bigtable.googleapis.com/projects/test-project"
        "/instances/my-instance/appProfiles/my-profile/tables/my-table'\n"
        ")\n"
        "AS (\n"
        "SELECT 1);"
    )


def test_to_pubsub_sql(session):
    streaming_dataframe._to_pubsub(
        "SELECT 1",
        topic="my-topic",
        service_account_email="sa@example.com",
        session=session,
    )

    assert _export_query(session) == (
        "EXPORT DATA\n"
        "OPTIONS (\n"
        "format = 'CLOUD_PUBSUB',\n"
        "uri = 'https://pubsub.googleapis.com/projects/test-project"
        "/topics/my-topic'\n"
        ")\n"
        "AS (\n"
        "SELECT 1);"
    )


def test_streaming_dataframe_to_bigtable(session):
    sdf = session.read_gbq_table_streaming(TABLE_ID)

    sdf.to_bigtable(instance="my-instance", table="my-table", truncate=np.True_)

    query = _export_query(session)
    assert "\ntruncate = TRUE,\n" in query
    assert (
        "\nuri = 'https://bigtable.googleapis.com/projects/test-project"
        "/instances/my-instance/tables/my-table'\n"
    ) in query
    assert "APPENDS(TABLE `test-project`.`test_dataset`.`test_table`" in query


def test_streaming_dataframe_to_pubsub_rejects_invalid_topic(session):
    sdf = session.read_gbq_table_streaming(TABLE_ID)

    with pytest.raises(ValueError, match="^Invalid topic "):
        sdf.to_pubsub(
            topic='x", format = "CLOUD_BIGTABLE',
            service_account_email="sa@example.com",
        )

    assert _export_queries(session) == []


@pytest.mark.parametrize("resource_id", INVALID_RESOURCE_IDS)
def test_check_resource_id_rejects_invalid_ids(resource_id):
    with pytest.raises(ValueError, match="^Invalid table "):
        streaming_dataframe._check_resource_id(resource_id, "table")


@pytest.mark.parametrize(
    "resource_id",
    [
        "my-instance",
        "a..b",
        "t.1",
        "_x-1",
        "123",
        123,
        "a~b+c%d",
        "my%20topic",
        "a%2Fb",
    ],
)
def test_check_resource_id_keeps_valid_ids(resource_id):
    checked = streaming_dataframe._check_resource_id(resource_id, "table")

    assert checked == str(resource_id)


@pytest.mark.parametrize(
    ("param_name", "resource_id"),
    [
        ("instance", "../../victim-project/instances/victim-inst"),
        ("table", 't") AS (SELECT session_user() AS rowkey); -- '),
        ("app_profile", 'x", format = "CLOUD_BIGTABLE'),
        # An empty app profile is rejected, not treated as no app profile.
        ("app_profile", ""),
    ],
)
def test_to_bigtable_rejects_invalid_resource_id(session, param_name, resource_id):
    kwargs: dict[str, Any] = {"instance": "my-instance", "table": "my-table"}
    kwargs[param_name] = resource_id

    with pytest.raises(ValueError, match=f"^Invalid {param_name} "):
        streaming_dataframe._to_bigtable("SELECT 1", session=session, **kwargs)

    assert _export_queries(session) == []


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, "TRUE"),
        (False, "FALSE"),
        (np.True_, "TRUE"),
        (np.False_, "FALSE"),
        (pa.scalar(True), "TRUE"),
        (pa.scalar(False), "FALSE"),
        ("true", "TRUE"),
        ("false", "FALSE"),
        ("FALSE", "FALSE"),
        (" False\n", "FALSE"),
    ],
)
def test_bool_option_sql(value, expected):
    assert streaming_dataframe._bool_option_sql(value, "truncate") == expected


@pytest.mark.parametrize(
    "value",
    [
        "false, uri = 'https://bigtable.googleapis.com/projects/victim/instances/v/tables/t'",
        "(false)",
        "NOT true",
        "yes",
        "",
        "0",
        None,
        pd.NA,
        0,
        1,
        1.0,
        b"false",
        np.array([True]),
    ],
)
def test_bool_option_sql_rejects_non_bool(value):
    with pytest.raises(ValueError, match="^truncate must be a bool"):
        streaming_dataframe._bool_option_sql(value, "truncate")


@pytest.mark.parametrize("param_name", BOOL_OPTIONS)
def test_to_bigtable_bool_options(session, param_name):
    kwargs: dict[str, Any] = {param_name: True}

    streaming_dataframe._to_bigtable(
        "SELECT 1",
        instance="my-instance",
        table="my-table",
        session=session,
        **kwargs,
    )

    query = _export_query(session)
    for name in BOOL_OPTIONS:
        expected = "TRUE" if name == param_name else "FALSE"
        assert f"\n{name} = {expected},\n" in query


@pytest.mark.parametrize("param_name", BOOL_OPTIONS)
def test_to_bigtable_rejects_non_bool_options(session, param_name):
    kwargs: dict[str, Any] = {param_name: "NOT true"}

    with pytest.raises(ValueError, match=f"^{param_name} must be a bool"):
        streaming_dataframe._to_bigtable(
            "SELECT 1",
            instance="my-instance",
            table="my-table",
            session=session,
            **kwargs,
        )

    assert _export_queries(session) == []


@pytest.mark.parametrize(
    "bigtable_options",
    [
        {},
        {
            "columnFamilies": [
                {
                    "familyId": "cf",
                    "columns": [{"qualifierString": "q", "fieldName": "x"}],
                }
            ]
        },
        {"k": 'a"""b', "j": "trailing\\"},
        {"familyId": 'f", "injected": "yes'},
        {"q": "a\\tb", "n": "line\nbreak"},
        {"s": "it's", "t": "'''"},
        {"u": "tabl\u00e9 \U0001f600"},
    ],
)
def test_to_bigtable_escapes_bigtable_options(session, bigtable_options):
    streaming_dataframe._to_bigtable(
        "SELECT 1",
        instance="my-instance",
        table="my-table",
        session=session,
        bigtable_options=bigtable_options,
    )

    expected = _string_literal(json.dumps(bigtable_options))
    assert f"\nbigtable_options = {expected},\n" in _export_query(session)


def test_to_bigtable_bigtable_options_backslash(session):
    streaming_dataframe._to_bigtable(
        "SELECT 1",
        instance="my-instance",
        table="my-table",
        session=session,
        bigtable_options={"q": "a\\tb"},
    )

    # json.dumps writes the backslash as \\ and the SQL literal doubles each
    # backslash again, so BigQuery reads back exactly what json.dumps wrote.
    assert r"""bigtable_options = '{"q": "a\\\\tb"}',""" in _export_query(session)


def test_to_bigtable_escapes_uri(session):
    session.bqclient.project = "my'project"

    streaming_dataframe._to_bigtable(
        "SELECT 1", instance="my-instance", table="my-table", session=session
    )

    assert (
        "\nuri = 'https://bigtable.googleapis.com/projects/my\\'project"
        "/instances/my-instance/tables/my-table'\n"
    ) in _export_query(session)


def test_to_pubsub_escapes_uri(session):
    session.bqclient.project = "my'project"

    streaming_dataframe._to_pubsub(
        "SELECT 1",
        topic="my-topic",
        service_account_email="sa@example.com",
        session=session,
    )

    assert (
        "\nuri = 'https://pubsub.googleapis.com/projects/my\\'project"
        "/topics/my-topic'\n"
    ) in _export_query(session)
