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
import decimal
from typing import Generator

import google.cloud.bigquery as bigquery
import pandas as pd
import pyarrow as pa
import pytest
import test_utils.prefixer

import bigframes
import bigframes.dtypes
import bigframes.exceptions as bfe
import bigframes.features
import bigframes.pandas as bpd

PUBLIC_TABLE_ID = (
    "bigquery-public-data.biglake-public-nyc-taxi-iceberg.public_data.nyc_taxicab_2021"
)
LAKEHOUSE_LOCATION = "us-central1"
LAKEHOUSE_PROJECT = "bigframes-dev"
LAKEHOUSE_DATASET = "bigframes_dev_iceberg_system_test_catalog.test_iceberg"

prefixer = test_utils.prefixer.Prefixer("bigframes", "tests/system")


def _quoted(table_id: str) -> str:
    project, catalog, namespace, table = table_id.split(".")
    return f"`{project}`.`{catalog}.{namespace}`.`{table}`"


def _create_lakehouse_table(bqclient: bigquery.Client, select_sql: str) -> str:
    table_id = f"{LAKEHOUSE_PROJECT}.{LAKEHOUSE_DATASET}.{prefixer.create_prefix()}"
    bqclient.query_and_wait(f"CREATE TABLE {_quoted(table_id)} AS {select_sql}")
    return table_id


def _drop_lakehouse_table(bqclient: bigquery.Client, table_id: str) -> None:
    bqclient.query_and_wait(f"DROP TABLE IF EXISTS {_quoted(table_id)}")


@pytest.fixture(scope="module", params=["strict", "partial"])
def lakehouse_session(request) -> Generator[bigframes.Session, None, None]:
    session = bigframes.Session(
        bigframes.BigQueryOptions(
            location=LAKEHOUSE_LOCATION, ordering_mode=request.param
        )
    )
    yield session
    session.close()


@pytest.fixture(scope="module")
def lakehouse_bqclient() -> bigquery.Client:
    bqclient = bigquery.Client(project=LAKEHOUSE_PROJECT, location=LAKEHOUSE_LOCATION)
    for table in bqclient.list_tables(
        bigquery.DatasetReference(LAKEHOUSE_PROJECT, LAKEHOUSE_DATASET)
    ):
        if prefixer.should_cleanup(table.table_id):
            _drop_lakehouse_table(
                bqclient, f"{LAKEHOUSE_PROJECT}.{LAKEHOUSE_DATASET}.{table.table_id}"
            )
    return bqclient


@pytest.fixture(scope="module")
def lakehouse_table_id(
    lakehouse_bqclient: bigquery.Client,
) -> Generator[str, None, None]:
    """A read-only Lakehouse table with 8 rows of mixed types."""
    table_id = _create_lakehouse_table(
        lakehouse_bqclient,
        """
        SELECT
          id,
          CONCAT('name_', CAST(id AS STRING)) AS name,
          id * 1.5 AS score,
          MOD(id, 2) = 0 AS flag,
          TIMESTAMP_ADD(TIMESTAMP '2026-01-01', INTERVAL id HOUR) AS created,
          DATE_ADD(DATE '2026-01-01', INTERVAL id DAY) AS start_date,
          CAST(id AS NUMERIC) / 4 AS amount,
          [CONCAT('t', CAST(id AS STRING)), 'common'] AS tags,
          STRUCT(id * 10 AS a, CONCAT('b', CAST(id AS STRING)) AS b) AS info
        FROM UNNEST(GENERATE_ARRAY(1, 8)) AS id
        """,
    )
    try:
        yield table_id
    finally:
        _drop_lakehouse_table(lakehouse_bqclient, table_id)


@pytest.fixture()
def lakehouse_mutable_table_id(
    lakehouse_bqclient: bigquery.Client,
) -> Generator[str, None, None]:
    table_id = _create_lakehouse_table(
        lakehouse_bqclient,
        "SELECT id FROM UNNEST(GENERATE_ARRAY(1, 3)) AS id",
    )
    try:
        yield table_id
    finally:
        _drop_lakehouse_table(lakehouse_bqclient, table_id)


def test_read_gbq_lakehouse_w_location():
    session = bigframes.Session(
        context=bigframes.BigQueryOptions(location="us-central1")
    )
    try:
        df = session.read_gbq(PUBLIC_TABLE_ID)
        assert len(df) > 0
    finally:
        session.close()


def test_read_gbq_lakehouse_w_wrong_location():
    session = bigframes.Session(
        context=bigframes.BigQueryOptions(location="europe-west1")
    )
    try:
        with pytest.raises(ValueError, match="Current session is in europe-west1"):
            session.read_gbq(PUBLIC_TABLE_ID)
    finally:
        session.close()


def test_read_gbq_lakehouse_wo_location(reset_default_session_and_location):
    assert not bpd.options.bigquery.location

    df = bpd.read_gbq(PUBLIC_TABLE_ID)

    assert bpd.options.bigquery.location == "us-central1"
    assert len(df) > 0


def test_read_gbq_table_lakehouse(lakehouse_session, lakehouse_table_id):
    df = lakehouse_session.read_gbq_table(lakehouse_table_id)

    result = df.to_pandas().sort_values("id").reset_index(drop=True)

    assert list(result.columns) == [
        "id",
        "name",
        "score",
        "flag",
        "created",
        "start_date",
        "amount",
        "tags",
        "info",
    ]
    assert result["id"].tolist() == list(range(1, 9))
    assert result["name"].tolist() == [f"name_{i}" for i in range(1, 9)]
    assert result["score"].tolist() == [i * 1.5 for i in range(1, 9)]
    assert result["flag"].tolist() == [i % 2 == 0 for i in range(1, 9)]
    assert result["created"].tolist() == [
        pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(hours=i)
        for i in range(1, 9)
    ]
    assert result["start_date"].tolist() == [
        datetime.date(2026, 1, 1) + datetime.timedelta(days=i) for i in range(1, 9)
    ]
    assert result["amount"].tolist() == [decimal.Decimal(i) / 4 for i in range(1, 9)]
    assert list(result["tags"].iloc[0]) == ["t1", "common"]
    assert result["info"].iloc[0] == {"a": 10, "b": "b1"}


def test_read_gbq_table_lakehouse_dtypes(lakehouse_session, lakehouse_table_id):
    df = lakehouse_session.read_gbq_table(lakehouse_table_id)

    expected: dict[str, object] = {
        "id": bigframes.dtypes.INT_DTYPE,
        "name": bigframes.dtypes.STRING_DTYPE,
        "score": bigframes.dtypes.FLOAT_DTYPE,
        "flag": bigframes.dtypes.BOOL_DTYPE,
        "created": bigframes.dtypes.TIMESTAMP_DTYPE,
        "start_date": bigframes.dtypes.DATE_DTYPE,
        "amount": bigframes.dtypes.NUMERIC_DTYPE,
        "tags": pd.ArrowDtype(pa.list_(pa.string())),
        "info": pd.ArrowDtype(pa.struct([("a", pa.int64()), ("b", pa.string())])),
    }
    assert df.dtypes.to_dict() == expected

    expected_downloaded = (
        expected
        if bigframes.features.PANDAS_VERSIONS.is_arrow_list_dtype_usable
        else {**expected, "tags": "object"}
    )
    assert df.to_pandas().dtypes.to_dict() == expected_downloaded


def test_read_gbq_table_lakehouse_w_columns_and_filters(
    lakehouse_session, lakehouse_table_id
):
    df = lakehouse_session.read_gbq_table(
        lakehouse_table_id,
        columns=["id", "name"],
        filters=[("id", ">", 5)],
    )

    result = df.to_pandas().sort_values("id").reset_index(drop=True)

    assert list(result.columns) == ["id", "name"]
    assert result["id"].tolist() == [6, 7, 8]


def test_read_gbq_table_lakehouse_w_index_col(lakehouse_session, lakehouse_table_id):
    df = lakehouse_session.read_gbq_table(lakehouse_table_id, index_col="id")

    result = df.to_pandas().sort_index()

    assert result.index.name == "id"
    assert result.index.tolist() == list(range(1, 9))
    assert "id" not in result.columns


def test_read_gbq_table_lakehouse_w_max_results(lakehouse_session, lakehouse_table_id):
    df = lakehouse_session.read_gbq_table(lakehouse_table_id, max_results=3)

    assert len(df.to_pandas()) == 3


def test_read_gbq_table_lakehouse_dry_run(lakehouse_session, lakehouse_table_id):
    stats = lakehouse_session.read_gbq_table(lakehouse_table_id, dry_run=True)

    assert not stats["isQuery"]
    assert stats["columnCount"] == 9
    assert stats["type"] == "TABLE"


def test_read_gbq_query_lakehouse(lakehouse_session, lakehouse_table_id):
    df = lakehouse_session.read_gbq(
        f"SELECT id FROM {_quoted(lakehouse_table_id)} WHERE flag"
    )

    assert sorted(df["id"].to_pandas().tolist()) == [2, 4, 6, 8]


def test_read_gbq_table_lakehouse_aggregate_and_join(
    lakehouse_session, lakehouse_table_id
):
    df = lakehouse_session.read_gbq_table(lakehouse_table_id, columns=["id", "flag"])
    labels = lakehouse_session.read_pandas(
        pd.DataFrame({"id": [1, 2, 3], "label": ["one", "two", "three"]})
    )

    joined = df.merge(labels, on="id").to_pandas().sort_values("id")

    assert int(df["id"].sum()) == 36
    assert int(df.groupby("flag")["id"].count().sum()) == 8
    assert joined["label"].tolist() == ["one", "two", "three"]


def test_read_gbq_colab_lakehouse_pyformat(lakehouse_session, lakehouse_table_id):
    df = lakehouse_session.read_gbq_table(lakehouse_table_id)

    result = lakehouse_session._read_gbq_colab(
        "SELECT COUNT(*) AS n FROM {df}", pyformat_args={"df": df}
    )

    assert result["n"].to_pandas().tolist() == [8]


def test_read_gbq_table_lakehouse_pinned_to_read_time(
    lakehouse_session, lakehouse_bqclient, lakehouse_mutable_table_id
):
    df = lakehouse_session.read_gbq_table(lakehouse_mutable_table_id)
    assert len(df.to_pandas()) == 3

    lakehouse_bqclient.query_and_wait(
        f"INSERT INTO {_quoted(lakehouse_mutable_table_id)} (id) VALUES (4)"
    )
    # Confirm the insert is visible, so the checks below can't pass vacuously.
    current = lakehouse_bqclient.query_and_wait(
        f"SELECT COUNT(*) AS n FROM {_quoted(lakehouse_mutable_table_id)}"
    )
    assert next(iter(current))["n"] == 4

    assert len(df.to_pandas()) == 3
    assert int(df["id"].count()) == 3
    assert int(df["id"].max()) == 3

    # Reading the same 4-part ID again in this session reuses the snapshot.
    with pytest.warns(bfe.TimeTravelCacheWarning):
        reread = lakehouse_session.read_gbq_table(lakehouse_mutable_table_id)
    assert len(reread.to_pandas()) == 3
