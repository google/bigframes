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

import google.api_core.exceptions
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


def _create_lakehouse_table(
    bqclient: bigquery.Client, select_sql: str, *, column_definitions: str = ""
) -> str:
    table_id = f"{LAKEHOUSE_PROJECT}.{LAKEHOUSE_DATASET}.{prefixer.create_prefix()}"
    columns_sql = f" ({column_definitions})" if column_definitions else ""
    bqclient.query_and_wait(
        f"CREATE TABLE {_quoted(table_id)}{columns_sql} AS {select_sql}"
    )
    return table_id


def _drop_lakehouse_table(bqclient: bigquery.Client, table_id: str) -> None:
    bqclient.query_and_wait(f"DROP TABLE IF EXISTS {_quoted(table_id)}")


def _read_ids(bqclient: bigquery.Client, table_id: str) -> list[int]:
    rows = bqclient.query_and_wait(f"SELECT id FROM {_quoted(table_id)} ORDER BY id")
    return [row["id"] for row in rows]


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


@pytest.fixture(scope="module")
def regular_table_id(
    lakehouse_bqclient: bigquery.Client,
) -> Generator[str, None, None]:
    """A regular BigQuery table, in the same location as the Lakehouse tables."""
    dataset = bigquery.Dataset(f"{LAKEHOUSE_PROJECT}.{prefixer.create_prefix()}")
    dataset.location = LAKEHOUSE_LOCATION
    lakehouse_bqclient.create_dataset(dataset)
    table_id = f"{LAKEHOUSE_PROJECT}.{dataset.dataset_id}.parities"
    try:
        lakehouse_bqclient.query_and_wait(
            f"""
            CREATE TABLE `{table_id}` AS
            SELECT * FROM UNNEST([
              STRUCT(TRUE AS flag, 'even' AS parity),
              (FALSE, 'odd')
            ])
            """
        )
        yield table_id
    finally:
        lakehouse_bqclient.delete_dataset(dataset, delete_contents=True)


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


@pytest.mark.parametrize(
    ("write_engine", "num_rows"),
    [
        pytest.param("bigquery_inline", 3, id="inline"),
        pytest.param("bigquery_load", 3, id="load"),
        # Over 5,000 bytes, so the data stays local until to_gbq() uploads it.
        pytest.param("_deferred", 1_000, id="deferred"),
        # TODO(b/502298527): Reenable bigquery_write test
        # pytest.param("bigquery_write", 3, id="write_api"),
    ],
)
def test_to_gbq_lakehouse_append_w_local_data(
    lakehouse_session,
    lakehouse_bqclient,
    lakehouse_mutable_table_id,
    write_engine,
    num_rows,
):
    new_ids = list(range(100, 100 + num_rows))
    df = lakehouse_session.read_pandas(
        pd.DataFrame({"id": new_ids}), write_engine=write_engine
    )

    destination = df.to_gbq(lakehouse_mutable_table_id, if_exists="append", index=False)

    assert destination == lakehouse_mutable_table_id
    assert _read_ids(lakehouse_bqclient, lakehouse_mutable_table_id) == (
        [1, 2, 3] + new_ids
    )


def test_to_gbq_lakehouse_replace_keeps_history(
    lakehouse_session, lakehouse_bqclient, lakehouse_mutable_table_id
):
    now = lakehouse_bqclient.query_and_wait("SELECT CURRENT_TIMESTAMP() AS now")
    before_replace = next(iter(now))["now"]
    df = lakehouse_session.read_pandas(pd.DataFrame({"id": [7, 8]}))

    df.to_gbq(lakehouse_mutable_table_id, if_exists="replace", index=False)

    # Confirm the replace happened, so the check below can't pass vacuously.
    assert _read_ids(lakehouse_bqclient, lakehouse_mutable_table_id) == [7, 8]
    history = lakehouse_bqclient.query_and_wait(
        f"SELECT id FROM {_quoted(lakehouse_mutable_table_id)} "
        f"FOR SYSTEM_TIME AS OF '{before_replace.isoformat()}' ORDER BY id"
    )
    assert [row["id"] for row in history] == [1, 2, 3]


def test_to_gbq_lakehouse_w_transformed_lakehouse_table(
    lakehouse_session, lakehouse_bqclient, lakehouse_table_id, regular_table_id
):
    destination = _create_lakehouse_table(
        lakehouse_bqclient,
        "SELECT TRUE AS flag, 0 AS total, 'even' AS parity, 'E' AS label",
    )
    try:
        df = lakehouse_session.read_gbq_table(lakehouse_table_id)
        parities = lakehouse_session.read_gbq_table(regular_table_id)
        labels = lakehouse_session.read_pandas(
            pd.DataFrame({"parity": ["even", "odd"], "label": ["E", "O"]})
        )
        totals = (
            df[df["id"] > 2]
            .groupby("flag", as_index=False)["id"]
            .sum()
            .rename(columns={"id": "total"})
        )
        result = totals.merge(parities, on="flag").merge(labels, on="parity")

        result.to_gbq(destination, if_exists="replace", index=False)

        rows = lakehouse_bqclient.query_and_wait(
            f"SELECT * FROM {_quoted(destination)} ORDER BY flag"
        )
        assert [tuple(row.values()) for row in rows] == [
            (False, 15, "odd", "O"),
            (True, 18, "even", "E"),
        ]
    finally:
        _drop_lakehouse_table(lakehouse_bqclient, destination)


@pytest.mark.parametrize(
    ("if_exists", "expected_ids"),
    [
        ("append", [1, 2, 3, 11, 12, 13]),
        ("replace", [11, 12, 13]),
    ],
)
def test_to_gbq_lakehouse_into_source_table(
    lakehouse_session,
    lakehouse_bqclient,
    lakehouse_mutable_table_id,
    if_exists,
    expected_ids,
):
    df = lakehouse_session.read_gbq_table(lakehouse_mutable_table_id)
    df["id"] = df["id"] + 10

    df.to_gbq(lakehouse_mutable_table_id, if_exists=if_exists, index=False)

    assert _read_ids(lakehouse_bqclient, lakehouse_mutable_table_id) == expected_ids


def test_to_gbq_lakehouse_replace_w_timedelta(lakehouse_session, lakehouse_bqclient):
    table_id = _create_lakehouse_table(
        lakehouse_bqclient,
        "SELECT 1 AS id, 1000000 AS elapsed",
        column_definitions=(
            "id INT64, elapsed INT64 OPTIONS (description = "
            f"'{bigframes.dtypes.TIMEDELTA_DESCRIPTION_TAG}')"
        ),
    )
    try:
        df = lakehouse_session.read_pandas(
            pd.DataFrame(
                {
                    "id": [2, 3],
                    "elapsed": [pd.Timedelta(2, "s"), pd.Timedelta(3, "s")],
                }
            )
        )

        df.to_gbq(table_id, if_exists="replace", index=False)

        table = lakehouse_bqclient.get_table(table_id)
        assert table.schema[1].description == bigframes.dtypes.TIMEDELTA_DESCRIPTION_TAG
        result = lakehouse_session.read_gbq_table(table_id).to_pandas()
        result = result.sort_values("id")
        assert result["elapsed"].dtype == bigframes.dtypes.TIMEDELTA_DTYPE
        assert result["elapsed"].tolist() == [
            pd.Timedelta(2, "s"),
            pd.Timedelta(3, "s"),
        ]
    finally:
        _drop_lakehouse_table(lakehouse_bqclient, table_id)


def test_to_gbq_lakehouse_w_sql_cell_result(
    lakehouse_session,
    lakehouse_bqclient,
    lakehouse_table_id,
    lakehouse_mutable_table_id,
):
    df = lakehouse_session.read_gbq_table(lakehouse_table_id)
    result = lakehouse_session._read_gbq_colab(
        "SELECT id FROM {df} WHERE id > 6", pyformat_args={"df": df}
    )

    # SQL cell results have no index, so there's no index column to leave out.
    result.to_gbq(lakehouse_mutable_table_id, if_exists="append")

    assert _read_ids(lakehouse_bqclient, lakehouse_mutable_table_id) == [1, 2, 3, 7, 8]


def test_read_gbq_colab_lakehouse_create_table_as_select(
    lakehouse_session, lakehouse_bqclient, lakehouse_table_id
):
    df = lakehouse_session.read_gbq_table(lakehouse_table_id)
    table_id = f"{LAKEHOUSE_PROJECT}.{LAKEHOUSE_DATASET}.{prefixer.create_prefix()}"
    try:
        lakehouse_session._read_gbq_colab(
            f"CREATE TABLE {_quoted(table_id)} AS SELECT id FROM {{df}} WHERE flag",
            pyformat_args={"df": df},
        )

        assert _read_ids(lakehouse_bqclient, table_id) == [2, 4, 6, 8]
    finally:
        _drop_lakehouse_table(lakehouse_bqclient, table_id)


def test_read_gbq_colab_lakehouse_insert(
    lakehouse_session,
    lakehouse_bqclient,
    lakehouse_table_id,
    lakehouse_mutable_table_id,
):
    df = lakehouse_session.read_gbq_table(lakehouse_table_id)

    lakehouse_session._read_gbq_colab(
        f"INSERT INTO {_quoted(lakehouse_mutable_table_id)} (id) "
        f"SELECT id FROM {{df}} WHERE id > 6",
        pyformat_args={"df": df},
    )

    assert _read_ids(lakehouse_bqclient, lakehouse_mutable_table_id) == [1, 2, 3, 7, 8]


def test_to_gbq_lakehouse_w_wrong_location(
    lakehouse_bqclient, lakehouse_mutable_table_id
):
    session = bigframes.Session(
        context=bigframes.BigQueryOptions(location="europe-west1")
    )
    try:
        df = session.read_pandas(pd.DataFrame({"id": [4]}))
        with pytest.raises(google.api_core.exceptions.NotFound, match="europe-west1"):
            df.to_gbq(lakehouse_mutable_table_id, if_exists="append", index=False)
    finally:
        session.close()

    assert _read_ids(lakehouse_bqclient, lakehouse_mutable_table_id) == [1, 2, 3]


def test_to_gbq_lakehouse_table_already_exists(lakehouse_session, lakehouse_table_id):
    df = lakehouse_session.read_pandas(pd.DataFrame({"id": [4]}))

    with pytest.raises(ValueError, match="Table already exists"):
        df.to_gbq(lakehouse_table_id)
