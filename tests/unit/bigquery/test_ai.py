# Copyright 2025 Google LLC
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

import pandas as pd
import pytest

import bigframes.bigquery as bbq
import bigframes.dataframe
import bigframes.series
import bigframes.session


@pytest.fixture
def mock_session():
    return mock.create_autospec(spec=bigframes.session.Session)


@pytest.fixture
def mock_dataframe(mock_session):
    df = mock.create_autospec(spec=bigframes.dataframe.DataFrame)
    df._session = mock_session
    df.sql = "SELECT * FROM my_table"
    df._to_sql_query.return_value = ("SELECT * FROM my_table", None, None)
    return df


@pytest.fixture
def mock_embedding_series(mock_session):
    series = mock.create_autospec(spec=bigframes.series.Series)
    series._session = mock_session
    # Mock to_frame to return a mock dataframe
    df = mock.create_autospec(spec=bigframes.dataframe.DataFrame)
    df._session = mock_session
    df.sql = "SELECT my_col AS content FROM my_table"
    df._to_sql_query.return_value = (
        "SELECT my_col AS content FROM my_table",
        None,
        None,
    )
    series.copy.return_value = series
    series.to_frame.return_value = df
    return series


@pytest.fixture
def mock_text_series(mock_session):
    series = mock.create_autospec(spec=bigframes.series.Series)
    series._session = mock_session
    # Mock to_frame to return a mock dataframe
    df = mock.create_autospec(spec=bigframes.dataframe.DataFrame)
    df._session = mock_session
    df.sql = "SELECT my_col AS prompt FROM my_table"
    df._to_sql_query.return_value = (
        "SELECT my_col AS prompt FROM my_table",
        None,
        None,
    )
    series.copy.return_value = series
    series.to_frame.return_value = df
    return series


def test_generate_embedding_with_dataframe(mock_dataframe, mock_session):
    model_name = "project.dataset.model"

    bbq.ai.generate_embedding(
        model_name,
        mock_dataframe,
        output_dimensionality=256,
    )

    mock_session.read_gbq_query.assert_called_once()
    query = mock_session.read_gbq_query.call_args[0][0]

    # Normalize whitespace for comparison
    query = " ".join(query.split())

    expected_part_1 = "SELECT * FROM AI.GENERATE_EMBEDDING("
    expected_part_2 = f"MODEL `{model_name}`,"
    expected_part_3 = "(SELECT * FROM my_table),"
    expected_part_4 = "STRUCT(256 AS `OUTPUT_DIMENSIONALITY`)"

    assert expected_part_1 in query
    assert expected_part_2 in query
    assert expected_part_3 in query
    assert expected_part_4 in query


def test_generate_embedding_with_series(mock_embedding_series, mock_session):
    model_name = "project.dataset.model"

    bbq.ai.generate_embedding(
        model_name,
        mock_embedding_series,
        start_second=0.0,
        end_second=10.0,
        interval_seconds=5.0,
    )

    mock_session.read_gbq_query.assert_called_once()
    query = mock_session.read_gbq_query.call_args[0][0]
    query = " ".join(query.split())

    assert f"MODEL `{model_name}`" in query
    assert "(SELECT my_col AS content FROM my_table)" in query
    assert (
        "STRUCT(0.0 AS `START_SECOND`, 10.0 AS `END_SECOND`, 5.0 AS `INTERVAL_SECONDS`)"
        in query
    )


def test_generate_embedding_defaults(mock_dataframe, mock_session):
    model_name = "project.dataset.model"

    bbq.ai.generate_embedding(
        model_name,
        mock_dataframe,
    )

    mock_session.read_gbq_query.assert_called_once()
    query = mock_session.read_gbq_query.call_args[0][0]
    query = " ".join(query.split())

    assert f"MODEL `{model_name}`" in query
    assert "STRUCT()" in query


@mock.patch("bigframes.pandas.read_pandas")
def test_generate_embedding_with_pandas_dataframe(
    read_pandas_mock, mock_dataframe, mock_session
):
    # This tests that pandas input path works and calls read_pandas
    model_name = "project.dataset.model"

    # Mock return value of read_pandas to be a BigFrames DataFrame
    read_pandas_mock.return_value = mock_dataframe

    pandas_df = pd.DataFrame({"content": ["test"]})

    bbq.ai.generate_embedding(
        model_name,
        pandas_df,
    )

    read_pandas_mock.assert_called_once()
    # Check that read_pandas was called with something (the pandas df)
    assert read_pandas_mock.call_args[0][0] is pandas_df

    mock_session.read_gbq_query.assert_called_once()


def test_generate_text_with_dataframe(mock_dataframe, mock_session):
    model_name = "project.dataset.model"

    bbq.ai.generate_text(
        model_name,
        mock_dataframe,
        max_output_tokens=256,
    )

    mock_session.read_gbq_query.assert_called_once()
    query = mock_session.read_gbq_query.call_args[0][0]

    # Normalize whitespace for comparison
    query = " ".join(query.split())

    expected_part_1 = "SELECT * FROM AI.GENERATE_TEXT("
    expected_part_2 = f"MODEL `{model_name}`,"
    expected_part_3 = "(SELECT * FROM my_table),"
    expected_part_4 = "STRUCT(256 AS `MAX_OUTPUT_TOKENS`)"

    assert expected_part_1 in query
    assert expected_part_2 in query
    assert expected_part_3 in query
    assert expected_part_4 in query


def test_generate_text_with_series(mock_text_series, mock_session):
    model_name = "project.dataset.model"

    bbq.ai.generate_text(
        model_name,
        mock_text_series,
    )

    mock_session.read_gbq_query.assert_called_once()
    query = mock_session.read_gbq_query.call_args[0][0]
    query = " ".join(query.split())

    assert f"MODEL `{model_name}`" in query
    assert "(SELECT my_col AS prompt FROM my_table)" in query


def test_generate_text_defaults(mock_dataframe, mock_session):
    model_name = "project.dataset.model"

    bbq.ai.generate_text(
        model_name,
        mock_dataframe,
    )

    mock_session.read_gbq_query.assert_called_once()
    query = mock_session.read_gbq_query.call_args[0][0]
    query = " ".join(query.split())

    assert f"MODEL `{model_name}`" in query
    assert "STRUCT()" in query


def test_generate_table_with_dataframe(mock_dataframe, mock_session):
    model_name = "project.dataset.model"

    bbq.ai.generate_table(
        model_name,
        mock_dataframe,
        output_schema="col1 STRING, col2 INT64",
    )

    mock_session.read_gbq_query.assert_called_once()
    query = mock_session.read_gbq_query.call_args[0][0]

    # Normalize whitespace for comparison
    query = " ".join(query.split())

    expected_part_1 = "SELECT * FROM AI.GENERATE_TABLE("
    expected_part_2 = f"MODEL `{model_name}`,"
    expected_part_3 = "(SELECT * FROM my_table),"
    expected_part_4 = "STRUCT('col1 STRING, col2 INT64' AS `output_schema`)"

    assert expected_part_1 in query
    assert expected_part_2 in query
    assert expected_part_3 in query
    assert expected_part_4 in query


def test_generate_table_with_options(mock_dataframe, mock_session):
    model_name = "project.dataset.model"

    bbq.ai.generate_table(
        model_name,
        mock_dataframe,
        output_schema="col1 STRING",
        temperature=0.5,
        max_output_tokens=100,
    )

    mock_session.read_gbq_query.assert_called_once()
    query = mock_session.read_gbq_query.call_args[0][0]
    query = " ".join(query.split())

    assert f"MODEL `{model_name}`" in query
    assert "(SELECT * FROM my_table)" in query
    assert (
        "STRUCT('col1 STRING' AS `output_schema`, 0.5 AS `temperature`, 100 AS `max_output_tokens`)"
        in query
    )


def test_generate_table_with_mapping_schema(mock_dataframe, mock_session):
    model_name = "project.dataset.model"

    bbq.ai.generate_table(
        model_name,
        mock_dataframe,
        output_schema={"col1": "STRING", "col2": "INT64"},
    )

    mock_session.read_gbq_query.assert_called_once()
    query = mock_session.read_gbq_query.call_args[0][0]

    # Normalize whitespace for comparison
    query = " ".join(query.split())

    expected_part_1 = "SELECT * FROM AI.GENERATE_TABLE("
    expected_part_2 = f"MODEL `{model_name}`,"
    expected_part_3 = "(SELECT * FROM my_table),"
    expected_part_4 = "STRUCT('col1 STRING, col2 INT64' AS `output_schema`)"

    assert expected_part_1 in query
    assert expected_part_2 in query
    assert expected_part_3 in query
    assert expected_part_4 in query


@mock.patch("bigframes.pandas.read_pandas")
def test_generate_text_with_pandas_dataframe(
    read_pandas_mock, mock_dataframe, mock_session
):
    # This tests that pandas input path works and calls read_pandas
    model_name = "project.dataset.model"

    # Mock return value of read_pandas to be a BigFrames DataFrame
    read_pandas_mock.return_value = mock_dataframe

    pandas_df = pd.DataFrame({"content": ["test"]})

    bbq.ai.generate_text(
        model_name,
        pandas_df,
    )

    read_pandas_mock.assert_called_once()
    # Check that read_pandas was called with something (the pandas df)
    assert read_pandas_mock.call_args[0][0] is pandas_df

    mock_session.read_gbq_query.assert_called_once()


def test_evaluate_timesfm(mock_dataframe, mock_session):
    mock_dataframe.columns = ["time", "value", "id"]

    bbq.ai.evaluate(
        df1=mock_dataframe,
        df2=mock_dataframe,
        data_col="value",
        timestamp_col="time",
        model="TimesFM 2.5",
        id_cols=["id"],
        horizon=100,
        context_window=64,
    )

    mock_session.read_gbq_query.assert_called_once()
    query = " ".join(mock_session.read_gbq_query.call_args[0][0].split())

    assert (
        "SELECT * FROM AI.EVALUATE((SELECT * FROM my_table),(SELECT * FROM my_table),"
        in query
    )
    assert "data_col => 'value'" in query
    assert "timestamp_col => 'time'" in query
    assert "model => 'TimesFM 2.5'" in query
    assert "id_cols => ['id']" in query
    assert "horizon => 100" in query
    assert "context_window => 64" in query


def test_evaluate_tabfm(mock_dataframe, mock_session):
    mock_dataframe.columns = ["feat", "body_mass_g"]

    bbq.ai.evaluate(
        df1=mock_dataframe,
        df2=mock_dataframe,
        label_col="body_mass_g",
    )

    mock_session.read_gbq_query.assert_called_once()
    query = " ".join(mock_session.read_gbq_query.call_args[0][0].split())

    assert (
        query
        == "SELECT * FROM AI.EVALUATE((SELECT * FROM my_table),(SELECT * FROM my_table), label_col => 'body_mass_g')"
    )


def test_evaluate_timesfm_defaults(mock_dataframe, mock_session):
    mock_dataframe.columns = ["time", "value"]

    bbq.ai.evaluate(
        mock_dataframe,
        mock_dataframe,
        data_col="value",
        timestamp_col="time",
    )

    mock_session.read_gbq_query.assert_called_once()
    query = " ".join(mock_session.read_gbq_query.call_args[0][0].split())

    assert (
        query
        == "SELECT * FROM AI.EVALUATE((SELECT * FROM my_table),(SELECT * FROM my_table), data_col => 'value', timestamp_col => 'time', model => 'TimesFM 2.5', horizon => 1024)"
    )


@mock.patch("bigframes.pandas.get_global_session")
def test_evaluate_with_pandas_dataframes(
    mock_get_global_session, mock_dataframe, mock_session
):
    mock_get_global_session.return_value = mock_session
    mock_dataframe.columns = ["feat", "body_mass_g"]
    mock_session.read_pandas.return_value = mock_dataframe

    train_pdf = pd.DataFrame({"feat": [1], "body_mass_g": [3500]})
    pred_pdf = pd.DataFrame({"feat": [2], "body_mass_g": [4000]})

    bbq.ai.evaluate(
        train_pdf,
        pred_pdf,
        label_col="body_mass_g",
    )

    mock_get_global_session.assert_called_once()
    assert mock_session.read_pandas.call_count == 2
    mock_session.read_gbq_query.assert_called_once()


@pytest.mark.parametrize(
    ("kwargs", "expected_error"),
    [
        (
            {"data_col": "value", "timestamp_col": "time", "label_col": "label"},
            "Cannot specify both",
        ),
        (
            {},
            "Must specify either",
        ),
        (
            {"data_col": "value"},
            "Must specify either",
        ),
        (
            {"timestamp_col": "time"},
            "Must specify either",
        ),
        (
            {"label_col": "missing_col"},
            "Column `missing_col` not found",
        ),
        (
            {"data_col": "value", "timestamp_col": "time"},
            "Column `value` not found",
        ),
    ],
)
def test_evaluate_validation_errors(
    mock_dataframe, mock_session, kwargs, expected_error
):
    mock_dataframe.columns = ["time", "value", "label"]
    other_df = mock.create_autospec(spec=bigframes.dataframe.DataFrame)
    other_df._session = mock_session
    other_df.columns = ["time", "label"]

    with pytest.raises(ValueError, match=expected_error):
        bbq.ai.evaluate(mock_dataframe, other_df, **kwargs)
