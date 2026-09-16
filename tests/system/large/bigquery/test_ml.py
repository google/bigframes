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

import pytest

import bigframes.bigquery.ml as ml
import bigframes.pandas as bpd


@pytest.fixture(scope="session")
def embedding_model(bq_connection, dataset_id):
    model_name = f"{dataset_id}.embedding_model"
    return ml.create_model(
        model_name=model_name,
        options={"endpoint": "gemini-embedding-001"},
        connection_name=bq_connection,
        replace=True,
    )


def test_generate_embedding(embedding_model):
    df = bpd.DataFrame(
        {
            "content": [
                "What is BigQuery?",
                "What is BQML?",
            ]
        }
    )

    result = ml.generate_embedding(embedding_model, df)
    assert len(result) == 2
    assert "ml_generate_embedding_result" in result.columns
    assert "ml_generate_embedding_status" in result.columns


def test_generate_embedding_with_options(embedding_model):
    df = bpd.DataFrame(
        {
            "content": [
                "What is BigQuery?",
                "What is BQML?",
            ]
        }
    )

    result = ml.generate_embedding(
        embedding_model, df, task_type="RETRIEVAL_DOCUMENT", output_dimensionality=256
    )
    assert len(result) == 2
    assert "ml_generate_embedding_result" in result.columns
    assert "ml_generate_embedding_status" in result.columns
    embedding = result["ml_generate_embedding_result"].to_pandas()
    assert len(embedding[0]) == 256


def test_get_insights(dataset_id):
    df = bpd.DataFrame(
        {
            "dim1": ["a", "a", "b", "b", "a", "a", "b", "b"],
            "dim2": ["x", "y", "x", "y", "x", "y", "x", "y"],
            "metric": [10, 20, 30, 40, 12, 25, 35, 45],
            "is_test": [False, False, False, False, True, True, True, True],
        }
    )
    model_name = f"{dataset_id}.contribution_analysis_model"

    ml.create_model(
        model_name=model_name,
        options={
            "model_type": "CONTRIBUTION_ANALYSIS",
            "contribution_metric": "SUM(metric)",
            "is_test_col": "is_test",
        },
        training_data=df,
        replace=True,
    )

    result = ml.get_insights(model_name)
    assert len(result) > 0
    assert "contributors" in result.columns


def test_create_model_linear_regression(dataset_id):
    df = bpd.DataFrame({"x": [1, 2, 3], "y": [2, 4, 6]})
    model_name = f"{dataset_id}.linear_regression_model"

    result = ml.create_model(
        model_name=model_name,
        options={"model_type": "LINEAR_REG", "input_label_cols": ["y"]},
        training_data=df,
        replace=True,
    )

    assert result["modelType"] == "LINEAR_REGRESSION"


def test_create_model_with_transform(dataset_id):
    df = bpd.DataFrame({"x": [1, 2, 3], "y": [2, 4, 6]})
    model_name = f"{dataset_id}.transform_model"

    result = ml.create_model(
        model_name=model_name,
        options={"model_type": "LINEAR_REG", "input_label_cols": ["y"]},
        training_data=df,
        transform=["x * 2 AS x_doubled", "y"],
        replace=True,
    )

    assert result["modelType"] == "LINEAR_REGRESSION"


@pytest.fixture(scope="module")
def matrix_factorization_model(ratings_df_default_index, dataset_id):
    model_name = f"{dataset_id}.mf_model"
    return ml.create_model(
        model_name=model_name,
        options={
            "model_type": "MATRIX_FACTORIZATION",
            "feedback_type": "explicit",
            "user_col": "user_id",
            "item_col": "item_id",
            "rating_col": "rating",
            "l2_reg": 9.83,
            "num_factors": 6,
        },
        training_data=ratings_df_default_index,
        replace=True,
    )


def test_recommend_no_input(matrix_factorization_model):
    # Without input data, ML.RECOMMEND returns a rating for every user-item
    # combination seen during training.
    result = ml.recommend(matrix_factorization_model)

    assert len(result) > 0
    assert "user_id" in result.columns
    assert "item_id" in result.columns
    assert "predicted_rating" in result.columns


def test_recommend_with_input(matrix_factorization_model):
    users = bpd.DataFrame({"user_id": ["1", "2"]})

    result = ml.recommend(matrix_factorization_model, users)

    assert len(result) > 0
    assert "predicted_rating" in result.columns
    assert set(result["user_id"].to_pandas()) == {"1", "2"}


@pytest.fixture(scope="module")
def arima_plus_model(time_series_df_default_index, dataset_id):
    model_name = f"{dataset_id}.arima_plus_model"
    return ml.create_model(
        model_name=model_name,
        options={
            "model_type": "ARIMA_PLUS",
            "time_series_timestamp_col": "parsed_date",
            "time_series_data_col": "total_visits",
            "horizon": 10,
            # ML.EXPLAIN_FORECAST requires this on ARIMA_PLUS models. It
            # defaults to True, but it is set explicitly so that the
            # requirement is visible.
            "decompose_time_series": True,
        },
        training_data=time_series_df_default_index[["parsed_date", "total_visits"]],
        replace=True,
    )


def test_forecast_no_input(arima_plus_model):
    # ARIMA_PLUS models forecast when the model is created, so ML.FORECAST
    # takes no input data.
    result = ml.forecast(arima_plus_model)

    assert len(result) > 0
    assert "forecast_timestamp" in result.columns
    assert "forecast_value" in result.columns
    assert "confidence_level" in result.columns


def test_forecast_with_options(arima_plus_model):
    result = ml.forecast(arima_plus_model, horizon=4, confidence_level=0.8)

    assert len(result) == 4
    assert "forecast_value" in result.columns
    assert set(result["confidence_level"].to_pandas()) == {0.8}


@pytest.fixture(scope="module")
def arima_plus_xreg_model(time_series_df_default_index, dataset_id):
    model_name = f"{dataset_id}.arima_plus_xreg_model"
    # Filter to a single time series so that timestamps are unique, and derive
    # an external covariate, since the shared table has no feature column.
    df = time_series_df_default_index[time_series_df_default_index["id"] == "1"][
        ["parsed_date", "total_visits"]
    ]
    df["day_of_week"] = df["parsed_date"].dt.dayofweek

    return ml.create_model(
        model_name=model_name,
        options={
            "model_type": "ARIMA_PLUS_XREG",
            "time_series_timestamp_col": "parsed_date",
            "time_series_data_col": "total_visits",
            "horizon": 10,
        },
        training_data=df,
        replace=True,
    )


@pytest.fixture(scope="module")
def future_features():
    # An ARIMA_PLUS_XREG model needs the future values of its covariates, which
    # the model cannot know. The training data ends on 2017-08-01, so supply the
    # three days that follow.
    return bpd.DataFrame(
        {
            "parsed_date": [
                datetime.datetime(2017, 8, 2, tzinfo=datetime.timezone.utc),
                datetime.datetime(2017, 8, 3, tzinfo=datetime.timezone.utc),
                datetime.datetime(2017, 8, 4, tzinfo=datetime.timezone.utc),
            ],
            "day_of_week": [2, 3, 4],
        }
    )


def test_forecast_xreg_with_input(arima_plus_xreg_model, future_features):
    result = ml.forecast(
        arima_plus_xreg_model, future_features, horizon=3, confidence_level=0.9
    )

    assert len(result) == 3
    assert "forecast_timestamp" in result.columns
    assert "forecast_value" in result.columns
    assert set(result["confidence_level"].to_pandas()) == {0.9}


def test_explain_forecast_no_input(arima_plus_model):
    # ARIMA_PLUS models forecast when the model is created, so
    # ML.EXPLAIN_FORECAST takes no input data.
    result = ml.explain_forecast(arima_plus_model)

    assert len(result) > 0
    assert "time_series_timestamp" in result.columns
    assert "time_series_data" in result.columns
    assert set(result["time_series_type"].to_pandas()) == {"history", "forecast"}
    assert result["trend"].notnull().all()
    assert result["time_series_adjusted_data"].notnull().all()


def test_explain_forecast_with_options(arima_plus_model):
    result = ml.explain_forecast(arima_plus_model, horizon=4, confidence_level=0.8)

    # Unlike ML.FORECAST, the output also contains the history rows that the
    # forecast is explained against, so only the forecast rows follow horizon.
    forecast = result[result["time_series_type"] == "forecast"]
    history = result[result["time_series_type"] == "history"]

    assert len(forecast) == 4
    assert forecast["prediction_interval_lower_bound"].notnull().all()
    assert forecast["prediction_interval_upper_bound"].notnull().all()
    assert set(forecast["confidence_level"].to_pandas()) == {0.8}
    assert history["confidence_level"].isnull().all()


def test_explain_forecast_xreg_with_input(arima_plus_xreg_model, future_features):
    result = ml.explain_forecast(
        arima_plus_xreg_model, future_features, horizon=3, confidence_level=0.9
    )
    forecast = result[result["time_series_type"] == "forecast"]

    assert len(forecast) == 3
    assert "time_series_adjusted_data" in result.columns
    assert "trend" in result.columns


@pytest.mark.parametrize(
    "model_fixture",
    [
        "arima_plus_model",
        "arima_plus_xreg_model",
    ],
)
def test_arima_evaluate(model_fixture, request):
    model = request.getfixturevalue(model_fixture)

    result = ml.arima_evaluate(model)

    expected_columns = {
        "non_seasonal_p",
        "non_seasonal_d",
        "non_seasonal_q",
        "has_drift",
        "log_likelihood",
        "AIC",
        "variance",
    }
    assert len(result) > 0
    assert expected_columns <= set(result.columns)
    assert result["AIC"].notnull().all()


@pytest.mark.parametrize(
    "model_fixture",
    [
        "arima_plus_model",
        "arima_plus_xreg_model",
    ],
)
@pytest.mark.parametrize(
    "show_all_candidate_models",
    [
        True,
        False,
    ],
)
def test_arima_evaluate_with_options(model_fixture, show_all_candidate_models, request):
    model = request.getfixturevalue(model_fixture)

    result = ml.arima_evaluate(
        model, show_all_candidate_models=show_all_candidate_models
    )

    if show_all_candidate_models:
        assert len(result) > 1
    else:
        assert len(result) == 1
