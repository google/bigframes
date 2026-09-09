# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (t
# you may not use this file except in compliance wi
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in
# distributed under the License is distributed on a
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, eit
# See the License for the specific language governi
# limitations under the License.


def test_imported_sklearn_onnx_model() -> None:
    # Determine project id, in this case prefer the one set in the environment
    # variable GOOGLE_CLOUD_PROJECT (if any)
    import os

    PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "bigframes-dev")
    your_model_id = "your_model_id"

    # [START bigquery_dataframes_imported_sklearn_onnx_tutorial_import_onnx_models]
    import bigframes
    import bigframes.pandas as bpd
    from bigframes.bigquery import ml

    bigframes.options.bigquery.project = PROJECT_ID
    # You can change the location to one of the valid locations: https://cloud.google.com/bigquery/docs/locations#supported_locations
    bigframes.options.bigquery.location = "US"

    # Set partial ordering mode for BigQuery DataFrames.
    # For more information, see the BigQuery DataFrames performance documentation:
    # https://cloud.google.com/bigquery/docs/dataframes-performance#partial-ordering-mode
    bpd.options.bigquery.ordering_mode = "partial"

    # Use ml.create_model to create and import the model in BigQuery.
    # The options parameter specifies the model type and the Cloud Storage path.
    # For more information, see the BigQuery DataFrames API reference documentation:
    # https://dataframes.bigquery.dev/reference/api/bigframes.bigquery.ml.create_model.html#bigframes.bigquery.ml.create_model
    ml.create_model(
        your_model_id,  # For example: "bqml_tutorial.imported_onnx_model"
        options={
            "model_type": "ONNX",
            "model_path": "gs://cloud-samples-data/bigquery/ml/onnx/pipeline_rf.onnx",
        },
        replace=True,
    )
    # [END bigquery_dataframes_imported_sklearn_onnx_tutorial_import_onnx_models]

    # [START bigquery_dataframes_imported_sklearn_onnx_tutorial_make_predictions]
    import bigframes.pandas as bpd
    from bigframes.bigquery import ml

    # Set partial ordering mode for BigQuery DataFrames.
    # For more information, see the BigQuery DataFrames performance documentation:
    # https://cloud.google.com/bigquery/docs/dataframes-performance#partial-ordering-mode
    bpd.options.bigquery.ordering_mode = "partial"

    df = bpd.read_gbq("bigquery-public-data.ml_datasets.iris")

    # Use the ml.predict method to predict results using your model.
    # For more information, see the BigQuery DataFrames API reference documentation:
    # https://dataframes.bigquery.dev/reference/api/bigframes.bigquery.ml.predict.html#bigframes.bigquery.ml.predict
    predictions = ml.predict(
        your_model_id,  # For example: "bqml_tutorial.imported_onnx_model"
        input_=df,
    )
    predictions.peek(5)
    # [END bigquery_dataframes_imported_sklearn_onnx_tutorial_make_predictions]
