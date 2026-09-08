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


def test_boosted_tree_model(random_model_id: str) -> None:
    your_model_id = random_model_id
    # [START bigquery_dataframes_bqml_boosted_tree_prepare]
    import bigframes.pandas as bpd

    # Set partial ordering mode for BigQuery DataFrames.
    # For more information, see the BigQuery DataFrames performance documentation:
    # https://cloud.google.com/bigquery/docs/dataframes-performance#partial-ordering-mode
    bpd.options.bigquery.ordering_mode = "partial"

    input_data = bpd.read_gbq(
        "bigquery-public-data.ml_datasets.census_adult_income",
        columns=(
            "age",
            "workclass",
            "marital_status",
            "education_num",
            "occupation",
            "hours_per_week",
            "income_bracket",
            "functional_weight",
        ),
    )
    input_data["dataframe"] = input_data["functional_weight"].case_when(
        [
            (((input_data["functional_weight"] % 10) == 8), "evaluation"),
            (((input_data["functional_weight"] % 10) == 9), "prediction"),
            (True, "training"),
        ]
    )
    del input_data["functional_weight"]
    # [END bigquery_dataframes_bqml_boosted_tree_prepare]
    # [START bigquery_dataframes_bqml_boosted_tree_create]
    import bigframes.pandas as bpd
    from bigframes.bigquery import ml

    # Set partial ordering mode for BigQuery DataFrames.
    # For more information, see the BigQuery DataFrames performance documentation:
    # https://cloud.google.com/bigquery/docs/dataframes-performance#partial-ordering-mode
    bpd.options.bigquery.ordering_mode = "partial"

    # input_data is defined in an earlier step.
    training_data = input_data[input_data["dataframe"] == "training"].drop(
        columns=["dataframe"]
    )

    # For more information, see the BigQuery DataFrames API reference documentation:
    # https://dataframes.bigquery.dev/reference/api/bigframes.bigquery.ml.create_model.html#bigframes.bigquery.ml.create_model
    ml.create_model(
        your_model_id,  # For example: "your-project.bqml_tutorial.tree_model"
        options={
            "model_type": "BOOSTED_TREE_CLASSIFIER",
            "booster_type": "GBTREE",
            "num_parallel_tree": 1,
            "max_iterations": 1,  # For a more accurate model, try 50 iterations.
            "tree_method": "HIST",
            "early_stop": False,
            "subsample": 0.85,
            "input_label_cols": ["income_bracket"],
        },
        training_data=training_data,
        replace=True,
    )
    # [END bigquery_dataframes_bqml_boosted_tree_create]
    # [START bigquery_dataframes_bqml_boosted_tree_evaluate]
    import bigframes.pandas as bpd
    from bigframes.bigquery import ml

    # Set partial ordering mode for BigQuery DataFrames.
    # For more information, see the BigQuery DataFrames performance documentation:
    # https://cloud.google.com/bigquery/docs/dataframes-performance#partial-ordering-mode
    bpd.options.bigquery.ordering_mode = "partial"

    # input_data is defined in an earlier step.
    evaluation_data = input_data[input_data["dataframe"] == "evaluation"]

    # For more information, see the BigQuery DataFrames API reference documentation:
    # https://dataframes.bigquery.dev/reference/api/bigframes.bigquery.ml.evaluate.html#bigframes.bigquery.ml.evaluate
    ml.evaluate(
        your_model_id,  # For example: "your-project.bqml_tutorial.tree_model"
        input_=evaluation_data,
    )
    # Output:
    #    precision    recall  accuracy  f1_score  log_loss   roc_auc
    # 0   0.671924  0.578804  0.839429  0.621897  0.344054  0.887335
    # [END bigquery_dataframes_bqml_boosted_tree_evaluate]
    # [START bigquery_dataframes_bqml_boosted_tree_predict]
    import bigframes.pandas as bpd
    from bigframes.bigquery import ml

    # Set partial ordering mode for BigQuery DataFrames.
    # For more information, see the BigQuery DataFrames performance documentation:
    # https://cloud.google.com/bigquery/docs/dataframes-performance#partial-ordering-mode
    bpd.options.bigquery.ordering_mode = "partial"

    # input_data is defined in an earlier step.
    prediction_data = input_data[input_data["dataframe"] == "prediction"]

    # For more information, see the BigQuery DataFrames API reference documentation:
    # https://dataframes.bigquery.dev/reference/api/bigframes.bigquery.ml.predict.html#bigframes.bigquery.ml.predict
    ml.predict(
        your_model_id,  # For example: "your-project.bqml_tutorial.tree_model"
        input_=prediction_data,
    )
    # Output:
    # predicted_income_bracket   predicted_income_bracket_probs.label  predicted_income_bracket_probs.prob
    #                   <=50K                                   >50K                   0.05183430016040802
    #                                                           <50K                   0.94816571474075317
    #                   <=50K                                   >50K                   0.00365859130397439
    #                                                           <50K                   0.99634140729904175
    #                   <=50K                                   >50K                   0.037775970995426178
    #                                                           <50K                   0.96222406625747681
    # [END bigquery_dataframes_bqml_boosted_tree_predict]
    assert input_data is not None
    assert training_data is not None
    assert evaluation_data is not None
    assert prediction_data is not None

