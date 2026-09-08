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


def test_linear_regression(random_model_id: str) -> None:
    your_model_id = random_model_id
    # [START bigquery_dataframes_bqml_linear_regression]
    import bigframes.pandas as bpd
    from bigframes.bigquery import ml

    # Set partial ordering mode for BigQuery DataFrames.
    # For more information, see the BigQuery DataFrames performance documentation:
    # https://cloud.google.com/bigquery/docs/dataframes-performance#partial-ordering-mode
    bpd.options.bigquery.ordering_mode = "partial"

    # Load data from BigQuery
    bq_df = bpd.read_gbq("bigquery-public-data.ml_datasets.penguins")

    # Drop rows with nulls to get training data
    training_data = bq_df.dropna(subset=["body_mass_g"])

    # A Linear Regression model predicts a continuous numerical value.
    #
    # Use ml.create_model to create and train the model in BigQuery.
    # The options parameter specifies the model type and the label column.
    # For more information, see the BigQuery DataFrames API reference documentation:
    # https://dataframes.bigquery.dev/reference/api/bigframes.bigquery.ml.create_model.html#bigframes.bigquery.ml.create_model
    ml.create_model(
        your_model_id,  # For example: "bqml_tutorial.penguins_model",
        options={
            "model_type": "LINEAR_REG",
            "input_label_cols": ["body_mass_g"],
        },
        training_data=training_data,
        replace=True,
    )
    # [END bigquery_dataframes_bqml_linear_regression]
    # [START bigquery_dataframes_bqml_linear_evaluate]
    # Use the ml.evaluate method to evaluate the model with test data.
    # For more information, see the BigQuery DataFrames API reference documentation:
    # https://dataframes.bigquery.dev/reference/api/bigframes.bigquery.ml.evaluate.html#bigframes.bigquery.ml.evaluate
    ml.evaluate(
        your_model_id,  # For example: "bqml_tutorial.penguins_model",
        input_=training_data,
    )
    # Expected output results:
    # index  mean_absolute_error  mean_squared_error  mean_squared_log_error  median_absolute_error  r2_score  explained_variance
    #   0        227.012237         81838.159892            0.00507                173.080816        0.872377    0.872377
    #   1 rows x 6 columns
    # [END bigquery_dataframes_bqml_linear_evaluate]
    # [START bigquery_dataframes_bqml_linear_predict]
    # Load data from BigQuery
    bq_df = bpd.read_gbq("bigquery-public-data.ml_datasets.penguins")

    # Use 'contains' function to filter by island containing the string
    # "Biscoe".
    biscoe_data = bq_df[bq_df["island"].str.contains("Biscoe")]

    # Use the ml.predict method to predict results using your model.
    # For more information, see the BigQuery DataFrames API reference documentation:
    # https://dataframes.bigquery.dev/reference/api/bigframes.bigquery.ml.predict.html#bigframes.bigquery.ml.predict
    ml.predict(
        your_model_id,  # For example: "bqml_tutorial.penguins_model",
        input_=biscoe_data,
    )

    # Expected output results:
    #     predicted_body_mass_g  	      species	                island	 culmen_length_mm  culmen_depth_mm   body_mass_g 	flipper_length_mm	sex
    # 23	  4681.782896	   Gentoo penguin (Pygoscelis papua)	Biscoe	      <NA>	            <NA>	        <NA>	          <NA>	        <NA>
    # 332	  4740.7907	       Gentoo penguin (Pygoscelis papua)	Biscoe	      46.2	            14.4	        214.0	          4650.0	    <NA>
    # 160	  4731.310452	   Gentoo penguin (Pygoscelis papua)	Biscoe	      44.5	            14.3	        216.0	          4100.0	    <NA>
    # [END bigquery_dataframes_bqml_linear_predict]
    # [START bigquery_dataframes_bqml_linear_predict_explain]
    # Use the ml.explain_predict method to understand why the model is
    # generating these prediction results.
    # For more information, see the BigQuery DataFrames API reference documentation:
    # https://dataframes.bigquery.dev/reference/api/bigframes.bigquery.ml.explain_predict.html#bigframes.bigquery.ml.explain_predict
    #
    # Using the trained model and utilizing data specific to Biscoe Island,
    # explain the predictions of the top 3 features.
    ml.explain_predict(
        your_model_id,  # For example: "bqml_tutorial.penguins_model",
        input_=biscoe_data,
        top_k_features=3,
    )

    # Expected results:
    #   predicted_body_mass_g               top_feature_attributions	        baseline_prediction_value	prediction_value	approximation_error	              species	            island	culmen_length_mm	culmen_depth_mm	flipper_length_mm	body_mass_g	    sex
    # 0	 5413.510134	        [{'feature': 'island', 'attribution': 7348.877...	-5320.222128	          5413.510134	            0.0	         Gentoo penguin (Pygoscelis papua)	Biscoe	    45.2	              16.4	        223.0	           5950.0	    MALE
    # 1	 4768.351092            [{'feature': 'island', 'attribution': 7348.877...	-5320.222128	          4768.351092	            0.0	         Gentoo penguin (Pygoscelis papua)	Biscoe	    46.5	              14.5	        213.0	           4400.0	   FEMALE
    # 2	 3235.896372	        [{'feature': 'island', 'attribution': 7348.877...	-5320.222128	          3235.896372	            0.0	        Adelie Penguin (Pygoscelis adeliae)	Biscoe	    37.7	              16.0          183.0	           3075.0	   FEMALE
    # 3	 5349.603734	        [{'feature': 'island', 'attribution': 7348.877...	-5320.222128	          5349.603734	            0.0	         Gentoo penguin (Pygoscelis papua)	Biscoe	    46.4	              15.6	        221.0	           5000.0	    MALE
    # 4	 4637.165037	        [{'feature': 'island', 'attribution': 7348.877...	-5320.222128	          4637.165037	            0.0	         Gentoo penguin (Pygoscelis papua)	Biscoe	    46.1	              13.2	        211.0	           4500.0	   FEMALE
    # [END bigquery_dataframes_bqml_linear_predict_explain]
    # [START bigquery_dataframes_bqml_linear_global_explain]
    # To use the ml.global_explain method, the model must be created with
    # enable_global_explain set to True.
    #
    # Use ml.create_model to create and train the model in BigQuery.
    # The options parameter specifies the model type and the label column.
    # For more information, see the BigQuery DataFrames API reference documentation:
    # https://dataframes.bigquery.dev/reference/api/bigframes.bigquery.ml.create_model.html#bigframes.bigquery.ml.create_model
    training_data = bq_df.dropna(subset=["body_mass_g"])
    ml.create_model(
        your_model_id,  # For example: "bqml_tutorial.penguins_model",
        options={
            "model_type": "LINEAR_REG",
            "input_label_cols": ["body_mass_g"],
            "enable_global_explain": True,
        },
        training_data=training_data,
        replace=True,
    )

    # Use the ml.global_explain method to explain the model.
    # For more information, see the BigQuery DataFrames API reference documentation:
    # https://dataframes.bigquery.dev/reference/api/bigframes.bigquery.ml.global_explain.html#bigframes.bigquery.ml.global_explain
    ml.global_explain(
        your_model_id,  # For example: "bqml_tutorial.penguins_model",
    )

    # Expected results:
    #                       attribution
    # feature
    # island	            5737.315921
    # species	            4073.280549
    # sex	                622.070896
    # flipper_length_mm	    193.612051
    # culmen_depth_mm	    117.084944
    # culmen_length_mm	    94.366793
    # [END bigquery_dataframes_bqml_linear_global_explain]
