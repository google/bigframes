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


def test_kmeans_sample(project_id: str, random_model_id_eu: str) -> None:
    your_gcp_project_id = project_id
    your_model_id = random_model_id_eu
    # [START bigquery_dataframes_bqml_kmeans]
    import datetime
    import typing

    import pandas as pd
    from shapely.geometry import Point

    import bigframes
    import bigframes.bigquery as bbq
    import bigframes.geopandas
    import bigframes.pandas as bpd

    bigframes.options.bigquery.project = your_gcp_project_id
    # Compute in the EU multi-region to query the London bicycles dataset.
    bigframes.options.bigquery.location = "EU"

    # Set partial ordering mode for BigQuery DataFrames.
    # For more information, see the BigQuery DataFrames performance documentation:
    # https://cloud.google.com/bigquery/docs/dataframes-performance#partial-ordering-mode
    bpd.options.bigquery.ordering_mode = "partial"

    # Extract the information you'll need to train the k-means model in this
    # tutorial. Use the read_gbq function to represent cycle hires
    # data as a DataFrame.
    h = bpd.read_gbq(
        "bigquery-public-data.london_bicycles.cycle_hire",
        col_order=["start_station_name", "start_station_id", "start_date", "duration"],
    ).rename(
        columns={
            "start_station_name": "station_name",
            "start_station_id": "station_id",
        }
    )

    # Use GeoSeries.from_xy and BigQuery.st_distance to analyze geographical
    # data. These functions determine spatial relationships between
    # geographical features.
    cycle_stations = bpd.read_gbq("bigquery-public-data.london_bicycles.cycle_stations")
    s = bpd.DataFrame(
        {
            "id": cycle_stations["id"],
            "xy": bigframes.geopandas.GeoSeries.from_xy(
                cycle_stations["longitude"], cycle_stations["latitude"]
            ),
        }
    )
    s_distance = bbq.st_distance(s["xy"], Point(-0.1, 51.5), use_spheroid=False) / 1000
    s = bpd.DataFrame({"id": s["id"], "distance_from_city_center": s_distance})

    # Define Python datetime objects in the UTC timezone for range comparison,
    # because BigQuery stores timestamp data in the UTC timezone.
    sample_time = datetime.datetime(2015, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
    sample_time2 = datetime.datetime(2016, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)

    h = h[(h["start_date"] >= sample_time) & (h["start_date"] <= sample_time2)]

    # Replace each day-of-the-week number with the corresponding "weekday" or
    # "weekend" label by using the Series.case_when method.
    dayofweek = h["start_date"].dt.dayofweek
    h = h.assign(
        isweekday=dayofweek.case_when(
            [
                (dayofweek.isin([5, 6]), "weekend"),
                (True, "weekday"),
            ]
        )
    )

    # Supplement each trip in "h" with the station distance information from
    # "s" by merging the two DataFrames by station ID.
    merged_df = h.merge(
        right=s,
        how="inner",
        left_on="station_id",
        right_on="id",
    )

    # Engineer features to cluster the stations. For each station, find the
    # average trip duration, number of trips, and distance from city center.
    stationstats = typing.cast(
        bpd.DataFrame,
        merged_df.groupby(["station_name", "isweekday"]).agg(
            {"duration": ["mean", "count"], "distance_from_city_center": "max"}
        ),
    )
    stationstats.columns = pd.Index(
        ["duration", "num_trips", "distance_from_city_center"]
    )
    stationstats = stationstats.sort_values(
        by="distance_from_city_center", ascending=True
    ).reset_index()

    # Expected output results: >>> stationstats.head(3)
    # station_name	isweekday duration  num_trips	distance_from_city_center
    # Borough Road...	weekday	    1110	    5749	    0.12624
    # Borough Road...	weekend	    2125	    1774	    0.12624
    # Webber Street...	weekday	    795	        6517	    0.164021
    #   3 rows × 5 columns

    # [END bigquery_dataframes_bqml_kmeans]

    # [START bigquery_dataframes_bqml_kmeans_fit]
    from bigframes.bigquery import ml

    # A k-means model groups data into clusters, which is useful for
    # descriptive analytics.
    #
    # Extract only the numerical feature columns for model training.
    # 'station_name' and 'isweekday' are excluded so clustering is based on
    # bicycle usage patterns rather than station identity or day of week.
    features = stationstats[["duration", "num_trips", "distance_from_city_center"]]

    # Use ml.create_model to create and train the model in BigQuery.
    # The options parameter specifies the model type and the number of clusters.
    # For more information, see the BigQuery DataFrames API reference documentation:
    # https://dataframes.bigquery.dev/reference/api/bigframes.bigquery.ml.create_model.html#bigframes.bigquery.ml.create_model
    ml.create_model(
        your_model_id,  # For example: "bqml_tutorial.london_station_clusters",
        options={
            "model_type": "KMEANS",
            "num_clusters": 4,
        },
        training_data=features,
        replace=True,
    )
    # [END bigquery_dataframes_bqml_kmeans_fit]

    # [START bigquery_dataframes_bqml_kmeans_predict]
    from bigframes.bigquery import ml

    # Use 'contains' function to filter by stations containing the string
    # "Kennington".
    stationstats = stationstats[stationstats["station_name"].str.contains("Kennington")]

    # Use the ml.predict method to predict results using your model.
    # For more information, see the BigQuery DataFrames API reference documentation:
    # https://dataframes.bigquery.dev/reference/api/bigframes.bigquery.ml.predict.html#bigframes.bigquery.ml.predict
    ml.predict(
        your_model_id,  # For example: "bqml_tutorial.london_station_clusters",
        input_=stationstats,
    )

    # Expected output results:
    # CENTROID...	NEAREST...	station_name  isweekday	 duration num_trips dist...
    # 	1	[{'CENTROID_ID'...	Borough...	  weekday	  1110	    5749	0.13
    # 	2	[{'CENTROID_ID'...	Borough...	  weekend	  2125      1774	0.13
    # 	1	[{'CENTROID_ID'...	Webber...	  weekday	  795	    6517	0.16
    #   3 rows × 7 columns
    # [END bigquery_dataframes_bqml_kmeans_predict]
