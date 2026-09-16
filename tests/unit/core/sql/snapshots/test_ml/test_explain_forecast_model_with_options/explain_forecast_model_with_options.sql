SELECT * FROM ML.EXPLAIN_FORECAST(MODEL `my_model`, STRUCT(30 AS `horizon`, 0.8 AS `confidence_level`), (SELECT * FROM new_data))
