SELECT * FROM ML.RECOMMEND(MODEL `my_model`, (SELECT * FROM new_data), STRUCT(3 AS `trial_id`))
