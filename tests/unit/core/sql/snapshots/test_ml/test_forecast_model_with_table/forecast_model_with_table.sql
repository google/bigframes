SELECT * FROM ML.FORECAST(MODEL `my_project.my_dataset.my_model`, STRUCT(), (SELECT * FROM new_data))
