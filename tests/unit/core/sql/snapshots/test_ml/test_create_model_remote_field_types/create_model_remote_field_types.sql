CREATE MODEL `my_remote_model`
INPUT (`a` ARRAY<STRUCT<x INT64, y STRING>>, `b` NUMERIC(10, 2), `c` STRUCT<`my field` INT64>, `e` STRING(MAX))
OUTPUT (`d` ARRAY<FLOAT64>)
REMOTE WITH CONNECTION `my_project.us.my_connection`
OPTIONS(endpoint = 'https://example.com/endpoint')
