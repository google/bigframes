CREATE MODEL `my_remote_model`
INPUT (`my field` STRING, `a\`b\\c` STRING, `already quoted` STRING, `padded` STRING)
OUTPUT (`f1 INT64) OUTPUT (injected` STRING)
REMOTE WITH CONNECTION `my_project.us.my_connection`
OPTIONS(endpoint = 'https://example.com/endpoint')
