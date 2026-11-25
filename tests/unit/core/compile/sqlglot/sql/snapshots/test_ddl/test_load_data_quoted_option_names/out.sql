LOAD DATA INTO `my-project.my_dataset.my_table` OPTIONS (
  `description`='my table'
) FROM FILES (`format`='CSV', uris=['gs://bucket/path*'])