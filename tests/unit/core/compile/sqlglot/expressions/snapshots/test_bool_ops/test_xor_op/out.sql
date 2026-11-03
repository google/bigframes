SELECT
  `rowindex`,
  `bool_col`,
  `int64_col`,
  `int64_col` ^ `int64_col` AS `int_and_int`,
  (
    `bool_col` AND NOT `bool_col`
  ) OR (
    NOT `bool_col` AND `bool_col`
  ) AS `bool_and_bool`,
  CAST(NULL AS BOOLEAN) AS `bool_and_null`,
  CAST(NULL AS BOOLEAN) AS `null_and_bool`,
  CAST(NULL AS BOOLEAN) AS `null_and_null`
FROM `bigframes-dev`.`sqlglot_test`.`scalar_types` AS `bft_0`