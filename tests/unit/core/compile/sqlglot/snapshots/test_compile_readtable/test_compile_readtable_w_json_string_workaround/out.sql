SELECT
  `rowindex`,
  `json_col`,
  PARSE_JSON(`json_string_col`) AS `json_string_col`
FROM `bigframes-dev`.`sqlglot_test`.`json_types` AS `bft_0`