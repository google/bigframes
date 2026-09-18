WITH `bfcte_0` AS (
  SELECT
    `bfcol_9` AS `bfcol_12`,
    `bfcol_10` AS `bfcol_13`,
    `bfcol_11` AS `bfcol_14`
  FROM (
    (
      SELECT
        `json_col` AS `bfcol_9`,
        0 AS `bfcol_10`,
        ROW_NUMBER() OVER () - 1 AS `bfcol_11`
      FROM (
        SELECT
          `rowindex`,
          `json_col`,
          PARSE_JSON(`json_string_col`) AS `json_string_col`
        FROM `bigframes-dev`.`sqlglot_test`.`json_types` AS `bft_0`
      )
    )
    UNION ALL
    (
      SELECT
        `json_string_col` AS `bfcol_6`,
        1 AS `bfcol_7`,
        ROW_NUMBER() OVER () - 1 AS `bfcol_8`
      FROM (
        SELECT
          `rowindex`,
          `json_col`,
          PARSE_JSON(`json_string_col`) AS `json_string_col`
        FROM `bigframes-dev`.`sqlglot_test`.`json_types` AS `bft_0`
      )
    )
  )
)
SELECT
  `bfcol_12` AS `0`
FROM `bfcte_0`
ORDER BY
  `bfcol_13` ASC NULLS LAST,
  `bfcol_14` ASC NULLS LAST