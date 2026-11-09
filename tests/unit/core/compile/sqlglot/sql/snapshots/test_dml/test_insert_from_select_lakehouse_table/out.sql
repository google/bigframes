INSERT INTO `bigframes-dev`.`my_catalog.my_namespace`.`dest_table`
(
  SELECT
    *
  FROM `source_table`
)