MERGE INTO `bigframes-dev`.`my_catalog.my_namespace`.`dest_table`
USING (
  SELECT
    *
  FROM `source_table`
)
ON FALSE
WHEN NOT MATCHED BY SOURCE THEN DELETE
WHEN NOT MATCHED THEN INSERT ROW