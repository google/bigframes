# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from bigframes.core import sql


def test_create_vector_search_sql_simple():
    result_query = sql.create_vector_search_sql(
        sql_string="SELECT embedding FROM my_embeddings_table WHERE id = 1",
        base_table="my_base_table",
        column_to_search="my_embedding_column",
    )
    assert (
        result_query
        == """
    SELECT
        query.*,
        base.*,
        distance,
    FROM VECTOR_SEARCH(TABLE `my_base_table`,
'my_embedding_column',
(SELECT embedding FROM my_embeddings_table WHERE id = 1))
    """
    )


def test_create_vector_search_sql_all_named_parameters():
    result_query = sql.create_vector_search_sql(
        sql_string="SELECT embedding FROM my_embeddings_table WHERE id = 1",
        base_table="my_base_table",
        column_to_search="my_embedding_column",
        query_column_to_search="another_embedding_column",
        top_k=10,
        distance_type="cosine",
        options={
            "fraction_lists_to_search": 0.1,
            "use_brute_force": False,
        },
    )
    assert (
        result_query
        == """
    SELECT
        query.*,
        base.*,
        distance,
    FROM VECTOR_SEARCH(TABLE `my_base_table`,
'my_embedding_column',
(SELECT embedding FROM my_embeddings_table WHERE id = 1),
query_column_to_search => 'another_embedding_column',
top_k=> 10,
distance_type => 'cosine',
options => '{"fraction_lists_to_search": 0.1, "use_brute_force": false}')
    """
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("`a b`", True),
        (r"`a\`b`", True),
        ("`a` STRING, `b`", False),
        ("`a``b`", False),
        ("`a\\\rb`", False),
        ("``", False),
    ],
)
def test_is_quoted_identifier(text, expected):
    assert sql.is_quoted_identifier(text) is expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("model_type", "model_type"),
        ("_option1", "_option1"),
        ("`model_type`", "`model_type`"),
        (" model_type\n", "model_type"),
    ],
)
def test_normalize_option_name(name, expected):
    assert sql.normalize_option_name(name) == expected


@pytest.mark.parametrize(
    "name",
    [
        "model_type = 'LINEAR_REG', l2_reg",
        "`a` = 1, `b`",
        "model-type",
        "   ",
        None,
    ],
)
def test_normalize_option_name_invalid(name):
    with pytest.raises(ValueError, match="Invalid option name"):
        sql.normalize_option_name(name)


@pytest.mark.parametrize(
    ("data_type", "expected"),
    [
        ("ARRAY<STRUCT<a INT64, b STRING>>", "ARRAY<STRUCT<a INT64, b STRING>>"),
        (
            r"STRUCT<`my field` INT64, `a\`b` ARRAY<STRING(MAX)>>",
            r"STRUCT<`my field` INT64, `a\`b` ARRAY<STRING(MAX)>>",
        ),
        ("STRUCT<`user's name` STRING>", "STRUCT<`user's name` STRING>"),
        ("NUMERIC(10, 2)", "NUMERIC(10, 2)"),
        ("MAP<STRING, INT64>", "MAP<STRING, INT64>"),
        ("array<float64>", "ARRAY<FLOAT64>"),
        ("DOUBLE PRECISION", "FLOAT64"),
        ("STRING -- comment", "STRING"),
        ("STRUCT<`a\nb` INT64>", r"STRUCT<`a\nb` INT64>"),
    ],
)
def test_normalize_data_type(data_type, expected):
    assert sql.normalize_data_type(data_type) == expected


@pytest.mark.parametrize(
    "data_type",
    [
        "float64) OUTPUT (injected_column array<float64>",
        "STRING, injected INT64",
        "STRING INT64",
        "STRING; INT64",
        "STRING((SELECT 1))",
        "STRUCT<a INT64>((SELECT 1))",
        "ARRAY<INT64>[0]",
        "STRING OPTIONS(description = 'x')",
        "STRUCT<a STRING OPTIONS(description = `'ok')>) OUTPUT (x INT64`)>",
        r"STRUCT<a STRING OPTIONS(description = b'\\', y = ')>) -- ')>",
        "INT64 AS (1)",
        "INT64 DEFAULT 1 < 2, injected STRING DEFAULT 3 > 2",
        "INT64 < 2, injected STRING > 1",
        "STRUCT<`` INT64>",
        "ARRAY<INT64>>",
        "my_udt",
        "",
        None,
    ],
)
def test_normalize_data_type_invalid(data_type):
    with pytest.raises(ValueError, match="Invalid data type"):
        sql.normalize_data_type(data_type)


def test_create_vector_index_ddl_invalid_option_name_raises():
    with pytest.raises(ValueError, match="Invalid option name"):
        sql.create_vector_index_ddl(
            index_name="my_index",
            table_name="my_table",
            column_name="my_col",
            replace=False,
            stored_column_names=[],
            options={"distance_type = 'COSINE', index_type": "IVF"},
        )
