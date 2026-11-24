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

import datetime
import decimal
import re

import bigframes_vendored.sqlglot as sg
import bigframes_vendored.sqlglot.expressions as sge
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import shapely.geometry  # type: ignore

import bigframes.core.compile.sqlglot.sql.base as sql
import bigframes.core.sql


@pytest.mark.parametrize(
    ("value", "expected_pattern"),
    (
        pytest.param(None, "NULL", id="null"),
        pytest.param(True, "TRUE", id="true"),
        pytest.param(False, "FALSE", id="false"),
        pytest.param(123, "123", id="int"),
        pytest.param(123.75, "123.75", id="float"),
        pytest.param("abc", "'abc'", id="string"),
        pytest.param(b"\x01\x02\x03ABC", "FROM_HEX('010203414243')", id="bytes"),
        pytest.param(
            decimal.Decimal("123.75"), "CAST(123.75 AS NUMERIC)", id="decimal"
        ),
        pytest.param(
            datetime.date(2025, 1, 1), "CAST('2025-01-01' AS DATE)", id="date"
        ),
        pytest.param(
            datetime.datetime(2025, 1, 2, 3, 45, 6, 789123),
            "CAST('2025-01-02T03:45:06.789123' AS DATETIME)",
            id="datetime",
        ),
        pytest.param(
            datetime.time(12, 34, 56, 789123),
            "CAST('12:34:56.789123' AS TIME)",
            id="time",
        ),
        pytest.param(
            datetime.datetime(
                2025, 1, 2, 3, 45, 6, 789123, tzinfo=datetime.timezone.utc
            ),
            "CAST('2025-01-02T03:45:06.789123+00:00' AS TIMESTAMP)",
            id="timestamp",
        ),
        pytest.param(
            pd.Timestamp("2026-03-01T09:52:10.799999952+00:00"),
            "CAST('2026-03-01T09:52:10.799999+00:00' AS TIMESTAMP)",
            id="timestamp_truncate_nanoseconds",
        ),
        pytest.param(
            pd.Timestamp("2026-03-01T09:52:10.799999952"),
            "CAST('2026-03-01T09:52:10.799999' AS DATETIME)",
            id="datetime_truncate_nanoseconds",
        ),
        pytest.param(np.int64(123), "123", id="np_int64"),
        pytest.param(np.float64(123.75), "123.75", id="np_float64"),
        pytest.param(float("inf"), "CAST('Infinity' AS FLOAT64)", id="inf"),
        pytest.param(float("-inf"), "CAST('-Infinity' AS FLOAT64)", id="neg_inf"),
        pytest.param(float("nan"), "NULL", id="nan"),
        pytest.param(pd.NA, "NULL", id="pd_na"),
        pytest.param(datetime.timedelta(seconds=1), "1000000", id="timedelta"),
        pytest.param("POINT (0 1)", "'POINT (0 1)'", id="string_geo"),
    ),
)
def test_literal(value, expected_pattern):
    assert sql.to_sql(sql.literal(value)) == expected_pattern


def test_literal_for_geo():
    value = shapely.geometry.Point(0, 1)
    expected_pattern = r"ST_GEOGFROMTEXT\('POINT \(0[.]?0* 1[.]?0*\)'\)"
    got = sql.to_sql(sql.literal(value))
    assert re.match(expected_pattern, got) is not None


@pytest.mark.parametrize(
    ("value", "dtype", "expected"),
    (
        pytest.param(
            decimal.Decimal("1.23"),
            sql.dtypes.BIGNUMERIC_DTYPE,
            "CAST(1.23 AS BIGNUMERIC)",
            id="bignumeric",
        ),
        pytest.param(
            [],
            pd.ArrowDtype(pa.list_(pa.int64())),
            "ARRAY<INT64>[]",
            id="empty_array",
        ),
        pytest.param(
            {"a": 1, "b": "hello"},
            pd.ArrowDtype(pa.struct([("a", pa.int64()), ("b", pa.string())])),
            "STRUCT(1 AS `a`, 'hello' AS `b`)",
            id="struct",
        ),
        pytest.param(
            float("nan"),
            sql.dtypes.FLOAT_DTYPE,
            "CAST('NaN' AS FLOAT64)",
            id="explicit_nan",
        ),
        pytest.param(
            pa.scalar(123, type=pa.int64()),
            None,
            "123",
            id="pa_scalar_int",
        ),
        pytest.param(
            pa.scalar(None, type=pa.int64()),
            None,
            "CAST(NULL AS INT64)",
            id="pa_scalar_null",
        ),
        pytest.param(
            {"a": 10},
            sql.dtypes.JSON_DTYPE,
            "PARSE_JSON('{\\'a\\': 10}')",
            id="json",
        ),
        pytest.param(
            2019,
            sql.dtypes.STRING_DTYPE,
            "'2019'",
            id="string_from_int",
        ),
        pytest.param(
            pa.scalar(2019),
            sql.dtypes.STRING_DTYPE,
            "'2019'",
            id="string_from_pyarrow_scalar",
        ),
        pytest.param(
            True,
            sql.dtypes.STRING_DTYPE,
            "'True'",
            id="string_from_bool",
        ),
        pytest.param(
            pd.Timestamp("2026-03-01 09:52:10.799999952+00:00"),
            sql.dtypes.TIMESTAMP_DTYPE,
            "CAST('2026-03-01T09:52:10.799999+00:00' AS TIMESTAMP)",
            id="timestamp_with_explicit_dtype",
        ),
        pytest.param(
            datetime.datetime(2025, 1, 2, 3, 45, 6),
            sql.dtypes.DATE_DTYPE,
            "CAST('2025-01-02' AS DATE)",
            id="date_from_datetime",
        ),
        pytest.param(
            pd.Timestamp("2026-03-01 09:52:10.799999952"),
            sql.dtypes.DATE_DTYPE,
            "CAST('2026-03-01' AS DATE)",
            id="date_from_timestamp",
        ),
        pytest.param(
            np.datetime64("2026-03-01T09:52:10.799999952"),
            sql.dtypes.DATETIME_DTYPE,
            "CAST('2026-03-01T09:52:10.799999' AS DATETIME)",
            id="np_datetime64_nanoseconds",
        ),
        pytest.param(
            "(SELECT 1)",
            sql.dtypes.BYTES_DTYPE,
            "FROM_HEX('2853454c454354203129')",
            id="bytes_from_string",
        ),
        pytest.param(
            "é",
            sql.dtypes.BYTES_DTYPE,
            "FROM_HEX('c3a9')",
            id="bytes_from_non_ascii_string",
        ),
        pytest.param(
            pa.scalar(b"\\' x"),
            sql.dtypes.BYTES_DTYPE,
            "FROM_HEX('5c272078')",
            id="bytes_from_pyarrow_scalar",
        ),
        pytest.param(
            bytearray(b"ab"),
            sql.dtypes.BYTES_DTYPE,
            "FROM_HEX('6162')",
            id="bytes_from_bytearray",
        ),
    ),
)
def test_literal_explicit_dtype(value, dtype, expected):
    assert sql.to_sql(sql.literal(value, dtype=dtype)) == expected


def test_literal_bytes_has_only_hex_digits_in_quotes():
    got = sql.to_sql(sql.literal(bytes(range(256))))
    assert re.fullmatch(r"FROM_HEX\(\s*'[0-9a-f]{512}'\s*\)", got) is not None


def test_literal_bytes_from_unsupported_type_raises():
    with pytest.raises(TypeError, match="Expected bytes or str for BYTES literal"):
        sql.literal(5, dtype=sql.dtypes.BYTES_DTYPE)


@pytest.mark.parametrize(
    ("value", "expected"),
    (
        pytest.param([True, False], "[TRUE, FALSE]", id="bool"),
        pytest.param([123, 456], "[123, 456]", id="int"),
        pytest.param(
            [123.75, 456.78, float("nan"), float("inf"), float("-inf")],
            "[\n  123.75,\n  456.78,\n  CAST('NaN' AS FLOAT64),\n  CAST('Infinity' AS FLOAT64),\n  CAST('-Infinity' AS FLOAT64)\n]",
            id="float",
        ),
        pytest.param(
            [b"\x01\x02\x03ABC", b"\x01\x02\x03ABC"],
            "[FROM_HEX('010203414243'), FROM_HEX('010203414243')]",
            id="bytes",
        ),
        pytest.param(
            [datetime.date(2025, 1, 1), datetime.date(2025, 1, 1)],
            "[CAST('2025-01-01' AS DATE), CAST('2025-01-01' AS DATE)]",
            id="date",
        ),
    ),
)
def test_literal_for_list(value: list, expected: str):
    got = sql.to_sql(sql.literal(value))
    assert got == expected


def test_literal_null_type():
    import unittest.mock as mock

    mock_dtype = mock.Mock()
    with mock.patch(
        "bigframes.core.compile.sqlglot.sql.base.sgt.from_bigframes_dtype",
        return_value="NULL",
    ):
        got = sql.to_sql(sql.literal(None, dtype=mock_dtype))
    assert got == "NULL"


SQLI_NAME = "m\\` OPTIONS(endpoint = 'evil') AS SELECT 1 --"

ESCAPED_NAMES = (
    pytest.param("a`b", r"`a\`b`", id="backtick"),
    pytest.param("a\\", r"`a\\`", id="backslash_at_end"),
    pytest.param("a\\b", r"`a\\b`", id="backslash"),
    pytest.param("a\\`b", r"`a\\\`b`", id="backslash_backtick"),
    pytest.param("a\nb", r"`a\nb`", id="line_feed"),
    pytest.param("a\rb", r"`a\rb`", id="carriage_return"),
    pytest.param(
        SQLI_NAME,
        r"`m\\\` OPTIONS(endpoint = 'evil') AS SELECT 1 --`",
        id="injection",
    ),
)

UNCHANGED_NAMES = (
    pytest.param("plain", id="plain"),
    pytest.param("with space", id="space"),
    pytest.param("日本語_é", id="non_english"),
    pytest.param("it's", id="single_quote"),
    pytest.param('say "hi"', id="double_quote"),
    pytest.param("what?", id="question_mark"),
    pytest.param("a-b:c#d", id="punctuation"),
    pytest.param("select", id="reserved_keyword"),
    pytest.param("1st", id="starts_with_digit"),
)


@pytest.mark.parametrize(("name", "expected"), ESCAPED_NAMES)
def test_identifier_escapes_name(name, expected):
    assert sql.to_sql(sql.identifier(name)) == expected


@pytest.mark.parametrize("name", UNCHANGED_NAMES)
def test_identifier_keeps_name(name):
    assert sql.to_sql(sql.identifier(name)) == f"`{name}`"


@pytest.mark.parametrize("name", ["SYSTEM_TIME", "a\\b", "a`b"])
def test_unquoted_identifier_is_not_escaped(name):
    assert sql.to_sql(sge.Identifier(this=name, quoted=False)) == name


@pytest.mark.parametrize(
    ("name", "identify", "expected"),
    (
        pytest.param("select", False, "`select`", id="reserved_keyword"),
        pytest.param("1a`b", False, r"`1a\`b`", id="starts_with_digit"),
        pytest.param("a`b", True, r"`a\`b`", id="identify"),
    ),
)
def test_unquoted_identifier_gets_backticks_when_required(name, identify, expected):
    expr = sge.Identifier(this=name, quoted=False)

    assert expr.sql(dialect=sql.DIALECT, identify=identify) == expected


@pytest.mark.parametrize(
    "name", [param.values[0] for param in (*ESCAPED_NAMES, *UNCHANGED_NAMES)]
)
def test_identifier_round_trips(name):
    written = sql.to_sql(sql.identifier(name))

    parsed = sg.parse_one(written, dialect=sql.DIALECT)

    assert parsed.name == name
    assert sql.to_sql(parsed) == written


@pytest.mark.parametrize(
    "name",
    [
        *(param.values[0] for param in (*ESCAPED_NAMES, *UNCHANGED_NAMES)),
        "a\tb",
        "a\a\b\f\vb",
    ],
)
def test_core_sql_identifier_is_parsed_back(name):
    written = bigframes.core.sql.identifier(name)

    parsed = sg.parse_one(written, dialect=sql.DIALECT)

    assert parsed.name == name


@pytest.mark.parametrize(
    ("written", "name"),
    (
        pytest.param(r"`a\x41b`", "aAb", id="hex"),
        pytest.param(r"`a\X41b`", "aAb", id="hex_uppercase_x"),
        pytest.param(r"`a\101b`", "aAb", id="octal"),
        pytest.param(r"`caf\u00e9`", "café", id="unicode_4"),
        pytest.param(r"`a\U0001F600b`", "a😀b", id="unicode_8"),
        pytest.param(r"`a\qb`", "a\\qb", id="unknown_escape"),
        pytest.param(r"`a\ud800b`", "a\\ud800b", id="surrogate"),
        pytest.param(r"`a\U00110000b`", "a\\U00110000b", id="out_of_range"),
    ),
)
def test_parse_quoted_identifier_escape_sequence(written, name):
    parsed = sg.parse_one(written, dialect=sql.DIALECT)

    assert parsed.name == name


def test_parse_doubled_backtick_as_two_names():
    parsed = sg.parse_one("SELECT `a``b` FROM `t`", dialect=sql.DIALECT)

    assert parsed.sql(dialect=sql.DIALECT) == "SELECT `a` AS `b` FROM `t`"


@pytest.mark.parametrize("written", ["`a", "`a\\`", "`a\\"])
def test_parse_unterminated_quoted_identifier_raises(written):
    with pytest.raises(sg.TokenError):
        sg.parse_one(written, dialect=sql.DIALECT)


def test_parse_dotted_table_name_in_one_pair_of_backticks():
    written = "SELECT * FROM `proj.ds.tbl` AS `t`"

    parsed = sg.parse_one(written, dialect=sql.DIALECT)

    assert parsed.sql(dialect=sql.DIALECT) == written


@pytest.mark.parametrize(
    ("parts", "expected"),
    (
        pytest.param(("only_one",), "`only_one`", id="single"),
        pytest.param(
            ("my-proj", "my_ds", "my_fn"),
            "`my-proj`.`my_ds`.`my_fn`",
            id="three_parts",
        ),
        pytest.param(
            ("google.com:my-proj", "my_ds", "my_fn"),
            "`google.com:my-proj`.`my_ds`.`my_fn`",
            id="domain_scoped_project",
        ),
        pytest.param(
            ("proj", "ds.extra", "fn`back\\tick\nline"),
            r"`proj`.`ds.extra`.`fn\`back\\tick\nline`",
            id="special_chars_in_segments",
        ),
    ),
)
def test_qualified_identifier(parts, expected):
    assert sql.to_sql(sql.qualified_identifier(*parts)) == expected


def test_qualified_identifier_empty_raises():
    with pytest.raises(ValueError, match="at least one identifier part"):
        sql.qualified_identifier()
