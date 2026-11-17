# Copyright 2026 Google LLC
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
#
# DO NOT MODIFY THIS FILE DIRECTLY.
# This file was generated from: scripts/data/sql-functions/global_namespace/datetime.yaml
# by the script: scripts/generate_bigframes_bigquery.py

from __future__ import annotations

import datetime as dt
from typing import Literal, Union

from dateutil import relativedelta

import bigframes.core.col
import bigframes.core.googlesql
import bigframes.core.sentinels as sentinels
import bigframes.series as series
from bigframes import dtypes
from bigframes.operations import googlesql

_CURRENT_DATETIME_OP = googlesql.GoogleSqlScalarOp(
    "CURRENT_DATETIME",
    args=(googlesql.ArgSpec(optional=True),),
    signature=lambda *args: dtypes.DATETIME_DTYPE,
)
_DATETIME_OP = googlesql.GoogleSqlScalarOp(
    "DATETIME",
    args=(
        googlesql.ArgSpec(optional=True),
        googlesql.ArgSpec(optional=True),
        googlesql.ArgSpec(optional=True),
        googlesql.ArgSpec(optional=True),
        googlesql.ArgSpec(optional=True),
        googlesql.ArgSpec(optional=True),
        googlesql.ArgSpec(optional=True),
        googlesql.ArgSpec(optional=True),
        googlesql.ArgSpec(optional=True),
    ),
    signature=lambda *args: dtypes.DATETIME_DTYPE,
)
_DATETIME_ADD_OP = googlesql.GoogleSqlScalarOp(
    "DATETIME_ADD",
    args=(googlesql.ArgSpec(), googlesql.ArgSpec(dtype=dtypes.INTERVAL_DTYPE)),
    signature=lambda *args: dtypes.DATETIME_DTYPE,
)
_DATETIME_DIFF_OP = googlesql.GoogleSqlScalarOp(
    "DATETIME_DIFF",
    args=(
        googlesql.ArgSpec(),
        googlesql.ArgSpec(),
        googlesql.ArgSpec(dtype="date_part"),
    ),
    signature=lambda *args: dtypes.INT_DTYPE,
)
_DATETIME_SUB_OP = googlesql.GoogleSqlScalarOp(
    "DATETIME_SUB",
    args=(googlesql.ArgSpec(), googlesql.ArgSpec(dtype=dtypes.INTERVAL_DTYPE)),
    signature=lambda *args: dtypes.DATETIME_DTYPE,
)


def _DATETIME_TRUNC_SIG(*args):
    # Pad args with None to match max expected args
    args = args + (None,) * (3 - len(args))
    # Try matching impl 0
    match_ok = True
    if match_ok and args[0] is not None:
        try:
            if (
                dtypes.coerce_to_common(args[0], dtypes.DATETIME_DTYPE)
                != dtypes.DATETIME_DTYPE
            ):
                match_ok = False
        except TypeError:
            match_ok = False
    if match_ok and args[1] is not None:
        pass
    if match_ok:
        return dtypes.DATETIME_DTYPE

    # Try matching impl 1
    match_ok = True
    if match_ok and args[0] is not None:
        try:
            if (
                dtypes.coerce_to_common(args[0], dtypes.TIMESTAMP_DTYPE)
                != dtypes.TIMESTAMP_DTYPE
            ):
                match_ok = False
        except TypeError:
            match_ok = False
    if match_ok and args[1] is not None:
        pass
    if match_ok and args[2] is not None:
        try:
            if (
                dtypes.coerce_to_common(args[2], dtypes.STRING_DTYPE)
                != dtypes.STRING_DTYPE
            ):
                match_ok = False
        except TypeError:
            match_ok = False
    if match_ok:
        return dtypes.TIMESTAMP_DTYPE

    raise TypeError(
        f"Could not find matching signature for datetime_trunc with argument types: {[str(t) for t in args]}"
    )


_DATETIME_TRUNC_OP = googlesql.GoogleSqlScalarOp(
    "DATETIME_TRUNC",
    args=(
        googlesql.ArgSpec(),
        googlesql.ArgSpec(dtype="date_part"),
        googlesql.ArgSpec(optional=True),
    ),
    signature=_DATETIME_TRUNC_SIG,
)
_FORMAT_DATETIME_OP = googlesql.GoogleSqlScalarOp(
    "FORMAT_DATETIME",
    args=(googlesql.ArgSpec(), googlesql.ArgSpec()),
    signature=lambda *args: dtypes.STRING_DTYPE,
)
_PARSE_DATETIME_OP = googlesql.GoogleSqlScalarOp(
    "PARSE_DATETIME",
    args=(googlesql.ArgSpec(), googlesql.ArgSpec()),
    signature=lambda *args: dtypes.DATETIME_DTYPE,
)


def current_datetime(
    time_zone: Union[
        series.Series,
        bigframes.core.col.Expression,
        Union[Literal[sentinels.Sentinel.ARGUMENT_DEFAULT], str],
    ] = sentinels.Sentinel.ARGUMENT_DEFAULT,
) -> Union[series.Series, bigframes.core.col.Expression]:
    """Returns the current date and time as a DATETIME value."""
    return bigframes.core.googlesql.apply_googlesql_scalar_op(
        _CURRENT_DATETIME_OP,
        time_zone,
    )


def datetime(
    expression: Union[
        series.Series,
        bigframes.core.col.Expression,
        Union[Literal[sentinels.Sentinel.ARGUMENT_DEFAULT], dt.date, dt.datetime, str],
    ] = sentinels.Sentinel.ARGUMENT_DEFAULT,
    time_zone: Union[
        series.Series,
        bigframes.core.col.Expression,
        Union[Literal[sentinels.Sentinel.ARGUMENT_DEFAULT], str],
    ] = sentinels.Sentinel.ARGUMENT_DEFAULT,
    time_expression: Union[
        series.Series,
        bigframes.core.col.Expression,
        Union[Literal[sentinels.Sentinel.ARGUMENT_DEFAULT], dt.time],
    ] = sentinels.Sentinel.ARGUMENT_DEFAULT,
    year: Union[
        series.Series,
        bigframes.core.col.Expression,
        Union[Literal[sentinels.Sentinel.ARGUMENT_DEFAULT], int],
    ] = sentinels.Sentinel.ARGUMENT_DEFAULT,
    month: Union[
        series.Series,
        bigframes.core.col.Expression,
        Union[Literal[sentinels.Sentinel.ARGUMENT_DEFAULT], int],
    ] = sentinels.Sentinel.ARGUMENT_DEFAULT,
    day: Union[
        series.Series,
        bigframes.core.col.Expression,
        Union[Literal[sentinels.Sentinel.ARGUMENT_DEFAULT], int],
    ] = sentinels.Sentinel.ARGUMENT_DEFAULT,
    hour: Union[
        series.Series,
        bigframes.core.col.Expression,
        Union[Literal[sentinels.Sentinel.ARGUMENT_DEFAULT], int],
    ] = sentinels.Sentinel.ARGUMENT_DEFAULT,
    minute: Union[
        series.Series,
        bigframes.core.col.Expression,
        Union[Literal[sentinels.Sentinel.ARGUMENT_DEFAULT], int],
    ] = sentinels.Sentinel.ARGUMENT_DEFAULT,
    second: Union[
        series.Series,
        bigframes.core.col.Expression,
        Union[Literal[sentinels.Sentinel.ARGUMENT_DEFAULT], int],
    ] = sentinels.Sentinel.ARGUMENT_DEFAULT,
) -> Union[series.Series, bigframes.core.col.Expression]:
    """Constructs a DATETIME value."""
    return bigframes.core.googlesql.apply_googlesql_scalar_op(
        _DATETIME_OP,
        expression,
        time_zone,
        time_expression,
        year,
        month,
        day,
        hour,
        minute,
        second,
    )


def datetime_add(
    datetime_expression: Union[
        series.Series,
        bigframes.core.col.Expression,
        Union[Literal[sentinels.Sentinel.ARGUMENT_DEFAULT], dt.datetime],
    ],
    interval: Union[
        series.Series,
        bigframes.core.col.Expression,
        Union[
            Literal[sentinels.Sentinel.ARGUMENT_DEFAULT], relativedelta.relativedelta
        ],
    ],
) -> Union[series.Series, bigframes.core.col.Expression]:
    """Adds a specified time interval to a DATETIME value."""
    return bigframes.core.googlesql.apply_googlesql_scalar_op(
        _DATETIME_ADD_OP,
        datetime_expression,
        interval,
    )


def datetime_diff(
    end_datetime: Union[
        series.Series,
        bigframes.core.col.Expression,
        Union[Literal[sentinels.Sentinel.ARGUMENT_DEFAULT], dt.datetime],
    ],
    start_datetime: Union[
        series.Series,
        bigframes.core.col.Expression,
        Union[Literal[sentinels.Sentinel.ARGUMENT_DEFAULT], dt.datetime],
    ],
    granularity: Union[
        series.Series,
        bigframes.core.col.Expression,
        Union[
            Literal[
                "YEAR",
                "ISOYEAR",
                "QUARTER",
                "MONTH",
                "WEEK",
                "ISOWEEK",
                "DAY",
                "DAYOFWEEK",
                "DAYOFYEAR",
                "HOUR",
                "MINUTE",
                "SECOND",
                "MILLISECOND",
                "MICROSECOND",
            ],
            Literal[sentinels.Sentinel.ARGUMENT_DEFAULT],
        ],
    ],
) -> Union[series.Series, bigframes.core.col.Expression]:
    """Gets the number of unit boundaries between two DATETIME values (end_datetime - start_datetime) at a particular time granularity."""
    return bigframes.core.googlesql.apply_googlesql_scalar_op(
        _DATETIME_DIFF_OP,
        end_datetime,
        start_datetime,
        granularity,
    )


def datetime_sub(
    datetime_expression: Union[
        series.Series,
        bigframes.core.col.Expression,
        Union[Literal[sentinels.Sentinel.ARGUMENT_DEFAULT], dt.datetime],
    ],
    interval: Union[
        series.Series,
        bigframes.core.col.Expression,
        Union[
            Literal[sentinels.Sentinel.ARGUMENT_DEFAULT], relativedelta.relativedelta
        ],
    ],
) -> Union[series.Series, bigframes.core.col.Expression]:
    """Subtracts a specified time interval from a DATETIME value."""
    return bigframes.core.googlesql.apply_googlesql_scalar_op(
        _DATETIME_SUB_OP,
        datetime_expression,
        interval,
    )


def datetime_trunc(
    datetime_value: Union[
        series.Series,
        bigframes.core.col.Expression,
        Union[Literal[sentinels.Sentinel.ARGUMENT_DEFAULT], dt.datetime],
    ],
    granularity: Union[
        series.Series,
        bigframes.core.col.Expression,
        Union[
            Literal[
                "YEAR",
                "ISOYEAR",
                "QUARTER",
                "MONTH",
                "WEEK",
                "ISOWEEK",
                "DAY",
                "DAYOFWEEK",
                "DAYOFYEAR",
                "HOUR",
                "MINUTE",
                "SECOND",
                "MILLISECOND",
                "MICROSECOND",
            ],
            Literal[sentinels.Sentinel.ARGUMENT_DEFAULT],
        ],
    ],
    time_zone: Union[
        series.Series,
        bigframes.core.col.Expression,
        Union[Literal[sentinels.Sentinel.ARGUMENT_DEFAULT], str],
    ] = sentinels.Sentinel.ARGUMENT_DEFAULT,
) -> Union[series.Series, bigframes.core.col.Expression]:
    """Truncates a DATETIME or TIMESTAMP value at a particular granularity."""
    return bigframes.core.googlesql.apply_googlesql_scalar_op(
        _DATETIME_TRUNC_OP,
        datetime_value,
        granularity,
        time_zone,
    )


def format_datetime(
    format_string: Union[
        series.Series,
        bigframes.core.col.Expression,
        Union[Literal[sentinels.Sentinel.ARGUMENT_DEFAULT], str],
    ],
    datetime_expr: Union[
        series.Series,
        bigframes.core.col.Expression,
        Union[Literal[sentinels.Sentinel.ARGUMENT_DEFAULT], dt.datetime],
    ],
) -> Union[series.Series, bigframes.core.col.Expression]:
    """Formats a DATETIME value according to a specified format string."""
    return bigframes.core.googlesql.apply_googlesql_scalar_op(
        _FORMAT_DATETIME_OP,
        format_string,
        datetime_expr,
    )


def parse_datetime(
    format_string: Union[
        series.Series,
        bigframes.core.col.Expression,
        Union[Literal[sentinels.Sentinel.ARGUMENT_DEFAULT], str],
    ],
    datetime_string: Union[
        series.Series,
        bigframes.core.col.Expression,
        Union[Literal[sentinels.Sentinel.ARGUMENT_DEFAULT], str],
    ],
) -> Union[series.Series, bigframes.core.col.Expression]:
    """Converts a STRING value to a DATETIME value."""
    return bigframes.core.googlesql.apply_googlesql_scalar_op(
        _PARSE_DATETIME_OP,
        format_string,
        datetime_string,
    )
