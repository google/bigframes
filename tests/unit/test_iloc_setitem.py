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

from typing import Generator

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import bigframes
import bigframes.pandas as bpd
from bigframes.testing.utils import assert_frame_equal

pytest.importorskip("polars")


@pytest.fixture(scope="module", autouse=True)
def session() -> Generator[bigframes.Session, None, None]:
    # Perform Polars imports here because the library is not available in 3.10 test envs.
    import bigframes.core.global_session
    from bigframes.testing import polars_session

    session = polars_session.TestSession()
    with bigframes.core.global_session._GlobalSessionContext(session):
        yield session


@pytest.fixture
def pd_df() -> pd.DataFrame:
    # Explicitly use "Int64" dtype for testing convenience to match BigFrames default integer dtype.
    return pd.DataFrame(
        {
            "A": [1, 2, 3],
            "B": [4, 5, 6],
            "C": [7, 8, 9],
        },
        dtype="Int64",
    )


@pytest.fixture
def duplicate_columns_pd_df() -> pd.DataFrame:
    # Explicitly use int64 dtype because some operators involving pyarrow array are not supported.
    return pd.DataFrame(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        columns=["A", "B", "A"],
        dtype="int64",
    )


def test_iloc_setitem_column_single_integer(pd_df):
    bf_df = bpd.read_pandas(pd_df)

    bf_df.iloc[:, 1] = 99
    pd_df.iloc[:, 1] = 99

    assert_frame_equal(bf_df.to_pandas(), pd_df, check_index_type=False)


def test_iloc_setitem_column_single_integer_negative(pd_df):
    bf_df = bpd.read_pandas(pd_df)

    bf_df.iloc[:, -1] = 99
    pd_df.iloc[:, -1] = 99

    assert_frame_equal(bf_df.to_pandas(), pd_df, check_index_type=False)


def test_iloc_setitem_columns_list_integer(pd_df):
    bf_df = bpd.read_pandas(pd_df)

    bf_df.iloc[:, [0, 2]] = [99, 88]
    pd_df.iloc[:, [0, 2]] = [99, 88]

    assert_frame_equal(bf_df.to_pandas(), pd_df, check_index_type=False)


def test_iloc_setitem_columns_slice(pd_df):
    bf_df = bpd.read_pandas(pd_df)

    bf_df.iloc[:, 0:2] = 99
    pd_df.iloc[:, 0:2] = 99

    assert_frame_equal(bf_df.to_pandas(), pd_df, check_index_type=False)


def test_iloc_setitem_columns_boolean_mask(pd_df):
    bf_df = bpd.read_pandas(pd_df)

    mask = [True, False, True]
    bf_df.iloc[:, mask] = 99
    pd_df.iloc[:, np.array(mask)] = 99

    assert_frame_equal(bf_df.to_pandas(), pd_df, check_index_type=False)


def test_iloc_setitem_columns_dataframe(pd_df):
    bf_df = bpd.read_pandas(pd_df)

    value_df = bpd.DataFrame({"B": [99, 88, 77], "C": [66, 55, 44]})
    bf_df.iloc[:, 1:3] = value_df
    pd_df.iloc[:, 1:3] = value_df.to_pandas()

    assert_frame_equal(bf_df.to_pandas(), pd_df, check_index_type=False)


def test_iloc_setitem_column_numpy_scalar(pd_df):
    bf_df = bpd.read_pandas(pd_df)

    bf_df.iloc[:, np.int64(1)] = 99
    pd_df.iloc[:, np.int64(1)] = 99

    assert_frame_equal(bf_df.to_pandas(), pd_df, check_index_type=False)


def test_iloc_setitem_columns_numpy_array(pd_df):
    bf_df = bpd.read_pandas(pd_df)

    bf_df.iloc[:, np.array([0, 2], dtype=np.int64)] = [99, 88]
    pd_df.iloc[:, np.array([0, 2], dtype=np.int64)] = [99, 88]

    assert_frame_equal(bf_df.to_pandas(), pd_df, check_index_type=False)


def test_iloc_setitem_column_pyarrow_scalar(pd_df):
    bf_df = bpd.read_pandas(pd_df)

    bf_df.iloc[:, pa.scalar(1, type=pa.int64())] = 99
    pd_df.iloc[:, 1] = 99

    assert_frame_equal(bf_df.to_pandas(), pd_df, check_index_type=False)


def test_iloc_setitem_columns_pyarrow_array(pd_df):
    bf_df = bpd.read_pandas(pd_df)

    bf_df.iloc[:, pa.array([0, 2], type=pa.int64())] = [99, 88]
    pd_df.iloc[:, [0, 2]] = [99, 88]

    assert_frame_equal(bf_df.to_pandas(), pd_df, check_index_type=False)


@pytest.mark.parametrize(
    ["key", "expected_error"],
    [
        pytest.param((slice(None), 3), IndexError, id="out_of_bounds_positive"),
        pytest.param((slice(None), -4), IndexError, id="out_of_bounds_negative"),
        pytest.param((0, 1), NotImplementedError, id="invalid_row_indexer"),
        pytest.param((slice(None), "B"), TypeError, id="invalid_col_indexer_type"),
    ],
)
def test_iloc_setitem_column_errors(pd_df, key, expected_error):
    bf_df = bpd.read_pandas(pd_df)

    with pytest.raises(expected_error):
        bf_df.iloc[key] = 99


def test_iloc_setitem_duplicate_columns_single_integer(duplicate_columns_pd_df):
    bf_df = bpd.read_pandas(duplicate_columns_pd_df)
    pd_df = duplicate_columns_pd_df

    bf_df.iloc[:, 2] = 99
    pd_df.iloc[:, 2] = 99

    assert_frame_equal(
        bf_df.to_pandas(), pd_df, check_index_type=False, check_dtype=False
    )


def test_iloc_setitem_duplicate_columns_list_integer(duplicate_columns_pd_df):
    bf_df = bpd.read_pandas(duplicate_columns_pd_df)
    pd_df = duplicate_columns_pd_df

    bf_df.iloc[:, [0, 2]] = [99, 88]
    pd_df.iloc[:, [0, 2]] = [99, 88]

    assert_frame_equal(
        bf_df.to_pandas(), pd_df, check_index_type=False, check_dtype=False
    )


def test_iloc_setitem_duplicate_columns_slice(duplicate_columns_pd_df):
    bf_df = bpd.read_pandas(duplicate_columns_pd_df)
    pd_df = duplicate_columns_pd_df

    bf_df.iloc[:, 1:3] = 99
    pd_df.iloc[:, 1:3] = 99

    assert_frame_equal(
        bf_df.to_pandas(), pd_df, check_index_type=False, check_dtype=False
    )


def test_iloc_setitem_duplicate_columns_numpy_scalar(duplicate_columns_pd_df):
    bf_df = bpd.read_pandas(duplicate_columns_pd_df)
    pd_df = duplicate_columns_pd_df

    bf_df.iloc[:, np.int64(2)] = 99
    pd_df.iloc[:, np.int64(2)] = 99

    assert_frame_equal(
        bf_df.to_pandas(), pd_df, check_index_type=False, check_dtype=False
    )


def test_iloc_setitem_duplicate_columns_numpy_array(duplicate_columns_pd_df):
    bf_df = bpd.read_pandas(duplicate_columns_pd_df)
    pd_df = duplicate_columns_pd_df

    bf_df.iloc[:, np.array([0, 2], dtype=np.int64)] = [99, 88]
    pd_df.iloc[:, np.array([0, 2], dtype=np.int64)] = [99, 88]

    assert_frame_equal(
        bf_df.to_pandas(), pd_df, check_index_type=False, check_dtype=False
    )


def test_iloc_setitem_duplicate_columns_pyarrow_scalar(duplicate_columns_pd_df):
    bf_df = bpd.read_pandas(duplicate_columns_pd_df)
    pd_df = duplicate_columns_pd_df

    bf_df.iloc[:, pa.scalar(2, type=pa.int64())] = 99
    pd_df.iloc[:, 2] = 99

    assert_frame_equal(
        bf_df.to_pandas(), pd_df, check_index_type=False, check_dtype=False
    )


def test_iloc_setitem_duplicate_columns_pyarrow_array(duplicate_columns_pd_df):
    bf_df = bpd.read_pandas(duplicate_columns_pd_df)
    pd_df = duplicate_columns_pd_df

    bf_df.iloc[:, pa.array([0, 2], type=pa.int64())] = [99, 88]
    pd_df.iloc[:, pa.array([0, 2], type=pa.int64())] = [99, 88]

    assert_frame_equal(
        bf_df.to_pandas(), pd_df, check_index_type=False, check_dtype=False
    )
