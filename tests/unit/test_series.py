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

from typing import cast

import pytest

import bigframes.series
from bigframes.testing import mocks


def test_series_rename(monkeypatch: pytest.MonkeyPatch):
    series = cast(bigframes.series.Series, mocks.create_dataframe(monkeypatch)["col"])
    assert series.name == "col"
    renamed = series.rename("renamed_col")
    assert renamed.name == "renamed_col"


def test_series_rename_inplace_returns_none(monkeypatch: pytest.MonkeyPatch):
    series = cast(bigframes.series.Series, mocks.create_dataframe(monkeypatch)["col"])
    assert series.name == "col"
    assert series.rename("renamed_col", inplace=True) is None
    assert series.name == "renamed_col"


def test_series_rename_axis(monkeypatch: pytest.MonkeyPatch):
    series = mocks.create_dataframe(
        monkeypatch, data={"index1": [], "index2": [], "col1": [], "col2": []}
    ).set_index(["index1", "index2"])["col1"]
    assert list(series.index.names) == ["index1", "index2"]
    renamed = series.rename_axis(["a", "b"])
    assert list(renamed.index.names) == ["a", "b"]


def test_series_rename_axis_inplace_returns_none(monkeypatch: pytest.MonkeyPatch):
    series = mocks.create_dataframe(
        monkeypatch, data={"index1": [], "index2": [], "col1": [], "col2": []}
    ).set_index(["index1", "index2"])["col1"]
    assert list(series.index.names) == ["index1", "index2"]
    assert series.rename_axis(["a", "b"], inplace=True) is None
    assert list(series.index.names) == ["a", "b"]


def test_series_repr_with_uninitialized_object():
    """Ensures Series.__init__ can be paused in a visual debugger without crashing.

    Regression test for https://github.com/googleapis/python-bigquery-dataframes/issues/728
    """
    # Avoid calling __init__ to simulate pausing __init__ in a debugger.
    # https://stackoverflow.com/a/6384982/101923
    series = bigframes.series.Series.__new__(bigframes.series.Series)
    got = repr(series)
    assert "Series" in got


def test_series_fillna_inplace_false(monkeypatch: pytest.MonkeyPatch):
    series = cast(
        bigframes.series.Series,
        mocks.create_dataframe(
            monkeypatch, data={"idx": [1, 2], "col": [10.0, None]}
        ).set_index("idx")["col"],
    )
    original_block = series._block
    original_name = series.name
    original_index_names = list(series.index.names)

    result = series.fillna(0.0, inplace=False)
    assert result is not None
    assert result is not series
    assert series._block is original_block
    assert result._block is not original_block
    assert series.name == original_name
    assert list(series.index.names) == original_index_names


def test_series_fillna_inplace_true(monkeypatch: pytest.MonkeyPatch):
    series = cast(
        bigframes.series.Series,
        mocks.create_dataframe(
            monkeypatch, data={"idx": [1, 2], "col": [10.0, None]}
        ).set_index("idx")["col"],
    )
    original_block = series._block
    original_name = series.name
    original_index_names = list(series.index.names)

    inplace_result = series.fillna(0.0, inplace=True)
    assert inplace_result is None
    assert series._block is not original_block
    assert series.name == original_name
    assert list(series.index.names) == original_index_names


def test_series_fillna_series_inplace_false(monkeypatch: pytest.MonkeyPatch):
    series = cast(
        bigframes.series.Series,
        mocks.create_dataframe(
            monkeypatch, data={"idx": [1, 2], "col": [10.0, None]}
        ).set_index("idx")["col"],
    )
    fill_series = cast(
        bigframes.series.Series,
        mocks.create_dataframe(
            monkeypatch,
            session=series._session,
            data={"idx": [1, 2], "col": [0.0, 99.0]},
        ).set_index("idx")["col"],
    )
    original_block = series._block
    original_name = series.name
    original_index_names = list(series.index.names)

    result = series.fillna(fill_series, inplace=False)
    assert result is not None
    assert result is not series
    assert series._block is original_block
    assert result._block is not original_block
    assert series.name == original_name
    assert list(series.index.names) == original_index_names
    assert result.name == original_name
    assert list(result.index.names) == original_index_names


def test_series_fillna_series_inplace_true(monkeypatch: pytest.MonkeyPatch):
    series = cast(
        bigframes.series.Series,
        mocks.create_dataframe(
            monkeypatch, data={"idx": [1, 2], "col": [10.0, None]}
        ).set_index("idx")["col"],
    )
    fill_series = cast(
        bigframes.series.Series,
        mocks.create_dataframe(
            monkeypatch,
            session=series._session,
            data={"idx": [1, 2], "col": [0.0, 99.0]},
        ).set_index("idx")["col"],
    )
    original_block = series._block
    original_name = series.name
    original_index_names = list(series.index.names)

    inplace_result = series.fillna(fill_series, inplace=True)
    assert inplace_result is None
    assert series._block is not original_block
    assert series.name == original_name
    assert list(series.index.names) == original_index_names


def test_series_fillna_dict_inplace_false(monkeypatch: pytest.MonkeyPatch):
    series = cast(
        bigframes.series.Series,
        mocks.create_dataframe(monkeypatch, data={"col": [10.0, None]})["col"],
    )
    original_block = series._block
    original_name = series.name
    original_index_names = list(series.index.names)

    result = series.fillna({1: 99.0}, inplace=False)
    assert result is not None
    assert result is not series
    assert series._block is original_block
    assert result._block is not original_block
    assert series.name == original_name
    assert list(series.index.names) == original_index_names
    assert result.name == original_name
    assert list(result.index.names) == original_index_names


def test_series_fillna_dict_inplace_true(monkeypatch: pytest.MonkeyPatch):
    series = cast(
        bigframes.series.Series,
        mocks.create_dataframe(monkeypatch, data={"col": [10.0, None]})["col"],
    )
    original_block = series._block
    original_name = series.name
    original_index_names = list(series.index.names)

    inplace_result = series.fillna({1: 99.0}, inplace=True)
    assert inplace_result is None
    assert series._block is not original_block
    assert series.name == original_name
    assert list(series.index.names) == original_index_names


def test_series_fillna_positional_inplace_raises_type_error(
    monkeypatch: pytest.MonkeyPatch,
):
    series = cast(
        bigframes.series.Series,
        mocks.create_dataframe(
            monkeypatch, data={"idx": [1, 2], "col": [10.0, None]}
        ).set_index("idx")["col"],
    )
    with pytest.raises(TypeError):
        series.fillna(0.0, True)  # type: ignore[call-overload]
