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

import inspect
import sys
from unittest.mock import patch

import bigframes_vendored.constants as constants
import pytest
from google.cloud import bigquery

from bigframes.functions import _utils, function_typing


@pytest.mark.parametrize(
    ("input_location", "expected_cf_region"),
    [
        ("us", "us-central1"),
        ("eu", "europe-west1"),
        ("US-east4", "us-east4"),
    ],
)
def test_gcf_location_from_bq_location(input_location, expected_cf_region):
    """Tests getting cloud function locations for various BigQuery locations."""
    gcf_location = _utils.gcf_location_from_bq_location(input_location)

    assert gcf_location == expected_cf_region


@patch("bigframes.functions._utils.numpy.__version__", "1.24.4")
@patch("bigframes.functions._utils.pyarrow.__version__", "14.0.1")
@patch("bigframes.functions._utils.pandas.__version__", "2.0.3")
@patch("bigframes.functions._utils.cloudpickle.__version__", "2.2.1")
def test_get_updated_package_requirements_is_row_processor_with_versions():
    """Tests with is_row_processor=True and specific versions."""
    expected = [
        "cloudpickle==2.2.1",
        "numpy==1.24.4",
        "pandas==2.0.3",
        "pyarrow==14.0.1",
    ]
    result = _utils.get_updated_package_requirements(is_row_processor=True)

    assert result == expected


@patch("bigframes.functions._utils.warnings.warn")
@patch("bigframes.functions._utils.cloudpickle.__version__", "2.2.1")
def test_get_updated_package_requirements_ignore_version(mock_warn):
    """
    Tests with is_row_processor=True and ignore_package_version=True.
    Should add packages without versions and raise a warning.
    """
    expected = ["cloudpickle==2.2.1", "numpy", "pandas", "pyarrow"]
    result = _utils.get_updated_package_requirements(
        is_row_processor=True, ignore_package_version=True
    )

    assert result == expected
    # Verify that a warning was issued.
    mock_warn.assert_called_once()


@patch("bigframes.functions._utils.numpy.__version__", "1.24.4")
@patch("bigframes.functions._utils.pyarrow.__version__", "14.0.1")
@patch("bigframes.functions._utils.pandas.__version__", "2.0.3")
def test_get_updated_package_requirements_capture_references_false():
    """
    Tests with capture_references=False.
    Should not add cloudpickle but should add others if requested.
    """
    # Case 1: Only capture_references=False.
    result_1 = _utils.get_updated_package_requirements(capture_references=False)

    assert len(result_1) == 0

    # Case 2: capture_references=False but is_row_processor=True.
    expected_2 = ["numpy==1.24.4", "pandas==2.0.3", "pyarrow==14.0.1"]
    result_2 = _utils.get_updated_package_requirements(
        is_row_processor=True, capture_references=False
    )

    assert result_2 == expected_2


@patch("bigframes.functions._utils.numpy.__version__", "1.24.4")
@patch("bigframes.functions._utils.pyarrow.__version__", "14.0.1")
@patch("bigframes.functions._utils.pandas.__version__", "2.0.3")
@patch("bigframes.functions._utils.cloudpickle.__version__", "2.2.1")
def test_get_updated_package_requirements_non_overlapping_packages():
    """Tests providing an initial list of packages that do not overlap."""
    initial_packages = ["scikit-learn==1.3.0", "xgboost"]
    expected = [
        "cloudpickle==2.2.1",
        "numpy==1.24.4",
        "pandas==2.0.3",
        "pyarrow==14.0.1",
        "scikit-learn==1.3.0",
        "xgboost",
    ]
    result = _utils.get_updated_package_requirements(
        package_requirements=initial_packages, is_row_processor=True
    )

    assert result == expected


@patch("bigframes.functions._utils.numpy.__version__", "1.24.4")
@patch("bigframes.functions._utils.pyarrow.__version__", "14.0.1")
@patch("bigframes.functions._utils.pandas.__version__", "2.0.3")
@patch("bigframes.functions._utils.cloudpickle.__version__", "2.2.1")
def test_get_updated_package_requirements_overlapping_packages():
    """Tests that packages are not added if they already exist."""
    # The function should respect the pre-existing pandas version.
    initial_packages = ["pandas==1.5.3", "numpy"]
    expected = [
        "cloudpickle==2.2.1",
        "numpy",
        "pandas==1.5.3",
        "pyarrow==14.0.1",
    ]
    result = _utils.get_updated_package_requirements(
        package_requirements=initial_packages, is_row_processor=True
    )

    assert result == expected


@patch("bigframes.functions._utils.cloudpickle.__version__", "2.2.1")
def test_get_updated_package_requirements_with_existing_cloudpickle():
    """Tests that cloudpickle is not added if it already exists."""
    initial_packages = ["cloudpickle==2.0.0"]
    expected = ["cloudpickle==2.0.0"]
    result = _utils.get_updated_package_requirements(
        package_requirements=initial_packages
    )

    assert result == expected


# Dynamically generate expected python versions for the test
_major = sys.version_info.major
_minor = sys.version_info.minor
_compat_version = f"python{_major}{_minor}"
_standard_version = f"python-{_major}.{_minor}"


@pytest.mark.parametrize(
    "is_compat, expected_version",
    [
        (True, _compat_version),
        (False, _standard_version),
    ],
)
def test_get_python_version(is_compat, expected_version):
    """Tests the python version for both standard and compat modes."""
    result = _utils.get_python_version(is_compat=is_compat)
    assert result == expected_version


def test_package_existed_helper():
    """Tests the _package_existed helper function directly."""
    reqs = ["pandas==1.0", "numpy", "scikit-learn>=1.2.0"]

    # Exact match
    assert _utils._package_existed(reqs, "pandas==1.0")
    # Different version
    assert _utils._package_existed(reqs, "pandas==2.0")
    # No version specified
    assert _utils._package_existed(reqs, "numpy")
    # Not in list
    assert not _utils._package_existed(reqs, "xgboost")
    # Empty list
    assert not _utils._package_existed([], "pandas")


# Helper functions for signature inspection tests
def _func_one_arg_annotated(x: int) -> int:
    """A function with one annotated arg and an annotated return type."""
    return x


def _func_one_arg_unannotated(x):
    """A function with one unannotated arg and no return type annotation."""
    return x


def _func_two_args_annotated(x: int, y: str):
    """A function with two annotated args and no return type annotation."""
    return f"{x}{y}"


def _func_two_args_unannotated(x, y):
    """A function with two unannotated args and no return type annotation."""
    return f"{x}{y}"


def test_has_conflict_input_type_too_few_inputs():
    """Tests conflict when there are fewer input types than parameters."""
    signature = inspect.signature(_func_one_arg_annotated)
    assert _utils.has_conflict_input_type(signature, input_types=[])


def test_has_conflict_input_type_too_many_inputs():
    """Tests conflict when there are more input types than parameters."""
    signature = inspect.signature(_func_one_arg_annotated)
    assert _utils.has_conflict_input_type(signature, input_types=[int, str])


def test_has_conflict_input_type_type_mismatch():
    """Tests has_conflict_input_type with a conflicting type annotation."""
    signature = inspect.signature(_func_two_args_annotated)

    # The second type (bool) conflicts with the annotation (str).
    assert _utils.has_conflict_input_type(signature, input_types=[int, bool])


def test_has_conflict_input_type_no_conflict_annotated():
    """Tests that a matching, annotated signature is compatible."""
    signature = inspect.signature(_func_two_args_annotated)
    assert not _utils.has_conflict_input_type(signature, input_types=[int, str])


def test_has_conflict_input_type_no_conflict_unannotated():
    """Tests that a signature with no annotations is always compatible."""
    signature = inspect.signature(_func_two_args_unannotated)
    assert not _utils.has_conflict_input_type(signature, input_types=[int, float])


def test_has_conflict_output_type_no_conflict():
    """Tests has_conflict_output_type with type annotation."""
    signature = inspect.signature(_func_one_arg_annotated)

    assert _utils.has_conflict_output_type(signature, output_type=float)
    assert not _utils.has_conflict_output_type(signature, output_type=int)


def test_has_conflict_output_type_no_annotation():
    """Tests has_conflict_output_type without type annotation."""
    signature = inspect.signature(_func_one_arg_unannotated)

    assert not _utils.has_conflict_output_type(signature, output_type=int)
    assert not _utils.has_conflict_output_type(signature, output_type=float)


@pytest.mark.parametrize(
    ["metadata_options", "metadata_string"],
    (
        pytest.param(
            {},
            '{"value": {}}',
            id="empty",
        ),
        pytest.param(
            {"python_output_type": None},
            '{"value": {}}',
            id="None",
        ),
        pytest.param(
            {"python_output_type": list[bool]},
            '{"value": {"python_array_output_type": "bool"}}',
            id="list-bool",
        ),
        pytest.param(
            {"python_output_type": list[float]},
            '{"value": {"python_array_output_type": "float"}}',
            id="list-float",
        ),
        pytest.param(
            {"python_output_type": list[int]},
            '{"value": {"python_array_output_type": "int"}}',
            id="list-int",
        ),
        pytest.param(
            {"python_output_type": list[str]},
            '{"value": {"python_array_output_type": "str"}}',
            id="list-str",
        ),
    ),
)
def test_get_bigframes_metadata(metadata_options, metadata_string):
    assert _utils.get_bigframes_metadata(**metadata_options) == metadata_string


@pytest.mark.parametrize(
    ["output_type"],
    (
        pytest.param(bool),
        pytest.param(bytes),
        pytest.param(float),
        pytest.param(int),
        pytest.param(str),
        pytest.param(list),
        pytest.param(list[bytes], id="list-bytes"),
    ),
)
def test_get_bigframes_metadata_array_type_not_serializable(output_type):
    with pytest.raises(ValueError) as context:
        _utils.get_bigframes_metadata(python_output_type=output_type)

    assert str(context.value) == (
        f"python_output_type {output_type} is not serializable. {constants.FEEDBACK_LINK}"
    )


@pytest.mark.parametrize(
    ["metadata_string", "python_output_type"],
    (
        pytest.param(
            None,
            None,
            id="None",
        ),
        pytest.param(
            "",
            None,
            id="empty",
        ),
        pytest.param(
            "{}",
            None,
            id="empty-dict",
        ),
        pytest.param(
            '{"value": {}}',
            None,
            id="empty-value",
        ),
        pytest.param(
            '{"value": {"python_array_output_type": "bool"}}',
            list[bool],
            id="list-bool",
        ),
        pytest.param(
            '{"value": {"python_array_output_type": "float"}}',
            list[float],
            id="list-float",
        ),
        pytest.param(
            '{"value": {"python_array_output_type": "int"}}',
            list[int],
            id="list-int",
        ),
        pytest.param(
            '{"value": {"python_array_output_type": "str"}}',
            list[str],
            id="list-str",
        ),
    ),
)
def test_get_python_output_type_from_bigframes_metadata(
    metadata_string, python_output_type
):
    assert (
        _utils.get_python_output_type_from_bigframes_metadata(metadata_string)
        == python_output_type
    )


def test_metadata_roundtrip_supported_array_types():
    for array_of in function_typing.RF_SUPPORTED_ARRAY_OUTPUT_PYTHON_TYPES:
        ser = _utils.get_bigframes_metadata(python_output_type=list[array_of])  # type: ignore
        deser = _utils.get_python_output_type_from_bigframes_metadata(ser)

        assert deser == list[array_of]  # type: ignore


@pytest.mark.parametrize(
    "valid_reqs",
    [
        ["pandas"],
        ["pandas>=2.0.0,<3.0"],
        ["scikit-learn==1.3.0", "xgboost"],
        ['requests[security]>=2.28; python_version >= "3.10"'],
    ],
)
def test_validate_package_requirements_accepts_valid_pep508(valid_reqs):
    assert _utils._validate_package_requirements(valid_reqs) is None


@pytest.mark.parametrize(
    ("bad_req", "match"),
    [
        pytest.param("", "empty string", id="empty"),
        pytest.param(
            "--extra-index-url https://example.com", "pip options/flags", id="pip_flag"
        ),
        pytest.param(
            "  -r requirements.txt", "pip options/flags", id="leading_space_flag"
        ),
        pytest.param(
            "pandas\n--extra-index-url https://example.com",
            "line breaks",
            id="newline",
        ),
        pytest.param("pandas\r\nnumpy", "line breaks", id="crlf"),
        pytest.param(
            "pandas; extra == 'a\u2028--extra-index-url=https://example.com\u2028b'",
            "line breaks",
            id="unicode_line_separator",
        ),
        pytest.param(
            "pandas; extra == 'a\u2029--extra-index-url=https://example.com\u2029b'",
            "line breaks",
            id="unicode_paragraph_separator",
        ),
        pytest.param(
            "pandas; extra == 'a\x0b--extra-index-url=https://example.com\x0bb'",
            "line breaks",
            id="vertical_tab",
        ),
        pytest.param(
            "pandas; extra == 'a\x85--extra-index-url=https://example.com\x85b'",
            "line breaks",
            id="nel",
        ),
        pytest.param(
            "pandas; extra == 'a\x00b'",
            "null bytes",
            id="null_byte",
        ),
        pytest.param(
            "pandas; extra == 'a\\'",
            "Invalid package requirement",
            id="unterminated_marker_escape",
        ),
        pytest.param(
            "requests @ https://example.com/pkg.whl",
            "direct URL",
            id="direct_url",
        ),
        pytest.param(
            "requests @ https://example.com/a\n--extra-index-url=https://example.com",
            "line breaks",
            id="direct_url_with_newline",
        ),
        pytest.param(
            "pandas''' ; DROP TABLE t; --",
            "Invalid package requirement",
            id="sql_payload",
        ),
    ],
)
def test_validate_package_requirements_rejects_unsafe_entries(bad_req, match):
    with pytest.raises(ValueError, match=match):
        _utils._validate_package_requirements([bad_req])


@pytest.mark.parametrize("bare_input", ["pandas", b"pandas"])
def test_validate_package_requirements_rejects_bare_string_or_bytes(bare_input):
    with pytest.raises(TypeError, match="sequence of strings"):
        _utils._validate_package_requirements(bare_input)  # type: ignore[arg-type]


def test_get_updated_package_requirements_validates_when_capture_references_false():
    with pytest.raises(ValueError, match="pip options/flags"):
        _utils.get_updated_package_requirements(
            package_requirements=["--extra-index-url https://example.com"],
            is_row_processor=False,
            capture_references=False,
        )


def test_routine_ref_to_string_for_query_escapes_each_segment():
    routine_ref = bigquery.DatasetReference("my-proj", "my`ds").routine("my_fn")

    assert (
        _utils.routine_ref_to_string_for_query(routine_ref)
        == r"`my-proj`.`my\`ds`.`my_fn`"
    )
