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

"""Data models for BigQuery code generator.

`BQ*` models the Substrait YAML extension structure of BigQuery SQL functions,
while `BigFrames*` models the Jinja template outputs.

BQ* Class Relations:
====================
+-------------------+
|     BQModule      |
+-------------------+
          |
          | functions: list[BQFunc]
          v
+-------------------+
|      BQFunc       |
+-------------------+
          |
          | impls: list[BQFuncImpl]
          v
+-------------------+
|    BQFuncImpl     |
+-------------------+
          |
          | args: list[BQFuncArg]
          v
+-------------------+
|     BQFuncArg     |
+-------------------+

BigFrames* Class Relations:
=================================
    +--------------------------------+
    |            Accessor            |<---+ children: list[Accessor]
    +--------------------------------+----+ (nested namespace hierarchy)
                    |
                    | functions: list[BigFramesFunc]
                    v
    +--------------------------------+
    |         BigFramesFunc          |
    +--------------------------------+
                    |
                    | args: list[BigFramesFuncArg]
                    v
    +--------------------------------+
    |        BigFramesFuncArg        |
    +--------------------------------+

    ----------------------------------

    +--------------------------------+
    |          BigFramesOp           |  (Standalone data model for op defs)
    +--------------------------------+
"""

from __future__ import annotations

import dataclasses
import pathlib

from . import constants


@dataclasses.dataclass(frozen=True)
class BQFuncArg:
    """
    Represents an argument of a SQL function loaded from a yaml file.
    """

    name: str
    value: str  # The type of the arg
    optional: bool
    keyword_only: bool


@dataclasses.dataclass(frozen=True)
class BQFuncImpl:
    """
    Represents an implementation (i.e. signature) for some SQL function loaded
    from a yaml file.
    """

    args: tuple[BQFuncArg, ...]
    return_type: str

    @property
    def requires_generic_types(self) -> bool:
        if "any1" in self.return_type:
            return True

        return any("any1" in arg.value for arg in self.args)


@dataclasses.dataclass(frozen=True)
class BQFunc:
    """
    Represents a SQL function loaded from a yaml file.
    """

    name: str
    description: str
    impls: tuple[BQFuncImpl, ...]
    series_accessor_arg: str | None

    @property
    def op_base_name(self) -> str:
        return self.name.split(".")[-1]


@dataclasses.dataclass(frozen=True)
class BQModule:
    """
    Represents the data loaded from a yaml file with SQL functions info.
    """

    yaml_file: pathlib.Path
    functions: tuple[BQFunc, ...]

    @property
    def module_path(self) -> pathlib.Path:
        return self.yaml_file.relative_to(constants.DATA_DIR).with_suffix("")

    @property
    def namespace(self) -> tuple[str, ...]:
        parts = self.module_path.parts
        if "global_namespace" in parts:
            return ()
        return parts

    @property
    def is_global(self) -> bool:
        return "global_namespace" in self.module_path.parts


@dataclasses.dataclass(frozen=True)
class BigFramesOp:
    """
    Represents a BigFrames GoogleScalarOp impl to be defined in the code base.
    """

    internal_name: str
    sql_name: str
    arg_specs: str
    signature: str
    signature_definition: str | None


@dataclasses.dataclass(frozen=True)
class BigFramesFuncArg:
    """
    Represents an argument of a BigFrames BigQuery function to be defined in the code base.
    """

    name: str
    types: frozenset[str]
    optional: bool
    keyword_only: bool

    @property
    def type_hint(self) -> str:
        types = [constants.PY_TYPE_MAP.get(t, "Any") for t in sorted(self.types)] + [
            "Literal[sentinels.Sentinel.ARGUMENT_DEFAULT]"
        ]

        if len(types) > 1:
            return "Union[" + ", ".join(sorted(set(types))) + "]"

        return types[0]

    @property
    def default(self) -> str | None:
        if self.optional:
            return "sentinels.Sentinel.ARGUMENT_DEFAULT"
        return None

    @property
    def test_value(self) -> str:
        name = self.name
        types = self.types

        if name == "year":
            return "2025"
        if name == "month":
            return "1"
        if name == "day":
            return "1"
        if name == "hour":
            return "12"
        if name == "minute":
            return "0"
        if name == "second":
            return "0"
        if name in ("granularity", "date_part", "part"):
            return '"DAY"'
        if name in ("time_zone", "time_zone_expression"):
            return '"UTC"'
        if "format" in name:
            return '"%Y-%m-%d"'
        if "date_string" in name:
            return '"2025-01-01"'
        if "datetime_string" in name:
            return '"2025-01-01 12:00:00"'
        if name == "delimiter":
            return '", "'
        if name in ("depth", "end_offset"):
            return "1"
        if name == "keyset":
            return 'b"keyset"'
        if name == "ciphertext":
            return 'b"ciphertext"'
        if name == "plaintext":
            return '"plaintext"'
        if name == "additional_data":
            if "binary" in types and "string" not in types:
                return 'b"additional_data"'
            return '"additional_data"'
        if name == "array_to_flatten":
            return "[[1, 2], [3, 4]]"
        if "date" in types:
            return "dt.date(2025, 1, 1)"
        if "time" in types:
            return "dt.time(12, 0, 0)"
        if "datetime" in types or "timestamp" in types:
            return "dt.datetime(2025, 1, 1, 12, 0, 0, tzinfo=dt.timezone.utc)"
        if "i64" in types or "i32" in types or "int64" in types:
            return "1"
        if "fp64" in types or "float64" in types:
            return "1.0"
        if "decimal<38,9>" in types or "decimal<76,38>" in types:
            return 'decimal.Decimal("1.5")'
        if "bool" in types or "boolean" in types:
            return "True"
        if "binary" in types and "string" not in types:
            return 'b"test"'
        if "string" in types:
            return '"test"'
        if "list<string>" in types:
            return '["a", "b"]'
        if "list<binary>" in types:
            return '[b"a", b"b"]'
        if any(t.startswith("list<") for t in types):
            return "[1, 2]"
        if "any1" in types:
            return "1"
        return f'bpd.col("{self.name}")'

    @property
    def test_is_literal(self) -> bool:
        return not self.test_value.startswith("bpd.col(")


@dataclasses.dataclass
class BigFramesFuncArgBuilder:
    name: str
    types: set[str]
    optional: bool
    keyword_only: bool

    def build(self) -> BigFramesFuncArg:
        return BigFramesFuncArg(
            name=self.name,
            types=frozenset(self.types),
            optional=self.optional,
            keyword_only=self.keyword_only,
        )


@dataclasses.dataclass(frozen=True)
class BigFramesFunc:
    """
    Represents a BigFrames BigQuery function to be defined in the codebase.
    """

    name: str
    op_name: str
    description: str
    args: tuple[BigFramesFuncArg, ...]
    series_accessor_arg: str | None
    import_module: str | None = None


@dataclasses.dataclass
class Accessor:
    """
    Represents the accessor extensions to be defined for pandas and BigFrames.
    It consists of multiple functions bundled under the different namespaces.

    This class is designed to be mutable because it has a recursive data structure.
    Mutability makes it easier to build the data structure trees from the top.
    """

    class_name: str
    bigframes_class_name: str
    pandas_class_name: str
    is_root: bool
    description: str
    children: list[Accessor]
    functions: list[BigFramesFunc]
    prop_name: str | None = None
