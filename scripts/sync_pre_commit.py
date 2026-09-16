#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Synchronize .pre-commit-config.yaml package versions with noxfile.py."""

from __future__ import annotations

import argparse
import ast
import configparser
import difflib
import pathlib
import re
import sys
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
NOXFILE_PATH = REPO_ROOT / "noxfile.py"
SETUP_PATH = REPO_ROOT / "setup.py"
MYPY_INI_PATH = REPO_ROOT / "mypy.ini"
PRE_COMMIT_CONFIG_PATH = REPO_ROOT / ".pre-commit-config.yaml"


def _eval_ast_node(node: ast.AST, env: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        raise KeyError(f"Unknown variable {node.id}")
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return [_eval_ast_node(elt, env) for elt in node.elts]
    if isinstance(node, ast.Dict):
        return {
            _eval_ast_node(k, env): _eval_ast_node(v, env)
            for k, v in zip(node.keys, node.values)
            if k is not None
        }
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _eval_ast_node(node.left, env) + _eval_ast_node(node.right, env)
    raise ValueError(f"Unsupported AST node: {ast.dump(node)}")


def parse_python_constants(path: pathlib.Path) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    env: dict[str, Any] = {}
    for stmt in tree.body:
        target_name = None
        value_node = None
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            if isinstance(stmt.targets[0], ast.Name):
                target_name = stmt.targets[0].id
                value_node = stmt.value
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            target_name = stmt.target.id
            value_node = stmt.value
        if target_name and value_node is not None:
            try:
                env[target_name] = _eval_ast_node(value_node, env)
            except Exception:
                pass
    return env


def get_ignored_mypy_packages(mypy_ini_path: pathlib.Path) -> set[str]:
    parser = configparser.ConfigParser()
    parser.read(mypy_ini_path, encoding="utf-8")
    ignored: set[str] = set()
    module_to_pkg = {
        "google.auth": "google-auth",
        "cloudpickle": "cloudpickle",
        "pydata_google_auth": "pydata-google-auth",
        "pytz": "pytz",
        "pyarrow": "pyarrow",
        "google.cloud.pubsub": "google-cloud-pubsub",
        "google.cloud.bigtable": "google-cloud-bigtable",
    }
    for section in parser.sections():
        if section.startswith("mypy-") and parser.getboolean(
            section, "ignore_missing_imports", fallback=False
        ):
            mod = section[len("mypy-") :].rstrip(".*")
            if mod in module_to_pkg:
                ignored.add(module_to_pkg[mod])
    return ignored


def normalize_req(req: str) -> tuple[str, str]:
    cleaned = re.sub(r"\s*([<>=!~]+)\s*", r"\1", req.strip())
    cleaned = re.sub(r"\s*,\s*", ",", cleaned)
    match = re.match(r"^([A-Za-z0-9_.-]+)", cleaned)
    pkg_name = match.group(1).lower().replace("_", "-") if match else cleaned.lower()
    return pkg_name, cleaned


def extract_version(spec: str, pkg_prefix: str) -> str:
    prefix = f"{pkg_prefix}=="
    if spec.startswith(prefix):
        return spec[len(prefix) :]
    raise ValueError(f"Expected {prefix}<version>, got {spec!r}")


def build_pre_commit_config() -> str:
    nox_consts = parse_python_constants(NOXFILE_PATH)
    setup_consts = parse_python_constants(SETUP_PATH)

    ruff_version = extract_version(nox_consts["RUFF_VERSION"], "ruff")
    mypy_version = extract_version(nox_consts["MYPY_VERSION"], "mypy")
    conv_version = nox_consts.get("CONVENTIONAL_PRE_COMMIT_VERSION", "v4.2.0")
    target_py = f"py{nox_consts['ALL_PYTHON'][0].replace('.', '')}"

    ignored_pkgs = get_ignored_mypy_packages(MYPY_INI_PATH)

    reqs_by_pkg: dict[str, str] = {}

    # 1. Start with setup.py dependencies and test extras
    setup_deps = list(setup_consts.get("dependencies", []))
    setup_extras = setup_consts.get("extras", {})
    for extra_name in ("tests", "polars", "scikit-learn", "anywidget"):
        setup_deps.extend(setup_extras.get(extra_name, []))

    for req in setup_deps:
        pkg_name, cleaned = normalize_req(req)
        if pkg_name not in ignored_pkgs:
            reqs_by_pkg[pkg_name] = cleaned

    # 2. Add noxfile mypy & test dependencies (overriding if more specific or pinned)
    nox_deps = (
        list(nox_consts.get("UNIT_TEST_STANDARD_DEPENDENCIES", []))
        + list(nox_consts.get("SYSTEM_TEST_STANDARD_DEPENDENCIES", []))
        + list(nox_consts.get("MYPY_TYPE_DEPENDENCIES", []))
    )
    for req in nox_deps:
        pkg_name, cleaned = normalize_req(req)
        if pkg_name in ignored_pkgs:
            continue
        existing = reqs_by_pkg.get(pkg_name)
        if existing is None or "==" in cleaned or "<=" in cleaned or ">=" in cleaned:
            reqs_by_pkg[pkg_name] = cleaned

    mypy_additional_deps = sorted(reqs_by_pkg.values(), key=lambda s: s.lower())
    deps_yaml = "\n".join(f"          - {dep}" for dep in mypy_additional_deps)

    return f"""# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# NOTE: Package versions in this file are synchronized with noxfile.py.
# Run `python scripts/sync_pre_commit.py` or `nox -s format` to update.

default_install_hook_types:
  - pre-commit
  - commit-msg

default_stages:
  - pre-commit

repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v{ruff_version}
    hooks:
      - id: ruff
        name: ruff lint
        args:
          - --fix
          - --select=I,F
          - --target-version={target_py}
          - --line-length=88
        files: ^(docs|bigframes|scripts|tests|third_party)/|^(noxfile|setup)\\.py$
      - id: ruff-format
        name: ruff format
        args:
          - --target-version={target_py}
          - --line-length=88
        files: ^(docs|bigframes|scripts|tests|third_party)/|^(noxfile|setup)\\.py$

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v{mypy_version}
    hooks:
      - id: mypy
        name: mypy
        args:
          - --check-untyped-defs
          - --explicit-package-bases
        files: ^(bigframes|tests/(system|unit))/
        exclude: ^third_party/
        additional_dependencies:
{deps_yaml}

  - repo: https://github.com/compilerla/conventional-pre-commit
    rev: {conv_version}
    hooks:
      - id: conventional-pre-commit
        name: conventional commits
        stages: [commit-msg]

  - repo: local
    hooks:
      - id: check-pre-commit-sync
        name: check pre-commit versions sync with noxfile.py
        entry: python3 scripts/sync_pre_commit.py --check
        language: python
        files: ^(noxfile\\.py|setup\\.py|mypy\\.ini|\\.pre-commit-config\\.yaml|scripts/sync_pre_commit\\.py)$
        pass_filenames: false
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if .pre-commit-config.yaml is in sync with noxfile.py without modifying it.",
    )
    args = parser.parse_args()

    expected = build_pre_commit_config()
    actual = (
        PRE_COMMIT_CONFIG_PATH.read_text(encoding="utf-8")
        if PRE_COMMIT_CONFIG_PATH.exists()
        else ""
    )

    if args.check:
        if actual != expected:
            diff = "".join(
                difflib.unified_diff(
                    actual.splitlines(keepends=True),
                    expected.splitlines(keepends=True),
                    fromfile=".pre-commit-config.yaml (actual)",
                    tofile=".pre-commit-config.yaml (expected from noxfile.py)",
                )
            )
            print(
                "ERROR: .pre-commit-config.yaml is out of sync with noxfile.py!\n"
                "Run `python scripts/sync_pre_commit.py` or `nox -s format` to synchronize.\n\n"
                f"{diff}",
                file=sys.stderr,
            )
            return 1
        print(".pre-commit-config.yaml is in sync with noxfile.py.")
        return 0

    if actual != expected:
        PRE_COMMIT_CONFIG_PATH.write_text(expected, encoding="utf-8")
        print("Updated .pre-commit-config.yaml to match noxfile.py.")
    else:
        print(".pre-commit-config.yaml is already up to date.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
