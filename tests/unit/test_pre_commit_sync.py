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

import pathlib
import subprocess
import sys


def test_pre_commit_config_in_sync_with_noxfile():
    repo_root = pathlib.Path(__file__).resolve().parent.parent.parent
    sync_script = repo_root / "scripts" / "sync_pre_commit.py"
    result = subprocess.run(
        [sys.executable, str(sync_script), "--check"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f".pre-commit-config.yaml is out of sync with noxfile.py:\n"
        f"{result.stdout}\n{result.stderr}"
    )
