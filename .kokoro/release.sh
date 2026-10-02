#!/bin/bash
set -euo pipefail

cd "${KOKORO_ARTIFACTS_DIR}"

version="$(python3 -c 'import json; print(json.load(open("git/bigframes-internal/bigframes/.release-please-manifest.json"))["."])')"

cat > manifest.json <<EOF
{
  "publish_all": true,
  "registry_config": {
    "github_releases": {
      "name": "bigframes: v${version}",
      "tag": "v${version}",
      "is_latest": true,
      "is_draft": false,
      "is_prerelease": false
    }
  }
}
EOF
