#!/bin/bash
set -euo pipefail

cd "${KOKORO_ARTIFACTS_DIR}"

version="$(python3 -c 'import json; print(json.load(open("git/bigframes-internal/bigframes/.release-please-manifest.json"))["."])')"

cat > manifest.json <<EOF
{
  "publish_all": true
}
EOF
