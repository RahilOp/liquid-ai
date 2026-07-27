#!/usr/bin/env bash
# Build main.pdf. Regenerates figures first, then runs tectonic (which handles
# the bibtex passes itself).
set -euo pipefail
cd "$(dirname "$0")"

echo "==> figures"
python3 figures/make_figures.py >/dev/null && echo "    ok"

echo "==> tectonic"
.tools/tectonic -X compile main.tex --keep-logs 2>&1 | grep -viE "^note: downloading" || true

echo "==> $(pwd)/main.pdf"
