#!/usr/bin/env bash
# Apply local patches to the CybORG_plus_plus submodule. Idempotent.
set -euo pipefail
cd "$(dirname "$0")/.."

SUBMODULE="CybORG_plus_plus"
shopt -s nullglob

for p in patches/*.patch; do
    name=$(basename "$p")
    if (cd "$SUBMODULE" && git apply --check "../$p") 2>/dev/null; then
        (cd "$SUBMODULE" && git apply "../$p")
        echo "  applied  $name"
    elif (cd "$SUBMODULE" && git apply --check --reverse "../$p") 2>/dev/null; then
        echo "  skipped  $name (already applied)"
    else
        echo "  ERROR    $name (does not apply, not already applied)" >&2
        exit 1
    fi
done
