#!/usr/bin/env bash
# Stage one ABI variant's wheels into a scratch dir and re-version them.
#
# rename_wheel.py reads ./wheels_input and rm -rf's ./renamed_wheels, both
# relative to CWD, so it must never be run from the repo root -- that would
# destroy the individually-verified wheel sets kept there.
#
# usage: stage_variant.sh <target_version> <scratch_dir> <wheel_source_dir>...
set -euo pipefail

target_version="$1"; shift
scratch="$1"; shift

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

rm -rf "$scratch"
mkdir -p "$scratch/wheels_input"

for source_dir in "$@"; do
    find "$source_dir" -name '*.whl' -exec cp {} "$scratch/wheels_input/" \;
done

echo "staged $(ls "$scratch/wheels_input" | wc -l | tr -d ' ') wheel(s) for $target_version"
ls "$scratch/wheels_input"

cd "$scratch"
TARGET_VERSION="$target_version" \
PLATFORM_RETAG="linux_aarch64=manylinux_2_34_aarch64" \
    python3 "$repo/rename_wheel.py"
