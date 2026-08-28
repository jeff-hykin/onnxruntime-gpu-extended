#!/usr/bin/env bash
# Stage one ABI variant's wheels into a scratch dir and re-version them.
#
# rename_wheel.py reads ./wheels_input and rm -rf's ./renamed_wheels, both
# relative to CWD, so it must never be run from the repo root -- that would
# destroy the individually-verified wheel sets kept there.
#
# usage: stage_variant.sh <target_version> <scratch_dir> <wheel_source_dir>...
#
# The platform tag has to match what the *target device's pip* accepts, which is
# not the same as what its glibc supports. JetPack 5 ships pip 20.0.2, released
# before PEP 600, so it understands manylinux2014_aarch64 and nothing newer --
# a manylinux_2_31 tag naming its own glibc exactly would still be rejected.
# Override PLATFORM_TAG for those builds.
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
PLATFORM_RETAG="linux_aarch64=${PLATFORM_TAG:-manylinux_2_34_aarch64}" \
    python3 "$repo/rename_wheel.py"
