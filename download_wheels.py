#!/usr/bin/env python3
"""
Download existing onnxruntime-gpu wheels from PyPI and collect locally-built ones.

Populates wheels_input/ with all wheels that will be renamed and uploaded
as onnxruntime-gpu-extended.

Sources:
  - PyPI: onnxruntime-gpu (upstream: Linux x86_64 + Windows CUDA wheels)
  - Local: result/*.whl (Nix-built Jetson aarch64 CUDA wheels not on PyPI)
"""

import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path

# Which PyPI package to download from
PYPI_PACKAGE = "onnxruntime-gpu"

# Only download wheels matching these patterns (None = download all)
# Set to a list of substrings to filter, e.g. ["cp312", "cp313"]
PYTHON_VERSION_FILTER = None  # download all available versions

OUTPUT_DIR = Path("wheels_input")


def get_pypi_info(package_name):
    """Fetch package info from PyPI JSON API."""
    url = f"https://pypi.org/pypi/{package_name}/json"
    print(f"Fetching {url} ...")
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def download_file(url, dest):
    """Download a file with progress."""
    print(f"  Downloading {dest.name} ...", end=" ", flush=True)
    urllib.request.urlretrieve(url, dest)
    size_mb = dest.stat().st_size / (1024 * 1024)
    print(f"({size_mb:.1f} MB)")


def download_pypi_wheels(package_name, output_dir):
    """Download all wheels for the latest version of a PyPI package."""
    info = get_pypi_info(package_name)
    # Pin to a specific upstream release when requested (keeps the check,
    # download, and rename steps consistent within a single publish run);
    # otherwise take whatever PyPI reports as latest.
    version = os.environ.get("UPSTREAM_VERSION") or info["info"]["version"]
    print(f"\nUsing {package_name} version: {version}")

    files = info["releases"].get(version, [])
    wheels = [f for f in files if f["filename"].endswith(".whl")]

    if not wheels:
        print(f"  No wheels found for {package_name} {version}")
        return []

    print(f"  Found {len(wheels)} wheel(s)")

    downloaded = []
    for whl in wheels:
        filename = whl["filename"]

        # Apply filter if set
        if PYTHON_VERSION_FILTER:
            if not any(pv in filename for pv in PYTHON_VERSION_FILTER):
                print(f"  Skipping {filename} (filtered)")
                continue

        dest = output_dir / filename
        if dest.exists():
            print(f"  Already exists: {filename}")
        else:
            download_file(whl["url"], dest)
        downloaded.append(dest)

    return downloaded


def collect_local_wheels(output_dir):
    """Copy locally-built wheels from result/ and result-*/ into the output directory."""
    # Collect from result/ and result-*/ (created by ./run/build)
    result_dirs = [Path("result")] + sorted(Path(".").glob("result-*"))
    local_wheels = []
    for d in result_dirs:
        if d.exists():
            local_wheels.extend(d.glob("*.whl"))

    if not local_wheels:
        print("\nNo local wheels found (run 'nix build' or './run/build' to create them)")
        return []

    print(f"\nFound {len(local_wheels)} local wheel(s)")
    collected = []
    for whl in local_wheels:
        dest = output_dir / whl.name
        # Remove existing file first (may be read-only from previous Nix copy)
        if dest.exists():
            dest.chmod(0o644)
            dest.unlink()
        shutil.copy2(whl, dest)
        # Ensure writable so future runs can overwrite
        dest.chmod(0o644)
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"  Copied {whl.name} ({size_mb:.1f} MB)")
        collected.append(dest)

    return collected


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_wheels = []

    # 1. Download from PyPI
    print("=" * 60)
    print(f"Downloading wheels from PyPI ({PYPI_PACKAGE})")
    print("=" * 60)
    all_wheels.extend(download_pypi_wheels(PYPI_PACKAGE, OUTPUT_DIR))

    # 2. Collect local builds
    print("\n" + "=" * 60)
    print("Collecting locally-built wheels")
    print("=" * 60)
    all_wheels.extend(collect_local_wheels(OUTPUT_DIR))

    # Summary
    print("\n" + "=" * 60)
    print(f"Total: {len(all_wheels)} wheel(s) in {OUTPUT_DIR}/")
    print("=" * 60)
    for whl in sorted(OUTPUT_DIR.glob("*.whl")):
        size_mb = whl.stat().st_size / (1024 * 1024)
        print(f"  {whl.name}  ({size_mb:.1f} MB)")

    if all_wheels:
        print(f"\nNext step: python rename_wheel.py")


if __name__ == "__main__":
    main()
