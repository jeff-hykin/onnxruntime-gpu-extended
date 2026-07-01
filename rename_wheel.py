#!/usr/bin/env python3
"""
Rename wheel distribution names to 'onnxruntime_gpu_extended' with a normalized
version.

Takes wheels with distribution name 'onnxruntime' or 'onnxruntime_gpu' and
renames them to 'onnxruntime_gpu_extended', keeping the internal 'onnxruntime'
package intact so `import onnxruntime` still works.

The version is normalized to match pyproject.toml so all wheels can be
uploaded together to PyPI.
"""

import os
import re
import sys
import hashlib
import base64
import zipfile
import shutil
import tempfile
from pathlib import Path

NEW_DIST_NAME = "onnxruntime_gpu_extended"

# Distribution names we know how to rename
KNOWN_DIST_NAMES = {"onnxruntime", "onnxruntime_gpu"}


def get_target_version():
    """Resolve the onnxruntime-gpu-extended version all wheels normalize to.

    Order of precedence:
      1. TARGET_VERSION env var (run/publish exports this so build + rename agree)
      2. Derived from upstream onnxruntime-gpu on PyPI (auto-tracks new releases)
      3. pyproject.toml (last-resort offline fallback)
    """
    # 1. Explicit override.
    env = os.environ.get("TARGET_VERSION")
    if env:
        return env
    # 2. Track the latest upstream release.
    try:
        sys.path.insert(0, str(Path(__file__).parent / "run"))
        from upstream_version import target_version
        return target_version()
    except Exception as e:
        print(f"WARNING: could not derive version from upstream ({e}); "
              f"falling back to pyproject.toml")
    # 3. Whatever is pinned in pyproject.toml.
    pyproject = Path(__file__).parent / "pyproject.toml"
    if not pyproject.exists():
        print("ERROR: pyproject.toml not found")
        sys.exit(1)
    text = pyproject.read_text()
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not match:
        print("ERROR: Could not find version in pyproject.toml")
        sys.exit(1)
    return match.group(1)


def hash_file(path):
    """Compute SHA256 hash in RECORD format (url-safe base64, no padding)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    digest = base64.urlsafe_b64encode(h.digest()).rstrip(b"=").decode("ascii")
    return f"sha256={digest}"


def detect_dist_name(wheel_path):
    """Detect the distribution name and version from a wheel filename.

    Returns (dist_name, version, tags) where tags is e.g.
    'cp312-cp312-manylinux2014_aarch64'.
    """
    stem = Path(wheel_path).stem
    parts = stem.split("-")
    # Find where version starts (first part that begins with a digit)
    for i, part in enumerate(parts):
        if part and part[0].isdigit():
            dist_name = "_".join(parts[:i])
            # version is the next part, tags are the rest
            version = parts[i]
            tags = "-".join(parts[i + 1:])
            return dist_name, version, tags
    return parts[0], parts[1] if len(parts) > 1 else "0", "-".join(parts[2:])


def rename_wheel(wheel_path, output_dir, target_version):
    wheel_path = Path(wheel_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    old_dist_name, old_version, tags = detect_dist_name(wheel_path)

    if old_dist_name not in KNOWN_DIST_NAMES and old_dist_name != NEW_DIST_NAME:
        print(f"WARNING: Unknown distribution name '{old_dist_name}' in {wheel_path.name}, renaming anyway")

    new_wheel_name = f"{NEW_DIST_NAME}-{target_version}-{tags}.whl"

    # Unpack
    tmpdir = tempfile.mkdtemp()
    with zipfile.ZipFile(wheel_path, "r") as zf:
        zf.extractall(tmpdir)

    # Find dist-info directory
    old_info_path = None
    for entry in Path(tmpdir).iterdir():
        if entry.is_dir() and entry.name.endswith(".dist-info"):
            # Match by distribution name prefix
            if entry.name.startswith(f"{old_dist_name}-"):
                old_info_path = entry
                break

    if old_info_path is None:
        # Try any .dist-info
        for entry in Path(tmpdir).iterdir():
            if entry.is_dir() and entry.name.endswith(".dist-info"):
                old_info_path = entry
                print(f"  Using dist-info: {entry.name}")
                break

    if old_info_path is None:
        print(f"ERROR: No .dist-info found in {wheel_path.name}")
        shutil.rmtree(tmpdir)
        return None

    # New dist-info name with normalized version
    new_info_name = f"{NEW_DIST_NAME}-{target_version}.dist-info"
    new_info_path = old_info_path.parent / new_info_name
    old_info_path.rename(new_info_path)

    # Update METADATA
    metadata_path = new_info_path / "METADATA"
    if metadata_path.exists():
        metadata = metadata_path.read_text()
        # Replace Name
        metadata = re.sub(
            r"^Name: .+$",
            f"Name: {NEW_DIST_NAME.replace('_', '-')}",
            metadata,
            count=1,
            flags=re.MULTILINE,
        )
        # Replace Version
        metadata = re.sub(
            r"^Version: .+$",
            f"Version: {target_version}",
            metadata,
            count=1,
            flags=re.MULTILINE,
        )
        metadata_path.write_text(metadata)

    # Rebuild RECORD
    record_path = new_info_path / "RECORD"
    record_lines = []

    for root, dirs, files in os.walk(tmpdir):
        for fname in sorted(files):
            fpath = Path(root) / fname
            arcname = str(fpath.relative_to(tmpdir))

            if fpath == record_path:
                continue

            file_hash = hash_file(fpath)
            file_size = fpath.stat().st_size
            record_lines.append(f"{arcname},{file_hash},{file_size}")

    record_lines.append(f"{new_info_name}/RECORD,,")
    record_path.write_text("\n".join(sorted(record_lines)) + "\n")

    # Repack
    new_wheel_path = output_dir / new_wheel_name
    with zipfile.ZipFile(new_wheel_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(tmpdir):
            for fname in sorted(files):
                fpath = Path(root) / fname
                arcname = str(fpath.relative_to(tmpdir))
                zf.write(fpath, arcname)

    shutil.rmtree(tmpdir)

    version_note = "" if old_version == target_version else f" (version {old_version} -> {target_version})"
    print(f"  {wheel_path.name} -> {new_wheel_name}{version_note}")
    return new_wheel_path


def main():
    target_version = get_target_version()
    print(f"Target version: {target_version}\n")

    input_dir = Path("wheels_input")

    if not input_dir.exists():
        print(f"ERROR: {input_dir}/ directory not found.")
        print("Run download_wheels.py first to populate it.")
        sys.exit(1)

    wheels = list(input_dir.glob("*.whl"))
    if not wheels:
        print(f"ERROR: No .whl files found in {input_dir}/")
        sys.exit(1)

    output_dir = Path("renamed_wheels")
    if output_dir.exists():
        shutil.rmtree(output_dir)

    print(f"Renaming {len(wheels)} wheel(s)...\n")
    for wheel in sorted(wheels):
        rename_wheel(wheel, output_dir, target_version)

    print(f"\nOutput in: {output_dir}/")
    for whl in sorted(output_dir.glob("*.whl")):
        size_mb = whl.stat().st_size / (1024 * 1024)
        print(f"  {whl.name}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
