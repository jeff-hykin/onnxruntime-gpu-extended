#!/usr/bin/env python3
"""Report the CUDA/cuDNN sonames each wheel's CUDA provider actually links.

The wheel tag cannot express cuDNN 8 vs 9, so a wheel built against the wrong
L4T base image is indistinguishable by filename. This reads DT_NEEDED out of
libonnxruntime_providers_cuda.so, which is the only real evidence.

usage: check_wheel_abi.py <expected_cudnn_major> <wheel>...
"""

import re
import sys
import zipfile
import tempfile
from pathlib import Path

from elftools.elf.elffile import ELFFile

PROVIDER = "libonnxruntime_providers_cuda.so"


def needed_sonames(wheel_path):
    with zipfile.ZipFile(wheel_path) as zf:
        member = next(
            (n for n in zf.namelist() if n.endswith(PROVIDER)),
            None,
        )
        if member is None:
            return None
        with tempfile.TemporaryDirectory() as tmp:
            extracted = Path(zf.extract(member, tmp))
            with open(extracted, "rb") as handle:
                elf = ELFFile(handle)
                dynamic = elf.get_section_by_name(".dynamic")
                return sorted(
                    tag.needed
                    for tag in dynamic.iter_tags()
                    if tag.entry.d_tag == "DT_NEEDED"
                )


def main():
    expected = sys.argv[1]
    failures = 0

    for wheel in sorted(sys.argv[2:]):
        sonames = needed_sonames(wheel)
        name = Path(wheel).name

        if sonames is None:
            print(f"FAIL {name}: no {PROVIDER} inside")
            failures += 1
            continue

        cudnn = [s for s in sonames if s.startswith("libcudnn.so.")]
        majors = {re.sub(r".*\.so\.", "", s) for s in cudnn}

        if majors == {expected}:
            cuda = [
                s
                for s in sonames
                if re.match(r"lib(cudart|cublas|cublasLt|cufft|curand|cusparse)\.", s)
            ]
            print(f"ok   {name}: cuDNN {expected}  ({', '.join(cuda)})")
        else:
            print(f"FAIL {name}: expected cuDNN {expected}, links {cudnn or 'nothing'}")
            failures += 1

    print(f"\n{len(sys.argv) - 2} wheel(s), {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
