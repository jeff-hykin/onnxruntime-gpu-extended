#!/usr/bin/env python3
"""Make abseil's IfRRef::AddPtr alias survive nvcc's frontend.

abseil lts_2025xx writes `IfRRef<int KQual>::AddPtr<K>`. nvcc re-emits that for
the host compiler with a `typename` prefix but without the required `template`
keyword, which is ill-formed, so the host gcc rejects it (reproduced on gcc 11
through 16 — swapping the host compiler does not help). Hoisting the nested
alias template to namespace scope makes nvcc's `typename` prefix legal.

No-op on older abseil (what onnxruntime <=1.22.x pins), which lacks the pattern.
"""

import re
import sys
from pathlib import Path

HELPER = """
template <class T, class Other>
using IfRRefAddPtr = typename IfRRef<T>::template AddPtr<Other>;
"""

ANCHOR = """template <class T>
struct IfRRef<T&&> {
  template <class Other>
  using AddPtr = Other*;
};
"""

USE = re.compile(r"IfRRef<int (\w+)Qual>::AddPtr<(\w+)>")

root = Path(sys.argv[1])
common = root / "absl/container/internal/common.h"
users = [
    root / "absl/container/internal/raw_hash_map.h",
    root / "absl/container/internal/btree_container.h",
]

# A missing tree means the caller passed a stale path; failing here beats
# silently skipping the patch and losing a multi-hour build to it.
if not common.exists():
    sys.exit(f"patch_abseil: {common} not found")

if ANCHOR not in common.read_text():
    print("patch_abseil: IfRRef pattern absent (older abseil), nothing to do")
    sys.exit(0)

text = common.read_text()
if "IfRRefAddPtr" in text:
    print("patch_abseil: already applied")
    sys.exit(0)
common.write_text(text.replace(ANCHOR, ANCHOR + HELPER, 1))

total = 0
for path in users:
    if not path.exists():
        continue
    before = path.read_text()
    after, count = USE.subn(r"IfRRefAddPtr<int \1Qual, \2>", before)
    if count:
        path.write_text(after)
    total += count

print(f"patch_abseil: rewrote {total} AddPtr use(s)")
if total == 0:
    sys.exit("patch_abseil: helper added but no uses rewritten — check abseil layout")
