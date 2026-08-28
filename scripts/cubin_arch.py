import sys, zipfile, io, struct, collections, tempfile, os

def cubin_archs(data):
    # Inside .nv_fatbin each embedded cubin is a normal ELF64; nvcc stores the SM
    # number in the low byte of e_flags (sm_87 -> 0x57). Scanning for the ELF magic
    # is enough -- we do not need to parse the fatbin container itself.
    archs = collections.Counter()
    offset = 0
    magic = b"\x7fELF"
    while True:
        index = data.find(magic, offset)
        if index == -1:
            break
        offset = index + 4
        if index + 0x38 > len(data):
            break
        if data[index + 4] != 2:  # ELFCLASS64
            continue
        e_machine = struct.unpack_from("<H", data, index + 0x12)[0]
        if e_machine != 190:  # EM_CUDA
            continue
        e_flags = struct.unpack_from("<I", data, index + 0x30)[0]
        archs[e_flags & 0xFF] += 1
    return archs

for wheel_path in sys.argv[1:]:
    with zipfile.ZipFile(wheel_path) as zf:
        name = next(n for n in zf.namelist() if n.endswith("libonnxruntime_providers_cuda.so"))
        blob = zf.read(name)
    archs = cubin_archs(blob)
    pretty = ", ".join(f"sm_{arch}: {count}" for arch, count in sorted(archs.items()))
    print(f"{os.path.basename(wheel_path)[:46]:46s} {pretty or 'NONE FOUND'}")
