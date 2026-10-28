"""Add an SVE512 CPU capability to a PyTorch source tree (A64FX: 512-bit SVE, no BF16).

PyTorch's only SVE capability is SVE256 (Graviton 3: 256-bit, BF16). Its vector
headers are written against VECTOR_WIDTH, so a 512-bit capability is mostly
plumbing: a second capability beside SVE256 in CMake, DispatchStub and the
version string, VECTOR_WIDTH 64, and the few `!SVE256` guards that would
otherwise switch NEON code on in a 512-bit build. SVE512 is selected only when
the CPU reports exactly 512-bit vectors; a missing SVE512 kernel falls back to
DEFAULT, never to a 256-bit kernel.

Usage: python make_sve512_patch.py <pytorch-tree>; then `git diff` in the tree.
Edits are exact-text replacements that fail loudly if PyTorch has changed.
"""

import re
import sys
from pathlib import Path

root = Path(sys.argv[1])


def edit(rel: str, old: str, new: str, count: int = 1) -> None:
    path = root / rel
    text = path.read_text()
    found = text.count(old)
    if found != count:
        sys.exit(f"{rel}: expected {count} of {old[:60]!r}, found {found}")
    path.write_text(text.replace(old, new))


def twin_blocks(rel: str) -> None:
    """Give every `#ifdef HAVE_SVE256_CPU_DEFINITION` block an SVE512 twin.

    Argument lists and declarations get the twin after (order must match
    across declaration, definition and call); capability checks get it before,
    because the highest capability is tested first.
    """
    path = root / rel
    text = path.read_text()
    # Unindented blocks without nested directives; the nested, indented ones in
    # compute_cpu_capability() are edited explicitly below.
    block = re.compile(
        r"^#ifdef HAVE_SVE256_CPU_DEFINITION\n((?:(?!\s*#\s*(?:if|endif)).*\n)*?)#endif\n", re.M
    )

    def twin(m: re.Match) -> str:
        body = m.group(0)
        copy = body.replace("SVE256", "SVE512").replace("sve256", "sve512")
        return copy + body if "capability >=" in body else body + copy

    new, n = block.subn(twin, text)
    if n == 0:
        sys.exit(f"{rel}: no SVE256 blocks found")
    path.write_text(new)
    print(f"{rel}: {n} blocks twinned")


# --- CMake: detect 512-bit SVE support and build the capability without BF16.
edit(
    "cmake/Modules/FindARM.cmake",
    '    CHECK_COMPILES(CXX "ARM_BF16" ',
    '    CHECK_COMPILES(CXX "SVE512" "-march=armv8.2-a+sve -msve-vector-bits=512" "${SVE_CODE}")\n'
    '    CHECK_COMPILES(CXX "ARM_BF16" ',
)
edit(
    "cmake/Codegen.cmake",
    "  list(LENGTH CPU_CAPABILITY_NAMES NUM_CPU_CAPABILITY_NAMES)",
    "  # A64FX: 512-bit SVE without BF16.\n"
    "  if(CXX_SVE_FOUND AND CXX_SVE512_FOUND)\n"
    '    list(APPEND CPU_CAPABILITY_NAMES "SVE512")\n'
    '    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_SVE_CPU_DEFINITION -DHAVE_SVE512_CPU_DEFINITION")\n'
    '    list(APPEND CPU_CAPABILITY_FLAGS "${OPT_FLAG} -march=armv8.2-a+sve -DCPU_CAPABILITY_SVE -msve-vector-bits=512")\n'
    "  endif()\n\n"
    "  list(LENGTH CPU_CAPABILITY_NAMES NUM_CPU_CAPABILITY_NAMES)",
)

# --- Dispatch: the capability, its runtime selection and its kernel slot.
edit(
    "aten/src/ATen/native/DispatchStub.h",
    "#elif defined(HAVE_SVE256_CPU_DEFINITION) && defined(HAVE_ARM_BF16_CPU_DEFINITION)\n  SVE256 = 1,\n",
    "#elif (defined(HAVE_SVE256_CPU_DEFINITION) && defined(HAVE_ARM_BF16_CPU_DEFINITION)) || \\\n"
    "    defined(HAVE_SVE512_CPU_DEFINITION)\n  SVE256 = 1,\n  SVE512 = 2,\n",
)
twin_blocks("aten/src/ATen/native/DispatchStub.h")
edit(
    "aten/src/ATen/native/DispatchStub.h",
    "  REGISTER_SVE256_DISPATCH(name, fn)\n",
    "  REGISTER_SVE256_DISPATCH(name, fn)                                          \\\n"
    "  REGISTER_SVE512_DISPATCH(name, fn)\n",
)
edit(
    "aten/src/ATen/native/DispatchStub.h",
    "#define ALSO_REGISTER_SVE256_DISPATCH(name, fn) REGISTER_ARCH_DISPATCH(name, CPU_CAPABILITY, fn)\n",
    "#define ALSO_REGISTER_SVE256_DISPATCH(name, fn) REGISTER_ARCH_DISPATCH(name, CPU_CAPABILITY, fn)\n"
    "#define ALSO_REGISTER_SVE512_DISPATCH(name, fn) REGISTER_ARCH_DISPATCH(name, CPU_CAPABILITY, fn)\n",
)
# Files that register one non-vectorised kernel per capability by hand.
edit(
    "aten/src/ATen/native/DispatchStub.h",
    "// Macro to register the same kernel for all CPU arch types.",
    "// Register one kernel in every SVE slot this build has.\n"
    "#define REGISTER_SVE_DISPATCH(name, fn) \\\n"
    "  REGISTER_SVE256_DISPATCH(name, fn)    \\\n"
    "  REGISTER_SVE512_DISPATCH(name, fn)\n\n"
    "// Macro to register the same kernel for all CPU arch types.",
)
for rel in [
    "aten/src/ATen/native/SegmentReduce.cpp",
    "aten/src/ATen/native/BatchLinearAlgebraKernel.cpp",
    "aten/src/ATen/native/transformers/attention.cpp",
    "aten/src/ATen/native/mkl/SpectralOps.cpp",
    "aten/src/ATen/native/sparse/FlattenIndicesKernel.cpp",
    "aten/src/ATen/native/sparse/SparseBinaryOpIntersectionKernel.cpp",
]:
    path = root / rel
    text = path.read_text()
    n = text.count("REGISTER_SVE256_DISPATCH(")
    if n == 0:
        sys.exit(f"{rel}: no REGISTER_SVE256_DISPATCH")
    path.write_text(text.replace("REGISTER_SVE256_DISPATCH(", "REGISTER_SVE_DISPATCH("))
    print(f"{rel}: {n} registrations")

twin_blocks("aten/src/ATen/native/DispatchStub.cpp")
edit(
    "aten/src/ATen/native/DispatchStub.cpp",
    "#ifdef HAVE_SVE256_CPU_DEFINITION\n    if (envar == \"sve256\") {",
    "#ifdef HAVE_SVE512_CPU_DEFINITION\n"
    "    if (envar == \"sve512\") {\n"
    "      if (sve_vl == 512) {\n"
    "        return CPUCapability::SVE512;\n"
    "      }\n"
    "      TORCH_WARN(\"SVE512 capability not available on hardware. Falling back to DEFAULT\");\n"
    "      return CPUCapability::DEFAULT;\n"
    "    }\n"
    "#endif\n"
    "#ifdef HAVE_SVE256_CPU_DEFINITION\n    if (envar == \"sve256\") {",
)
edit(
    "aten/src/ATen/native/DispatchStub.cpp",
    "    #ifdef HAVE_SVE256_CPU_DEFINITION\n        if (sve_vl == 256) { // Check for SVE256",
    "    #ifdef HAVE_SVE512_CPU_DEFINITION\n"
    "        if (sve_vl == 512) { // A64FX\n"
    "          return CPUCapability::SVE512;\n"
    "        }\n"
    "    #endif\n"
    "    #ifdef HAVE_SVE256_CPU_DEFINITION\n        if (sve_vl == 256) { // Check for SVE256",
)
edit(
    "aten/src/ATen/Version.cpp",
    '#elif defined(HAVE_SVE256_CPU_DEFINITION) && defined(HAVE_ARM_BF16_CPU_DEFINITION)\n'
    '    case native::CPUCapability::SVE256:\n      return "SVE256";\n',
    '#elif (defined(HAVE_SVE256_CPU_DEFINITION) && defined(HAVE_ARM_BF16_CPU_DEFINITION)) || \\\n'
    '    defined(HAVE_SVE512_CPU_DEFINITION)\n'
    '    case native::CPUCapability::SVE256:\n      return "SVE256";\n'
    '    case native::CPUCapability::SVE512:\n      return "SVE512";\n',
)

# --- Vectors: 64-byte width, and the SVE headers rather than NEON or AVX2 ones.
edit(
    "aten/src/ATen/cpu/vec/vec_base.h",
    "#define VECTOR_WIDTH 64\n#define int_vector __m512i\n#elif",
    "#define VECTOR_WIDTH 64\n#define int_vector __m512i\n"
    "#elif defined(CPU_CAPABILITY_SVE512)\n"
    "#if defined(__GNUC__)\n#define __at_align__ __attribute__((aligned(64)))\n"
    "#else\n#define __at_align__\n#endif\n"
    "#define VECTOR_WIDTH 64\n#elif",
)
edit(
    "aten/src/ATen/cpu/vec/vec256/vec256.h",
    "#if defined(CPU_CAPABILITY_SVE256)\n#include <ATen/cpu/vec/sve/vec_common_sve.h>",
    "#if defined(CPU_CAPABILITY_SVE)\n#include <ATen/cpu/vec/sve/vec_common_sve.h>",
)
# NEON-only code guarded as "aarch64 and not SVE256" must stay off for SVE512.
for rel, n in [
    ("aten/src/ATen/cpu/vec/vec256/vec256_half.h", 1),
    ("aten/src/ATen/cpu/vec/vec256/vec256_bfloat16.h", 1),
    ("aten/src/ATen/cpu/vec/vec256/vec256_qint.h", 2),
    ("aten/src/ATen/cpu/vec/vec128/vec128_convert.h", 2),
]:
    edit(rel, "!defined(CPU_CAPABILITY_SVE256)", "!defined(CPU_CAPABILITY_SVE)", n)
print("done")
