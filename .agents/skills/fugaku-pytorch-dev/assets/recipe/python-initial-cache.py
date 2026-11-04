"""Reuse validated CPU options without reusing build-directory or host artifacts."""
from pathlib import Path

root = Path(__file__).resolve().parent
options = {}
for line in (root / "build-pytorch/CMakeCache.txt").read_text().splitlines():
    if not line or line.startswith(("#", "//")) or "=" not in line:
        continue
    keytype, value = line.split("=", 1)
    key, kind = keytype.split(":", 1)
    if (key.startswith("USE_") and kind == "BOOL") or key in (
        "BLAS", "BLAS_LIBRARIES", "SLEEF_LIBRARY", "SLEEF_INCLUDE",
        "CXX_ARM_BF16_FOUND", "CXX_SVE256_FOUND",
    ):
        options[key] = (kind, value)
with (root / "python-initial-cache.cmake").open("w") as stream:
    for key, (kind, value) in sorted(options.items()):
        assert "]==]" not in value
        stream.write(f'set({key} [==[{value}]==] CACHE {kind} "Validated CPU option")\n')
