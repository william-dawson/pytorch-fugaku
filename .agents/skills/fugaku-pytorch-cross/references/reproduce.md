# Reproduce from a fresh workspace

## Inputs and boundaries

Use Python 3.12+ and Git on the preparation machine. The two build interpreters
on Fugaku must be x86 CPython 3.12 and 3.11; the target installation must be
AArch64 CPython 3.11, with development headers. Paths are explicit inputs.
The validated compiler/SDK paths are in toolchain.md. Prepare under project
storage, not the home quota, and allow space for sources, two builds, package
staging, the installed wheel and its dependencies. Do not commit those outputs.

The recipe came from [tttapa/cross-pytorch](https://github.com/tttapa/cross-pytorch/tree/8a262aae82d254972c3d1b4065395bc8f3cd9412),
especially native `protoc`, native SLEEF generators followed by target SLEEF,
and its static-SLEEF import/include patch. That reference disabled MPI and
Python in its sample; this recipe explicitly enables and validates both.

`inputs.lock.json` records PyTorch v2.11.0 at
`70d99e998b4955e0049d13a98d77ae1b14db1f45`, all CPU dependency commits,
Eigen 5.0.0 at its pinned commit, and every downloaded wheel's SHA-256.
The selected sources include protobuf, pybind11, cpuinfo, ONNX, SLEEF, fmt,
pthreadpool, FXdiv, FP16, psimd, FlatBuffers, nlohmann, pocketfft, cpp-httplib,
**mimalloc and Kineto**. Nested optional GPU dependencies are not needed.
Eigen is an optional checkout rather than a normal gitlink. Do not accidentally
fetch NCCL just to obtain Eigen. SLEEF is the **bundled 3.8.0**, not the older
3.6 from the reference recipe; host and target builds use the same source.

## 1. Fetch or supply verified caches

From the repository root, after authorizing the network activity:

```sh
SKILL="$PWD/.agents/skills/fugaku-pytorch-cross"
python3 "$SKILL/scripts/fetch-inputs.py" /path/to/input-cache
```

This clones pinned sources, initializes only selected submodules, and downloads
binary wheels without installing them into an existing environment. It checks
revision and wheel hashes and refuses mismatches. A cache has `source/`,
`wheels/` (Linux x86 cp312), `host311-wheels/` (Linux x86 cp311) and
`python-wheels/` (Linux AArch64 cp311 and architecture-independent dependencies).

Preparation can instead receive `--source`, `--host-wheels`, `--host311-wheels`
and `--target-wheels` separately. It never downloads anything itself.

## 2. Stage an isolated tree

Example paths from the successful account; choose a **new** remote directory:

```sh
python3 "$SKILL/scripts/prepare.py" \
  --cache /path/to/input-cache \
  --workspace /path/to/new-local-stage \
  --remote-root /vol0402/data/ra000009/u10035/agent/work/pytorch-llvm-repro-001 \
  --project ra000009 \
  --host-python312 /vol0402/data/ra000009/u10035/fminiconda3/bin/python3.12 \
  --host-python311 /vol0402/data/ra000009/u10035/fminiconda3/envs/front/bin/python3.11 \
  --target-python-prefix /vol0402/data/ra000009/u10035/miniconda3
```

The preparer exports **committed** source content, so unrelated changes in a
cache checkout are not copied. It verifies all pins, renders account/site paths,
copies public wheels, stages AArch64 runtime dependencies without executing
them, and writes an input checksum manifest and scheduler JSON specs.
It refuses an existing destination. It does not copy old builds, core libraries,
wheels built by this recipe, or historical validation results.

The prepared Python 3.11 generator venv is isolated and gets pinned PyYAML
6.0.2 and NumPy 2.2.6 x86 wheels. The original successful run inherited YAML
and NumPy 2.2.1 from its pre-existing front environment; this explicit isolation
is a reproducibility adaptation that still needs a fresh full cluster run.

Transfer the prepared tree to exactly `--remote-root`, using a typed filesystem
upload/extract or an approved transfer. An archive can be made outside the tree:

```sh
tar -C /path/to/new-local-stage -czf /path/to/reproduction-inputs.tar.gz .
```

Before transfer, run the local preflight and validation-recorder regression checks:

```sh
python3 "$SKILL/scripts/validate-stage.py" /path/to/new-local-stage
python3 "$SKILL/scripts/test-record-validation.py"
```

The preflight exercises all five patches in an isolated temporary copy, reverses
them, verifies the input seals and AArch64 NumPy extension, and checks helper
syntax. It neither compiles target code nor submits a job. The original local
preflight report is in `references/evidence/reproduction-preflight.json`.

Do not copy private SSH keys, Conda installations, commercial Fujitsu libraries,
or the older port's container into this bundle. The staged checksum manifest
covers scripts, patches, specs, locks and public wheel inputs. Source exports
are identified by commit; source files intentionally changed by patches and
generated files are excluded from the launch-time checksum check. If a helper
is deliberately changed, restage from the reviewed recipe into a new workspace
instead of silently bypassing a checksum failure.

## 3. Build and validate LibTorch

On the x86 login node, in the staged remote tree:

```sh
bash /absolute/new/workspace/launch.sh
```

The launcher records `build.host`, `build.pid`, `build.log`, `status.txt` and
`exit-code.txt`, and keeps a shared lock throughout the build. `build.sh`:

1. Creates the native x86 host venv from offline wheels.
2. Applies the five patches idempotently.
3. Builds and installs host protoc with `/usr/bin/gcc` and `/usr/bin/g++`.
4. Builds native SLEEF, then cross-builds static target SLEEF with
   `NATIVE_BUILD_DIR` pointing to the native build and `SVE_VECTOR_BITS=512`.
5. Configures CPU LibTorch with `BUILD_PYTHON=OFF`, MPI/OpenMP ON, explicit
   Fujitsu math libraries, target feature restrictions, and native protoc.
6. Runs `check-build.py configuration` before compiling. It checks MPI options,
   a real LAPACK link, the generated-file warning exemption and compilation of
   `ProcessGroupMPI.cpp` with `USE_C10D_MPI`.
7. Builds with at most eight jobs, installs into `target/`, verifies AArch64 ELF
   libraries, retained Fujitsu BLAS, MPI factory symbols and OpenMP dependencies,
   and links `build-smoke/torch-mpi-smoke`.

Wait for `ready-for-compute-node-validation` and exit code 0. Submit
`job-libtorch-r2.json` through `submit_job` after the required resource preview.
The JSON runs `run-validation.sh libtorch 2` on two compute nodes. A manual PJM
equivalent must retain the group, storage declaration, node/rank layout and
power mode in validation.md; compiling in that job is not part of the recipe.

After reading the complete job outputs, record them with a host interpreter:

```sh
/absolute/new/workspace/host-venv/bin/python \
  /absolute/new/workspace/record-validation.py libtorch JOB_ID
```

This requires a successful build and application exit plus all expected PASS
markers on two different hosts. It creates a **new** `validation-results.json`.
The Python launcher requires that record; preparation does not manufacture it.

## 4. Build and package Python

After the two-node core test passes:

```sh
bash /absolute/new/workspace/launch-python.sh
```

This shares the same lock with the core launcher, records separate Python logs,
and builds in `build-python/`, installing into `python-prefix/`:

- `BUILD_LIBTORCHLESS=ON`, `BUILD_PYTHON=ON`, `LIBTORCH_LIB_PATH=<workspace>/target/lib`.
  The validated core is imported, not rebuilt.
- The initial cache copies validated CPU choices, without reusing the core
  build directory or its host Python discovery results.
- Explicit target Python include/library paths and target NumPy 2.2.6 headers
  accompany the native x86 Python 3.11 generator.
- MPI, distributed support, OpenMP and NumPy must remain ON. The guard checks
  the binding compilation unit's `USE_C10D_MPI` define and rejects accidentally
  rebuilding `torch_cpu` or using the known host Python header path.
- Executable link flags include `-rpath-link` for both imported LibTorch and
  Fujitsu runtime directories. Runtime RPATH alone was insufficient.
- Both `PYTORCH_BUILD_VERSION` and `PYTORCH_BUILD_NUMBER=1` are set.

CMake builds `libtorch_python`, `libshm` and `torch_shm_manager`.
`python-extension/CMakeLists.txt` separately builds `torch/csrc/stub.c` into
`_C.cpython-311-aarch64-linux-gnu.so`, linked to `libtorch_python`, with
`$ORIGIN/lib` lookup. This route does not run `setup.py`.

`package-python.py` copies Python package trees (`torch`, `torchgen`,
`functorch`), generated version metadata, the extension, core/binding libraries,
headers and shared-memory manager. It writes standard wheel metadata and RECORD
using `wheel.wheelfile`, then checks ELF architectures and OpenMP dependencies.
The pinned source commit is written into `torch/version.py` because the source
export has no `.git` directory. The result is:

```text
dist/torch-2.11.0+fugaku.llvm-cp311-cp311-linux_aarch64.whl
```

It is **Fugaku-specific**, not a general manylinux claim. Fujitsu libraries are
resolved from the site installation, not redistributed. The assembler has not
recreated all official setuptools console-entry-point metadata; the tested
launch path is `mpiexec` plus the target Python interpreter. CLI launchers,
`torch.compile`, extension compilation and general model coverage need separate
validation before claiming parity with an official PyTorch distribution.

The script installs the resulting wheel into `wheel-install/` with pip's
explicit cp311/linux_aarch64 target options, without executing target Python
on x86 or modifying the user's Python installation. Wait for
`ready-for-python-runtime-validation` and Python build exit code 0.

## 5. Validate the installed wheel

Run the generated two-node smoke and async specs, then the three separate
one-node benchmark specs. The rank wrapper imports from `wheel-install/` and
the isolated `runtime-site/`, not the source checkout or another installed torch.
Record each new job with `record-validation.py smoke|async|benchmark JOB_ID`;
for benchmarks also supply `--ranks 1`, `2` or `4`. See validation.md for what
each test establishes and what it does not.

Archive new source/input pins, script hashes, resolved toolchain versions,
build logs, job specs, complete outputs, exit codes and wheel SHA-256. Compare
behavior and metrics with results.md, not only filenames. A rebuilt wheel may
have a different byte checksum due to timestamps or documented staging changes.
