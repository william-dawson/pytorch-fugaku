# Failure signatures and demonstrated fixes

**November 2026 update:** see [large-pages.md](large-pages.md) for the fresh
allocator rebuild, linked runner, current validation and performance limitations.
Earlier benchmark records below describe the previous allocator build.

Use these diagnoses when the signature matches. Preserve the log and inspect
the actual command before changing flags. Do not globally disable warnings or
pretend configure probes succeeded.

| Signature | Demonstrated cause and fix |
|---|---|
| Missing `third_party/mimalloc` | Linux/aarch64 configuration enables it. Stage the pinned submodule up front. |
| `USE_MPI` silently becomes OFF | FindMPI compiled and linked successfully but `MPI_CXX_HEADER_DIR` was NOTFOUND outside the sysroot. Set the real C and C++ header paths, reconfigure with MPI ON, and check the cache and backend compilation unit. |
| LAPACK becomes unavailable; `elf_nextscn`, `elf_getdata`, etc. unresolved | Add `-lelf` to the complete Fujitsu link set and rerun the failed `FUGAKU_LAPACK_LINKS` check rather than reusing its cached failure. |
| `Cannot change output format whilst linking AArch64 binaries` | PyTorch's prioritized-text generator ran host `ld -verbose`, emitting an x86 linker script. Export `LD` as the cross-linker before configuration. Verify `OUTPUT_FORMAT` is AArch64. |
| `RegisterFunctionalization_*.cpp` fails `-Werror=missing-prototypes` | PyTorch intended to exempt generated sources, but REALPATH of a source did not match an uncanonicalized build-directory alias. `generated-paths.patch` compares canonical paths and applies the property to the original source spelling. Keep handwritten-code warnings enabled. |
| Missing `ActivityType.h` with `USE_KINETO=OFF` | The profiler shim includes it unconditionally. Include pinned Kineto sources even though the Kineto runtime stays disabled. |
| Missing `<version>` in `c10/util/Semaphore.h` | GCC 8 target headers lack this optional feature-test header. `semaphore-version.patch` guards the include; PyTorch's existing moodycamel semaphore fallback is then used. Its headers are already vendored. |
| `std::atomic<T>::value_type` missing | GCC 8's atomic specialization lacks that alias. `atomic-value-type.patch` uses the equivalent `T` in the internal memory planner helper. |
| Unresolved `std::filesystem::*` | GCC 8 needs separate `stdc++fs`. `filesystem-link.patch` links it to `torch_cpu` for this toolchain and adds `-z defs` so missing core definitions fail at library link time. |
| Final executable says Fujitsu libraries "needed by ... not found" | A cross-linker does not necessarily use runtime RPATHs to resolve transitive dependencies. Supply `-rpath-link` for Fujitsu's library directory. This fixed the C++ smoke executable. |
| `libshm` allocator/vtable symbols unresolved; warning that `libtorch_cpu.so` is not found | The split Python build's shared-memory manager needed `-rpath-link` to both `<workspace>/target/lib` and Fujitsu's `lib64`. The allocator implementation was present. Do not duplicate or alter the allocator to mask a search-path problem. |
| Custom version asserts `PYTORCH_BUILD_NUMBER must be set` | Set `PYTORCH_BUILD_VERSION=2.11.0+fugaku.llvm` and `PYTORCH_BUILD_NUMBER=1`. |
| Python or NumPy silently disabled | Use a native x86 Python 3.11 for generation and explicit AArch64 Python 3.11/NumPy headers. Check `BUILD_PYTHON`, `USE_NUMPY`, the include paths and MPI binding defines before compiling. |
| Threads reported incorrectly; threaded BLAS crashes with LLVM runtime | Check active runtime precedence, the `libfj90i` stubs and unwanted `libomp` linkage. Use the Fujitsu OpenMP configuration described in toolchain.md. |

## Runtime and performance signatures

From the November 2026 training work; details and evidence in [performance.md](performance.md).

| Signature | Cause and fix |
|---|---|
| Hundreds of major faults every backward, on the same `libtorch_cpu.so` code pages; steps never settle | LLIO compute-node cache (default 128 MiB) smaller than torch's code. `#PJM --llio cn-cache-size=384Mi`. `mlock`, `$PJM_LOCALTMP` and `/2ndfs` do not help. |
| Backward stalls of seconds to minutes; perf shows libfjomp barrier spin and one thread in libmpi/libtofucom | torch `ProcessGroupMPI`. Use `fjmpi_port` (`assets/backends`). `OMP_WAIT_POLICY=PASSIVE` makes it worse. |
| `environment variable RANK expected` from `init_process_group("fjmpi_port")` | Source pf-setup's `pf.env` and export `RANK=$PMIX_RANK`, `LOCAL_RANK`, `LOCAL_SIZE`, `USE_LIBUV=0`. |
| 1B: exit 137 on some ranks, or `Bus error ... Non-existant physical address` inside libmpi | Node out of memory: libmpg large pages, host glibc, or mimalloc with `MIMALLOC_PURGE_DELAY=-1`. Preload mimalloc with its default purge delay. |
| 1B: 20-30x the container's minor faults, high system CPU, matmul time identical | libmpg with `XOS_MMM_L_HPAGE_TYPE=none` returns freed memory and refaults it. Preload mimalloc. |
| Step times jitter 2x; involuntary context switches concentrate on one rank per node | Something runs inside that rank's CPU set (in our case a memory sampler) and preempts spinning OpenMP threads. Remove it. |
| `/dev/shm` staging: `Permission denied`, exit 126 | `/dev/shm` is noexec on compute nodes. |
| `module: command not found` when building over ssh | Non-login shell. `bash -lc 'bash build.sh'`. |
| LLVM build 40x slower at 2 ranks x 24 threads, sequence 2048 (step 0 > 1000 s); fine at 4 x 12 and at 2 x 24 with sequence 512 | Open; see [comparisons.md](comparisons.md). Use 4 ranks x 12 threads. |
| `lstat /share/...: no such file or directory` in a job | `/share` is not mounted on compute nodes; copy the file to a data volume. |
| Legacy SIF: `No module named 'torch'` | Use `singularity run` (the runscript activates the venv), not `exec`. |
| Legacy SIF: `libfjlapackexsve.so.1: cannot open shared object file` | Bind the host's `/opt/FJSVxtclanga` (the image expects tcsds-1.2.36) plus pf-run's interconnect binds. |
| Legacy SIF: `libunwind.so.8: cannot open shared object file` from `/bin/sh` | Caused by preloading the image's tcmalloc; libunwind exists nowhere on the system. Drop the preload. |

## Caches, aliases and restart discipline

The same data directory was visible as `/vol0402/data/...` and
`/vol0004/ra000009/data/...`. Canonicalization affected generated-source
warning exemptions. Switching aliases can also change CMake command hashes and
trigger regeneration/recompilation. Keep one configured remote path and do not
reuse a cache from another architecture or destination.

The first attempt to increase build parallelism checked a PID on a different
login node. That missed the running build and briefly started a duplicate in
the same directory. The duplicate was stopped, the original was monitored
through shared logs, and subsequent launches used a shared lock plus host/PID
records. The failed resizing/peer-SSH scripts are deliberately not included.

The new reproduction launchers require a fresh workspace, use one lock for both
phases, inherit its descriptor into the build, archive prior logs, and remove
stale completion files. A lock is only effective for processes using that same
lock and on a filesystem supporting cross-host locking. Do not use it to infer
that an older, unlocked build has exited.

A later login can land on another host. `kill -0 PID` there is not a global
status check and can even match an unrelated process. If the recorded host is
unreachable, do not hunt or signal processes through guessed peer SSH settings.
Read shared progress files and state what remains unverified.
