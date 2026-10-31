# Recorded baseline: 2026-10-30/31 (Asia/Tokyo)

These are historical results from the successful staged build, not a substitute
for validation of a fresh reproduction. The tested remote workspace was
`/vol0402/data/ra000009/u10035/agent/work/pytorch-llvm-cross`.
Exact pinned inputs are in `assets/inputs.lock.json`. Original milestone
provenance is retained in `evidence/`; later scripts may have changed (for
example to add the async wrapper). `assets/recipe.sha256.json` identifies the
exact script snapshot committed with this skill.

## Completed artifacts and tests

| Job | Test | Result |
|---|---|---|
| 53142196 | LibTorch, two nodes, 2 ranks x 48 threads | GEMM, LAPACK, autograd, MPI all-reduce, active OpenMP PASS; exit 0 |
| 53147118 | Target Python probe | AArch64 CPython 3.11.5 and expected cp311 ABI |
| 53149105 | Installed Python wheel, two nodes, 2 ranks x 48 threads | Imports, NumPy interoperability, GEMM, LAPACK, autograd, MPI and OpenMP PASS; exit 0 |
| 53149351 / 53149353 / 53149355 | One-node Python GEMM, 1/2/4 ranks | Both dtypes and correctness PASS; all exit 0 |
| 53149769 | Async MPI, two nodes, 2 ranks x 12 threads | All seven cases PASS on both ranks; exit 0 |

The built wheel was `torch-2.11.0+fugaku.llvm-cp311-cp311-linux_aarch64.whl`,
77,468,960 bytes, SHA-256:

```text
326306e3d147f01f53bfc06e97c0ba30a179b1524113d2abb2fd33ab705cbef6
```

The wheel itself, source trees and third-party binary wheels are deliberately
not committed. Fujitsu software was not redistributed. Build and validation
scripts plus hashes support reconstruction; no fresh rebuild is claimed solely
because a wheel filename or old record matches.

## Python GEMM results

4096x4096, one node, requested 2.0 GHz, eco off. Aggregate TFLOP/s; see
validation.md for the precise statistic and limited sample size.

| Ranks x threads | FP32 | FP64 |
|---|---:|---:|
| 1 x 48 | 3.5545 | 2.2282 |
| 2 x 24 | 4.5170 | 2.2376 |
| 4 x 12 | 4.3997 | 1.8260 |

Two ranks performed best in this first case. Do not infer full model,
collective, larger-matrix or multi-node training performance from these numbers.

For context, the earlier 4096 raw probes in normal mode measured:

| Ranks | Native Fujitsu FP32/FP64 TFLOP/s | LLVM + Fujitsu math FP32/FP64 TFLOP/s |
|---|---:|---:|
| 1 | 1.543 / 0.825 | 1.710 / 0.923 |
| 2 | 3.868 / 1.581 | 3.102 / 1.295 |
| 4 | 5.476 / 2.231 | 5.233 / 1.975 |

The native reference used **Fortran** `mpifrtpx -Kfast -Kopenmp -SSL2BLAMP`.
A traditional Fujitsu C++ benchmark had pathological slow cases/timeouts and
was not the dependable reference. Probe sources are preserved in `assets/probes`.
The raw and Python measurements used different harnesses/runtime settings;
repeat matched experiments before interpreting differences as compiler speedups.

## Remaining numerical/performance limits

- Turning eco off substantially improved raw GEMM throughput. At four ranks,
  4096 native FP32 rose from about 3.19 to 5.48 TFLOP/s; LLVM from 3.17 to 5.23.
- A **raw LLVM 8192x8192, four-rank regression remains unresolved**: SGEMM took
  6.3599 s and DGEMM 21.1199 s (aggregate 0.692 and 0.208 TFLOP/s), versus native
  Fortran 0.9339 s and 2.8357 s (4.709 and 1.551 TFLOP/s). Do not hide this by
  reporting only the successful 4096 results. It was not retested through the Python wheel.
- Several eco-mode control runs had incomplete output despite wrapper exit 0;
  `evidence/math-library-results.json` marks them incomplete.
- No real model/DDP training, production overlap speedup, multi-node scaling,
  general torch.compile coverage or custom SVE512 kernel tuning was validated
  for this new cross-built wheel.

## Async detail

On hosts `a01-0101c` and `a01-0200c`, early-rank async all-reduce returned in
0.00023085 s with its peer delayed three seconds. An independent GEMM took
0.00109208 s and the reduction remained pending afterward. The receive-side
`irecv` returned in 0.00009043 s and completed through polling. Both nodes
verified all final values. These prove asynchronous API behavior; the delay
experiment is not an interconnect bandwidth/latency benchmark.

Evidence files include `validation-results.json`, `python-probe.json`,
`python-validation-results.json` and `mpi-async-results.json`. The original
`provenance.json` and `python-provenance.json` record input and artifact hashes.
