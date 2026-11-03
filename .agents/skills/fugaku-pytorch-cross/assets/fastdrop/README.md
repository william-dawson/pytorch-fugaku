# Fast CPU `bernoulli_` (dropout masks) for the LLVM build

`fast_bernoulli.cpp` replaces `aten::bernoulli_.float` on CPU with a counter-based generator: 114× faster than stock on 12 A64FX threads. See references/performance.md, "Fix 4".

The extension prints one "Overriding a previously registered kernel" warning at load; that warning is expected.

## Build

Place this directory at `$WORK/h2h-fastdrop`, so that `build.sh` takes `WORK = ..`. Then, on a login node:

```sh
bash -lc 'bash build.sh'
```

## Use

```python
torch.ops.load_library("/path/to/libfastdrop.so")   # before training
```

In the h2h harness, set `H2H_FASTDROP=/path/to/libfastdrop.so`.

## Test

`run_test.sh` runs `test_fastdrop.py` (one CMG, the LLVM python). It covers:
- speed against stock PyTorch, in the same process
- mask means and values
- reproducibility under `manual_seed`
- neighbour correlation
- dtypes and non-contiguous tensors
- `F.dropout`
