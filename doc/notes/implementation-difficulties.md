# Implementation difficulties

This file records problems of the implementation that the thesis can report.
Each entry gives the problem, the cause, what we did, and the state.

## 1. The compiled selective-scan kernel on Windows (2026-10-10)

**Goal.** The reference Mamba package (`mamba-ssm`) has a compiled CUDA kernel for the selective
scan. Our layer runs in plain PyTorch. We tried to build the kernel on the training computer.

**Environment.** Windows 11, Visual Studio 18.6 (MSVC), CUDA toolkit 13.2, PyTorch 2.12 for
CUDA 13.2, `mamba-ssm` 2.3.2.post1 from PyPI. The package gives no Windows binary, so the
installer builds it from source.

**Problems, in the sequence in which they occurred.**

| # | Symptom | Cause | Repair |
|---|---|---|---|
| 1 | `cl : ignoring unknown option '-O3'` and `'-std=c++17'` | The build script passes compiler options in the GCC form. MSVC ignores them. | Set the environment variable `CL=/std:c++17`. |
| 2 | `fatal error C1189: MSVC/cl.exe with traditional preprocessor is used` | The CUDA 13 headers need the standard preprocessor of MSVC. The default is the old one. | Add `/Zc:preprocessor` to `CL` and to `NVCC_APPEND_FLAGS`. |
| 3 | `identifier "M_LOG2E" is undefined` | MSVC gives the mathematics constants only when `_USE_MATH_DEFINES` is set. | Add `/D_USE_MATH_DEFINES`. |
| 4 | `"#" not expected here` in `selective_scan_bwd_kernel.cuh` and `selective_scan_fwd_kernel.cuh` | The source has `#ifndef USE_ROCM` inside the argument of a macro. GCC permits this. MSVC does not. | Change the source: keep the branch for CUDA, remove the three preprocessor lines. This is a change of the package source. |
| 5 | Each kernel file compiles for approximately 5 minutes | The build makes code for each GPU generation that PyTorch supports. | None. The full build needed 46 minutes. |
| 6 | `ModuleNotFoundError: No module named 'triton'` at import | The package imports the Triton compiler for a different function. Triton has no Windows release. | Load the scan interface without the start file of the package, and give an empty module in place of the Triton layer. `scripts/benchmarks/kernel_probe.py` does this. |
| 7 | `No module named 'einops'` | The build ran without the dependencies of the package. | Install `einops`. |

**What the kernel can change, and what it cannot.**

- It replaces the parallel scan of **training**. The kernel uses less memory, so the gradient
  checkpoint of each block is not necessary. Training is the part that is 9 times slower than
  the transformer today.
- It does **not** change the step path of **streaming**. A stream makes one token at each step
  and does not use the scan. The measured inference cost stays the same.
- A model that is trained with the kernel is a different run. To keep the comparison of the two
  twins valid, a result with the kernel needs a new matched run.

**State.** All 7 problems are solved. The kernel builds and runs on Windows.

**Measurement** (`scripts/benchmarks/kernel_probe.py`, run `20261010T153517Z_kernel_probe`).
The 34M Mamba generator, the same weights in the two versions, batch 8, 49 tokens, one forward
and backward pass of the generator only (no loss decoder, no text encoder, no data loader):

| | Plain PyTorch scan | Compiled kernel |
|---|---:|---:|
| Time for one training step | 957 ms | 65 ms |
| Peak GPU memory | 991 MB | 800 MB |
| Largest difference of the outputs | | 9.5e-7 |

The kernel is 14.7 times faster for this step and gives the same output to 6 decimal places.
**Full epoch** (run `20261012T145330Z_generator_mamba`, the settings of the 60-epoch run, no
data worker processes): 9.3 minutes with the kernel, 66.5 minutes without it. The loss after
epoch 1 is 6.399 with the kernel and 6.408 without it. The overfit gate passes in 26 steps in
the two cases.

**In the training code.** The import repair for problem 6 is in
`src/text2motion/generation/scan_kernel.py`, so `use_kernel: true` works in a config.

## 2. Memory of the training computer (2026-10-12)

The computer has 15.9 GB of RAM. With the editor, a browser and other programs open, 0.5 GB
was free. A training run with 2 data worker processes, a test run and a benchmark at the same
time stopped the computer. The training log shows `Couldn't open shared file mapping ...
error code: <1455>`: Windows could not give more memory to a data worker. Each worker is a full
Python process with its own copy of the libraries. The repair is to run one heavy job at a time
and to use no worker processes when memory is low.

## 3. Earlier records

- Cloud (AWS, June 2026): the kernel of `mamba-ssm` did not link with PyTorch 2.7 for CUDA 12.8
  (an undefined symbol at import). Training used the plain PyTorch scan.
- The plain PyTorch scan keeps several tensors of the full sequence, so a 4 GB GPU needs a
  gradient checkpoint for each Mamba block. This is one cause of the training cost.
