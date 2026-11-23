# Mamba selective-scan kernel: our MSVC port

## Upstream

- Package: mamba-ssm 2.3.2.post1.
- Source: https://github.com/state-spaces/mamba (PyPI sdist mamba_ssm-2.3.2.post1.tar.gz).
- License: Apache License 2.0 (file LICENSE in this folder).
- Authors: Albert Gu and Tri Dao.
- Reference: Gu and Dao (2023), Mamba: Linear-Time Sequence Modeling with Selective State Spaces, arXiv:2312.00752. The bibliography key is gu2023mamba (docs/paper/cls/bib.bib).

## Our copy

- Repository commit: e2c484e ("Add the mamba-ssm 2.3.2 source with the two changes for MSVC").
- Tracked files: 80.
- Checked on 2026-11-22: every file matches upstream except the two kernel files below and the egg-info folder. The egg-info folder is a build artifact and holds no changes of ours.

## Our changes (the only two kernel edits)

Both files lose the same block: the ROCm (AMD) branch around the shared-memory call.

| File | Change | Lines removed |
|---|---|---|
| csrc/selective_scan/selective_scan_fwd_kernel.cuh | Remove the `#ifndef USE_ROCM` branch, keep the CUDA call | 6 |
| csrc/selective_scan/selective_scan_bwd_kernel.cuh | Remove the `#ifndef USE_ROCM` branch, keep the CUDA call | 6 |

Reason: this machine builds only for CUDA. The removed branch is never compiled here, and it adds preprocessor blocks that MSVC handles poorly.

To list the exact diff, compare each file with the upstream sdist. Use `--ignore-cr-at-eol`, because line endings differ.

## What we use from Mamba

- The selective scan kernel, called through mamba_ssm.ops.selective_scan_interface (function selective_scan_fn).
- Our call sites: src/text2motion/generation/nn_backbones.py (the scan function lookup near line 157, and the kernel branch near line 172). The flag use_kernel selects this path. The Python scan is the default (use_kernel false).
- The compiled module is selective_scan_cuda.cp312-win_amd64.pyd in third_party/mamba_kernel. It is built for Torch 2.12.0+cu132 and it is git-ignored (size about 326 MB). Its SHA-256 hash is in third_party/mamba_kernel/SHA256SUMS.
- Target GPU: RTX 4050, CUDA architecture 8.9.

## Third change: launch dispatch without nested lambdas (2026-11-22)

- Problem: MSVC stops with error C2131 at the BOOL_SWITCH block of selective_scan_fwd_launch (line 315 in the upstream file). MSVC rejects a named constant from an outer lambda used as a template argument in a nested lambda.
- Change: in both kernel files, the chain of BOOL_SWITCH lambdas is replaced by plain function templates. Each level takes one flag as a template parameter. The launch functions keep their names and signatures, so the callers do not change.
- The macro in static_switch.h is unchanged and is not used by these two files any more.
- Verification:
  - Build for Python 3.14, Torch 2.12.0+cu132, architecture 8.9, exit 0. Log: runs/logs/kernel_build_py314_sm89.log.
  - Kernel against the reference scan (selective_scan_ref), random inputs, lengths 128, 250, 256 and 1000: relative forward error 4e-7 to 4e-6; relative gradient error 4e-7 to 4e-6 for all five inputs.
  - Speed, one backbone training step (batch 64, length 64, default generator size, TF32): Python scan median 49.9 s, peak GPU memory 10.6 GB; kernel median 2.9 s, peak 1.7 GB. The GPU has 6 GB, so the Python scan spilled into shared memory.
- The original upstream pattern is in the backup of the kernel files (not in the repository). The upstream sdist is the reference for the diff.

## Known open issue

- The ROCm branch removal (first change) is unchanged.
- Any further change must be recorded in this file.

## How to cite the port in the thesis

State that the kernel comes from Gu and Dao (2023) under Apache 2.0, with the two MSVC changes listed above. Do not describe the port as new work.
