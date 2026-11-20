# run: .venv\Scripts\python.exe scripts\checks\test_ext.py
from typing import Any

import torch
from torch.utils.cpp_extension import load_inline

CPP_SRC = "torch::Tensor add_one(torch::Tensor x);"

CUDA_SRC = r"""
#include <torch/extension.h>

__global__ void add_one_kernel(const float* in, float* out, int64_t n) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] + 1.0f;
}

torch::Tensor add_one(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat32, "need float32 CUDA tensor");
    auto xc = x.contiguous();
    auto out = torch::empty_like(xc);
    int64_t n = xc.numel();
    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    add_one_kernel<<<blocks, threads>>>(xc.data_ptr<float>(), out.data_ptr<float>(), n);
    return out;
}
"""

MSVC_FLAGS = ("-Xcompiler", "/Zc:preprocessor")


def build_extension() -> Any:
    return load_inline(
        name="add_one_ext",
        cpp_sources=CPP_SRC,
        cuda_sources=CUDA_SRC,
        functions=["add_one"],
        extra_cuda_cflags=list(MSVC_FLAGS),
        verbose=True,
    )


def check_extension() -> None:
    extension = build_extension()
    tensor = torch.arange(10, dtype=torch.float32, device="cuda")
    result = extension.add_one(tensor)
    print("result:", result.tolist())
    assert torch.equal(result, tensor + 1)
    print("C++/CUDA extension OK")


if __name__ == "__main__":
    check_extension()
