from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from collections.abc import Callable
from pathlib import Path

INTERFACE_MODULE = "mamba_ssm.ops.selective_scan_interface"
TRITON_MODULES = ("mamba_ssm.ops.triton", "mamba_ssm.ops.triton.layer_norm")


def __register_package_without_triton() -> None:
    package_spec = importlib.util.find_spec("mamba_ssm")
    if package_spec is None or package_spec.origin is None:
        raise ModuleNotFoundError(
            "use_kernel needs the mamba_ssm package; build it from third_party/ "
            "(see doc/notes/implementation-difficulties.md)"
        )
    package_dir = Path(package_spec.origin).parent
    for name, folder in (("mamba_ssm", package_dir), ("mamba_ssm.ops", package_dir / "ops")):
        package_stub = types.ModuleType(name)
        package_stub.__path__ = [str(folder)]
        sys.modules[name] = package_stub
    for name in TRITON_MODULES:
        triton_stub = types.ModuleType(name)
        triton_stub.__path__ = []
        sys.modules[name] = triton_stub
    sys.modules[TRITON_MODULES[1]].__dict__["_layer_norm_fwd"] = None


def selective_scan_function() -> Callable:
    if INTERFACE_MODULE not in sys.modules and importlib.util.find_spec("triton") is None:
        __register_package_without_triton()
    return importlib.import_module(INTERFACE_MODULE).selective_scan_fn
