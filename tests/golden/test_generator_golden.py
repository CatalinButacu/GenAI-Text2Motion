import importlib.util
import json
from pathlib import Path

import pytest
import torch

GOLDEN_OUTPUTS_SCRIPT = Path("scripts/qa/golden_outputs.py")


def __golden_outputs_module():
    spec = importlib.util.spec_from_file_location("golden_outputs", GOLDEN_OUTPUTS_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def restored_torch_flags():
    saved = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.get_float32_matmul_precision(),
    )
    yield
    torch.use_deterministic_algorithms(saved[0], warn_only=saved[1])
    torch.backends.cudnn.benchmark = saved[2]
    torch.backends.cudnn.deterministic = saved[3]
    torch.backends.cuda.matmul.allow_tf32 = saved[4]
    torch.backends.cudnn.allow_tf32 = saved[5]
    torch.set_float32_matmul_precision(saved[6])


def test_the_generator_metrics_match_the_golden_values(restored_torch_flags):
    assert torch.cuda.is_available(), (
        "the generator golden values come from CUDA; this run has none"
    )
    golden = __golden_outputs_module()
    record = json.loads(golden.GENERATOR_GOLDEN_PATH.read_text(encoding="utf-8"))
    assert record["torch"] == torch.__version__, (
        f"the golden values come from torch {record['torch']}, this run uses "
        f"{torch.__version__}; check the outputs, then run golden_outputs.py save-generator"
    )
    missing = [path for path in record["inputs"] if not Path(path).is_file()]
    assert not missing, (
        f"golden inputs are missing (artifacts/ and data/ are not in git): {missing}"
    )
    pins = golden.input_pins(tuple(Path(path) for path in record["inputs"]))
    changed = sorted(path for path, digest in pins.items() if record["inputs"][path] != digest)
    assert not changed, f"golden inputs changed since the values were saved: {changed}"

    metrics = json.loads(json.dumps(golden.generator_metrics()))

    assert metrics == record["outputs"]
