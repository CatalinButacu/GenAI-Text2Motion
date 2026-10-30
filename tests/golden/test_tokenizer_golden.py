import importlib.util
import json
from pathlib import Path

import torch

GOLDEN_OUTPUTS_SCRIPT = Path("scripts/qa/golden_outputs.py")


def __golden_outputs_module():
    spec = importlib.util.spec_from_file_location("golden_outputs", GOLDEN_OUTPUTS_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_tokenizer_outputs_match_the_golden_fingerprints():
    golden = __golden_outputs_module()
    record = json.loads(golden.TOKENIZER_GOLDEN_PATH.read_text(encoding="utf-8"))
    assert record["torch"] == torch.__version__, (
        f"the golden fingerprints come from torch {record['torch']}, this run uses "
        f"{torch.__version__}; check the outputs, then run golden_outputs.py save-tokenizer"
    )
    missing = [path for path in record["inputs"] if not Path(path).is_file()]
    assert not missing, f"golden inputs are missing (artifacts/ is not in git): {missing}"
    pins = golden.input_pins(tuple(Path(path) for path in record["inputs"]))
    changed = sorted(path for path, digest in pins.items() if record["inputs"][path] != digest)
    assert not changed, f"golden inputs changed since the fingerprints were saved: {changed}"

    outputs = {
        name: golden.tensor_fingerprint(value) for name, value in golden.tokenizer_outputs().items()
    }

    assert outputs.keys() == record["outputs"].keys()
    different = sorted(name for name, value in outputs.items() if value != record["outputs"][name])
    assert not different, f"tokenizer outputs differ from the golden fingerprints: {different}"
