import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from text2motion.tokenization.contracts import TokenizerConfig
from text2motion.tokenization.nn_model import build_tokenizer_network

AUDIT_SCRIPT = Path("scripts/qa/checkpoint_audit.py")
TOKENIZER_BLOCK = "tokenizer:\n  width: 64\n  num_quantizers: 2\n  fsq_levels: {levels}\n"


def __audit_module(monkeypatch):
    spec = importlib.util.spec_from_file_location("checkpoint_audit", AUDIT_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("yaml_levels", "result"),
    [("[4, 4]", "OK"), ("[3, 5]", "MISMATCH"), (None, "NO_CONFIG")],
    ids=["same_levels", "same_length_other_levels", "no_yaml"],
)
def test_the_audit_compares_the_saved_levels_with_the_yaml(
    tmp_path, monkeypatch, yaml_levels, result
):
    audit = __audit_module(monkeypatch)
    monkeypatch.setattr(audit, "TOKENIZER_CONFIG_DIR", tmp_path)
    network = build_tokenizer_network(
        TokenizerConfig(width=64, num_quantizers=2, fsq_levels=(4, 4)), None
    )
    checkpoint = tmp_path / "small.pt"
    torch.save(network.state_dict(), checkpoint)
    if yaml_levels is not None:
        (tmp_path / "small.yaml").write_text(
            TOKENIZER_BLOCK.format(levels=yaml_levels), encoding="utf-8"
        )

    row = audit.audit_tokenizer(checkpoint)

    assert row.result == result
