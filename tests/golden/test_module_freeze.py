import importlib.util
import sys
from pathlib import Path

FREEZE_SCRIPT = Path("scripts/maintenance/module_freeze.py")


def __freeze_module(monkeypatch):
    spec = importlib.util.spec_from_file_location("module_freeze", FREEZE_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_every_frozen_module_matches_its_freeze_hash(monkeypatch):
    freeze = __freeze_module(monkeypatch)

    changed = {
        module: expected
        for module, expected in freeze.frozen_hashes().items()
        if freeze.module_hash(module) != expected
    }

    assert not changed, (
        f"frozen modules changed: {sorted(changed)}; revert the change, or add an unfreeze line "
        "with the reason to doc/notes/module_ledger.csv"
    )
