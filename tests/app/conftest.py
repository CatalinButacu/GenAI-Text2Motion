import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

MAINTENANCE_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "maintenance"


def loaded_maintenance_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, MAINTENANCE_SCRIPTS / f"{name}.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load the script {name} from {MAINTENANCE_SCRIPTS}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def rename_everywhere() -> ModuleType:
    return loaded_maintenance_script("rename_everywhere")


@pytest.fixture(scope="module")
def add_class_separators() -> ModuleType:
    return loaded_maintenance_script("add_class_separators")
