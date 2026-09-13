from __future__ import annotations

import ast
from pathlib import Path

PACKAGE_ROOT = Path("src/text2motion")
ALLOWED_DEPENDENCIES = {
    "motion": {"motion"},
    "tokenization": {"motion", "tokenization"},
    "generation": {"motion", "tokenization", "generation"},
    "evaluation": {"motion", "tokenization", "generation", "evaluation"},
    "streaming": {"motion", "tokenization", "generation", "streaming"},
    "studio": {"motion", "studio"},
    "app": {
        "motion",
        "tokenization",
        "generation",
        "evaluation",
        "streaming",
        "studio",
        "app",
    },
}


def imported_stages(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    stages: set[str] = set()
    for node in ast.walk(tree):
        modules: list[str] = []
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
        elif isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        for module in modules:
            parts = module.split(".")
            if len(parts) >= 2 and parts[0] == "text2motion":
                stages.add(parts[1])
    return stages


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
        elif isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
    return modules


def test_pipeline_dependencies_point_inward() -> None:
    violations: list[str] = []
    for stage, allowed in ALLOWED_DEPENDENCIES.items():
        for path in (PACKAGE_ROOT / stage).rglob("*.py"):
            forbidden = imported_stages(path) - allowed
            if forbidden:
                violations.append(f"{path}: imports {sorted(forbidden)}; allowed {sorted(allowed)}")
    assert not violations, "upward or cross-stage imports:\n" + "\n".join(violations)


def test_application_bootstrap_is_used_only_at_entrypoints() -> None:
    allowed = {
        PACKAGE_ROOT / "app" / "bootstrap.py",
        PACKAGE_ROOT / "app" / "cli.py",
        PACKAGE_ROOT / "app" / "__init__.py",
    }
    offenders = []
    for path in PACKAGE_ROOT.rglob("*.py"):
        if path in allowed:
            continue
        if "ApplicationBootstrap" in path.read_text(encoding="utf-8"):
            offenders.append(str(path))
    assert not offenders, "composition leaked outside app entrypoints: " + ", ".join(offenders)


CONTRACT_SAFE_ROLES = {"contracts", "representation", "model"}
OWN_STAGE_CONTRACT_SAFE_ROLES = {"representation"}


def stage_packages_on_disk() -> set[str]:
    return {
        path.name
        for path in PACKAGE_ROOT.iterdir()
        if path.is_dir() and (path / "__init__.py").exists()
    }


def test_stage_map_matches_the_packages_on_disk() -> None:
    declared = set(ALLOWED_DEPENDENCIES)
    actual = stage_packages_on_disk()
    drift: list[str] = []
    for stage in sorted(declared - actual):
        drift.append(f"declared but missing on disk: {stage}")
    for stage in sorted(actual - declared):
        drift.append(f"on disk but ungoverned: {stage}")
    assert not drift, "stage map drifted from the package layout:\n" + "\n".join(drift)


def stage_module_names(stage: str) -> set[str]:
    stage_root = PACKAGE_ROOT / stage
    return {path.stem for path in stage_root.glob("*.py") if path.stem != "__init__"}


def test_every_stage_with_contracts_still_has_its_declaration_module() -> None:
    missing: list[str] = []
    for stage in ALLOWED_DEPENDENCIES:
        if not (PACKAGE_ROOT / stage / "contracts.py").exists():
            continue
        if "contracts" not in stage_module_names(stage):
            missing.append(f"{stage}: no contracts.py")
    assert not missing, "declaration modules vanished:\n" + "\n".join(missing)


def test_contract_modules_import_only_declarations() -> None:
    violations: list[str] = []
    for stage in ALLOWED_DEPENDENCIES:
        path = PACKAGE_ROOT / stage / "contracts.py"
        if not path.exists():
            continue
        for module in sorted(imported_modules(path)):
            parts = module.split(".")
            if len(parts) < 3 or parts[0] != "text2motion":
                continue
            imported_stage = parts[1]
            role = parts[2]
            if imported_stage not in ALLOWED_DEPENDENCIES:
                continue
            if role not in stage_module_names(imported_stage):
                violations.append(f"{path}: imports missing module {module}")
                continue
            allowed = (
                OWN_STAGE_CONTRACT_SAFE_ROLES
                if imported_stage == stage
                else CONTRACT_SAFE_ROLES
            )
            if role not in allowed:
                violations.append(
                    f"{path}: imports operation {module}; allowed roles {sorted(allowed)}"
                )
    assert not violations, "contract modules depend on operations:\n" + "\n".join(violations)
