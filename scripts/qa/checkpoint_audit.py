# .\.venv\Scripts\python.exe scripts/qa/checkpoint_audit.py
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml

from text2motion.app.bootstrap import ApplicationBootstrap
from text2motion.app.checkpoint import GeneratorBundle
from text2motion.app.config import load_config
from text2motion.motion.storage import sha256_file
from text2motion.tokenization.contracts import TokenizerKind
from text2motion.tokenization.nn_model import build_tokenizer_network

TOKENIZER_DIR = Path("artifacts/checkpoints/tokenizer")
TOKENIZER_CONFIG_DIR = Path("configs/tokenizer")
GENERATOR_DIR = Path("artifacts/checkpoints/generator")
DEMO_REGISTRY = Path("configs/demo_models.yaml")
BUNDLE_FORMAT = "text2motion.generator_bundle"


############################################################
#       AuditRow
############################################################
@dataclass(frozen=True)
class AuditRow:
    checkpoint: str
    result: str
    detail: str


def __stored_kind(state: dict[str, torch.Tensor]) -> TokenizerKind | None:
    if any(name.endswith(".levels") for name in state):
        return TokenizerKind.FINITE_SCALAR
    if any(name.endswith(".embed") for name in state):
        return TokenizerKind.RESIDUAL_VECTOR
    return None


def __stored_levels(state: dict[str, torch.Tensor]) -> set[tuple[int, ...]]:
    return {
        tuple(int(level) for level in value.tolist())
        for name, value in state.items()
        if name.endswith(".levels")
    }


def __strict_load_error(config_path: Path, state: dict[str, torch.Tensor]) -> str | None:
    config = load_config(config_path)
    network = build_tokenizer_network(config.tokenizer, config.rvq_baseline)
    try:
        network.load_state_dict(state)
    except RuntimeError as error:
        return str(error).splitlines()[0]
    return None


def __tokenizer_mismatch(config_path: Path, state: dict[str, torch.Tensor]) -> str | None:
    config = load_config(config_path).tokenizer
    stored_kind = __stored_kind(state)
    if stored_kind != config.kind:
        return f"the checkpoint holds {stored_kind}, the YAML says {config.kind}"
    if stored_kind is TokenizerKind.FINITE_SCALAR:
        levels = __stored_levels(state)
        if levels != {tuple(config.fsq_levels)}:
            return f"the checkpoint levels are {sorted(levels)}, the YAML says {config.fsq_levels}"
    return __strict_load_error(config_path, state)


def audit_tokenizer(checkpoint: Path) -> AuditRow:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config_path = TOKENIZER_CONFIG_DIR / f"{checkpoint.stem}.yaml"
    stored = f"{__stored_kind(state)} levels {sorted(__stored_levels(state))}"
    if not config_path.is_file():
        return AuditRow(checkpoint.as_posix(), "NO_CONFIG", stored)
    mismatch = __tokenizer_mismatch(config_path, state)
    if mismatch is not None:
        return AuditRow(checkpoint.as_posix(), "MISMATCH", mismatch)
    return AuditRow(checkpoint.as_posix(), "OK", f"{stored}; {config_path.as_posix()}")


def __tokenizer_names_by_hash() -> dict[str, str]:
    return {sha256_file(path): path.name for path in sorted(TOKENIZER_DIR.glob("*.pt"))}


def __generator_format(payload: object) -> str:
    if isinstance(payload, dict) and payload.get("format") == BUNDLE_FORMAT:
        return "bundle"
    if isinstance(payload, dict) and "generator" in payload:
        return "resume"
    return "raw"


def audit_generator(checkpoint: Path, tokenizer_names: dict[str, str]) -> AuditRow:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    kind = __generator_format(payload)
    if kind != "bundle":
        return AuditRow(checkpoint.as_posix(), "LEGACY", f"{kind}: no config, no tokenizer check")
    bundle = GeneratorBundle.from_payload(payload)
    tokenizer = tokenizer_names.get(bundle.tokenizer_sha256)
    if tokenizer is None:
        return AuditRow(
            checkpoint.as_posix(),
            "MISMATCH",
            f"no tokenizer file has the recorded hash ({bundle.tokenizer_path})",
        )
    return AuditRow(checkpoint.as_posix(), "OK", f"bundle; tokenizer {tokenizer}")


def __spec_difference(
    bundle: GeneratorBundle, config_path: Path, tokenizer_path: Path
) -> list[str]:
    app = ApplicationBootstrap.from_config(config_path, device="cpu")
    config = app.config
    codebook_size = build_tokenizer_network(config.tokenizer, config.rvq_baseline).codebook_size
    backbone = bundle.resolved_generator_config["backbone"]
    if sha256_file(tokenizer_path) != bundle.tokenizer_sha256:
        return [f"tokenizer {tokenizer_path.name} is not the recorded tokenizer"]
    try:
        bundle.verify_generator_config(
            app.resolve_generator_spec(backbone, codebook_size).as_dict()
        )
    except ValueError as error:
        return [str(error)]
    return []


def audit_demo_entry(entry: dict[str, str], defaults: dict[str, str]) -> AuditRow:
    checkpoint = Path(entry["ckpt"])
    label = f"demo {entry['name']} {entry['version']}"
    if not checkpoint.is_file():
        return AuditRow(label, "MISSING", checkpoint.as_posix())
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    kind = __generator_format(payload)
    if kind != "bundle":
        return AuditRow(
            label, "LEGACY", f"{checkpoint.name} is a {kind} file: live weights, no checks"
        )
    config_path = Path(entry.get("config", defaults["config"]))
    tokenizer_path = Path(entry.get("tokenizer_ckpt", defaults["tokenizer_ckpt"]))
    differences = __spec_difference(
        GeneratorBundle.from_payload(payload), config_path, tokenizer_path
    )
    if differences:
        return AuditRow(label, "MISMATCH", "; ".join(differences))
    return AuditRow(label, "OK", f"{checkpoint.name} matches {config_path.as_posix()}")


def audit_rows() -> list[AuditRow]:
    rows = [audit_tokenizer(path) for path in sorted(TOKENIZER_DIR.glob("*.pt"))]
    tokenizer_names = __tokenizer_names_by_hash()
    rows += [audit_generator(path, tokenizer_names) for path in sorted(GENERATOR_DIR.glob("*.pt"))]
    registry = yaml.safe_load(DEMO_REGISTRY.read_text(encoding="utf-8"))
    rows += [audit_demo_entry(entry, registry["defaults"]) for entry in registry["models"]]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="compare every checkpoint with its own contents")
    parser.parse_args()
    rows = audit_rows()
    print("| checkpoint | result | detail |")
    print("|---|---|---|")
    for row in rows:
        print(f"| {row.checkpoint} | {row.result} | {row.detail} |")
    failed = [row for row in rows if row.result in ("MISMATCH", "MISSING")]
    print(f"\n{len(rows)} rows, {len(failed)} MISMATCH or MISSING")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
