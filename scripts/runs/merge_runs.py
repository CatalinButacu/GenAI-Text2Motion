# .\.venv\Scripts\python.exe scripts/runs/merge_runs.py --apply
from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

RUN_FOLDERS = (
    Path("artifacts/logs/train"),
    Path("artifacts/outputs/runs"),
    Path("artifacts/outputs/runs/generator"),
    Path("artifacts/outputs/runs/tokenizer"),
    Path("artifacts/outputs/runs/evaluate"),
)
CHECKPOINT_KEYS = ("ckpt_name", "out", "out_path", "ckpt", "name")
SEGMENTS_FOLDER = "segments"
LINEAGE_FILE = "lineage.json"


@dataclass(frozen=True)
class RunSegment:
    directory: Path
    manifest: dict
    rows: tuple[dict, ...]

    @property
    def extra(self) -> dict:
        return self.manifest.get("extra") or {}

    @property
    def name(self) -> str:
        return str(self.manifest["name"])

    @property
    def started(self) -> str:
        return str(self.manifest["started_utc"])

    @property
    def config(self) -> str:
        return Path(str(self.extra.get("config") or "")).name

    @property
    def checkpoint(self) -> str:
        for key in CHECKPOINT_KEYS:
            value = self.extra.get(key)
            if value:
                return Path(str(value)).name
        return ""

    @property
    def tokenizer(self) -> str:
        return Path(str(self.extra.get("tokenizer_ckpt") or "")).name

    @property
    def resumed(self) -> bool:
        return bool(self.extra.get("resume"))

    @property
    def already_merged(self) -> bool:
        return (self.directory / LINEAGE_FILE).is_file()

    @property
    def target_epochs(self) -> int | None:
        if self.already_merged:
            lineage_text = (self.directory / LINEAGE_FILE).read_text(encoding="utf-8")
            return json.loads(lineage_text)["target_epochs"]
        return self.extra.get("epochs")

    @property
    def epochs(self) -> list[int]:
        return [int(row["epoch"]) for row in self.rows if "epoch" in row]

    @property
    def first_epoch(self) -> int:
        return min(self.epochs)

    @property
    def last_epoch(self) -> int:
        return max(self.epochs)


@dataclass
class RunChain:
    segments: list[RunSegment] = field(default_factory=list)

    @property
    def trained_segments(self) -> list[RunSegment]:
        return [segment for segment in self.segments if segment.epochs]

    def kept_rows(self, segment: RunSegment) -> list[dict]:
        trained = self.trained_segments
        if segment not in trained:
            return []
        later = trained[trained.index(segment) + 1 :]
        if not later:
            return list(segment.rows)
        cutoff = min(other.first_epoch for other in later)
        kept_epoch_rows = [row for row in segment.rows if "epoch" in row and row["epoch"] < cutoff]
        if not kept_epoch_rows:
            return []
        return [row for row in segment.rows if "epoch" not in row or row["epoch"] < cutoff]

    @property
    def contributing_segments(self) -> list[RunSegment]:
        return [segment for segment in self.segments if self.kept_rows(segment)]

    @property
    def home(self) -> RunSegment:
        contributing = self.contributing_segments
        return contributing[0] if contributing else self.segments[0]

    @property
    def merged_rows(self) -> list[dict]:
        merged: list[dict] = []
        for segment in self.segments:
            merged.extend(self.kept_rows(segment))
        return merged

    @property
    def merged_epochs(self) -> list[int]:
        return [int(row["epoch"]) for row in self.merged_rows if "epoch" in row]

    @property
    def last_epoch(self) -> int:
        return max(self.merged_epochs)

    @property
    def last_trained(self) -> RunSegment:
        return self.trained_segments[-1]

    @property
    def complete(self) -> bool:
        target = self.last_trained.target_epochs
        return target is not None and self.last_epoch == target

    def role(self, segment: RunSegment) -> str:
        if not segment.epochs:
            return "empty"
        if not self.kept_rows(segment):
            return "superseded"
        return "home" if segment is self.home else "continuation"


def load_segment(directory: Path) -> RunSegment:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    metrics_path = directory / "metrics.jsonl"
    rows: list[dict] = []
    if metrics_path.is_file():
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return RunSegment(directory, manifest, tuple(rows))


def read_segments(folder: Path) -> list[RunSegment]:
    if not folder.is_dir():
        raise FileNotFoundError(f"run folder not found: {folder}")
    segments: list[RunSegment] = []
    for directory in sorted(folder.iterdir()):
        if directory.is_dir() and (directory / "manifest.json").is_file():
            segments.append(load_segment(directory))
    return segments


def continuation_target(chains: list[RunChain], segment: RunSegment) -> RunChain | None:
    if not segment.resumed or segment.already_merged:
        return None
    same_checkpoint = [
        chain
        for chain in chains
        if chain.last_trained.checkpoint == segment.checkpoint
        and segment.first_epoch <= chain.last_trained.last_epoch + 1
    ]
    if same_checkpoint:
        return same_checkpoint[-1]
    contiguous = [
        chain
        for chain in chains
        if chain.last_trained.last_epoch + 1 == segment.first_epoch
        and chain.last_trained.tokenizer == segment.tokenizer
    ]
    if len(contiguous) > 1:
        candidates = [chain.home.directory.name for chain in contiguous]
        raise RuntimeError(f"{segment.directory.name} continues more than one chain: {candidates}")
    return contiguous[0] if contiguous else None


def build_chains(segments: list[RunSegment]) -> list[RunChain]:
    groups: dict[tuple, list[RunSegment]] = defaultdict(list)
    for segment in segments:
        groups[(segment.directory.parent, segment.name, segment.config)].append(segment)
    all_chains: list[RunChain] = []
    for group in groups.values():
        ordered = sorted(group, key=lambda segment: segment.started)
        chains: list[RunChain] = []
        for segment in ordered:
            if not segment.epochs:
                continue
            target = continuation_target(chains, segment)
            if target is None:
                chains.append(RunChain([segment]))
            else:
                target.segments.append(segment)
        for segment in ordered:
            if segment.epochs:
                continue
            owners = [
                chain
                for chain in chains
                if chain.trained_segments and chain.last_trained.checkpoint == segment.checkpoint
            ]
            if not owners or segment.target_epochs is None:
                chains.append(RunChain([segment]))
                continue
            same_target = [
                chain
                for chain in owners
                if chain.last_trained.target_epochs == segment.target_epochs
            ]
            candidates = same_target or owners
            launched_after = [
                chain for chain in candidates if chain.trained_segments[0].started > segment.started
            ]
            owner = launched_after[0] if launched_after else candidates[-1]
            owner.segments.append(segment)
        for chain in chains:
            chain.segments.sort(key=lambda segment: segment.started)
        all_chains.extend(chains)
    return sorted(all_chains, key=lambda chain: chain.segments[0].started)


def parent_of(chain: RunChain, chains: list[RunChain]) -> dict | None:
    if not chain.trained_segments:
        return None
    first_epoch = min(chain.merged_epochs)
    if first_epoch <= 1:
        return None
    home = chain.home
    candidates = [
        other
        for other in chains
        if other is not chain
        and other.trained_segments
        and other.home.directory.parent == home.directory.parent
        and other.home.name == home.name
        and other.home.tokenizer == home.tokenizer
        and first_epoch - 1 in other.merged_epochs
    ]
    if len(candidates) != 1:
        return {"run": None, "epoch": first_epoch - 1, "candidates": len(candidates)}
    return {"run": candidates[0].home.directory.name, "epoch": first_epoch - 1, "candidates": 1}


def lineage_of(chain: RunChain, chains: list[RunChain]) -> dict:
    segment_records = []
    for segment in chain.segments:
        kept_epochs = [int(row["epoch"]) for row in chain.kept_rows(segment) if "epoch" in row]
        segment_records.append(
            {
                "run": segment.directory.name,
                "role": chain.role(segment),
                "resumed": segment.resumed,
                "checkpoint": segment.checkpoint,
                "target_epochs": segment.target_epochs,
                "logged_epochs": [segment.first_epoch, segment.last_epoch]
                if segment.epochs
                else None,
                "kept_epochs": [min(kept_epochs), max(kept_epochs)] if kept_epochs else None,
                "rows_logged": len(segment.rows),
                "rows_kept": len(chain.kept_rows(segment)),
            }
        )
    trained = bool(chain.trained_segments)
    return {
        "home": chain.home.directory.name,
        "name": chain.home.name,
        "config": chain.home.config,
        "epochs": [min(chain.merged_epochs), chain.last_epoch] if trained else None,
        "target_epochs": chain.last_trained.target_epochs if trained else None,
        "complete": chain.complete if trained else False,
        "parent": parent_of(chain, chains),
        "segments": segment_records,
    }


def describe(chain: RunChain, chains: list[RunChain]) -> str:
    lineage = lineage_of(chain, chains)
    if lineage["epochs"] is None:
        status = "ONE-SHOT" if chain.home.target_epochs is None else "EMPTY"
        span = "-"
    else:
        status = "complete" if lineage["complete"] else "INCOMPLETE"
        span = f"{lineage['epochs'][0]}-{lineage['epochs'][1]}/{lineage['target_epochs']}"
    action = "MERGE" if len(chain.segments) > 1 else "keep"
    header = (
        f"{action:5s} {status:10s} {span:12s} {chain.home.directory.parent.as_posix()}/"
        f"{lineage['home']}  [{chain.home.checkpoint or lineage['config']}]"
    )
    lines = [header]
    parent = lineage["parent"]
    if parent is not None and parent["run"] is not None:
        lines.append(f"        parent: {parent['run']} @ epoch {parent['epoch']}")
    if parent is not None and parent["run"] is None:
        lines.append(
            f"        parent: UNRESOLVED @ epoch {parent['epoch']} ({parent['candidates']} candidates)"
        )
    if len(chain.segments) > 1:
        for record in lineage["segments"]:
            logged = record["logged_epochs"]
            kept = record["kept_epochs"]
            logged_text = f"{logged[0]}-{logged[1]}" if logged else "none"
            kept_text = f"{kept[0]}-{kept[1]}" if kept else "none"
            lines.append(
                f"        {record['role']:12s} {record['run']}  logged {logged_text:8s} kept {kept_text:8s}"
                f" [{record['checkpoint']}]"
            )
    return "\n".join(lines)


def apply_merge(chain: RunChain, chains: list[RunChain]) -> None:
    home = chain.home
    segments_dir = home.directory / SEGMENTS_FOLDER
    if segments_dir.exists():
        raise FileExistsError(f"already merged: {segments_dir}")
    lineage = lineage_of(chain, chains)
    merged_rows = chain.merged_rows
    later_manifests = [
        segment.manifest for segment in chain.segments if segment.started > home.started
    ]
    existing_resumes = home.directory / "resumes.jsonl"
    if existing_resumes.exists():
        raise FileExistsError(f"home already carries resumes: {existing_resumes}")

    home_copy = segments_dir / home.directory.name
    home_copy.mkdir(parents=True)
    for file_name in ("manifest.json", "metrics.jsonl"):
        shutil.copy2(home.directory / file_name, home_copy / file_name)
    for segment in chain.segments:
        if segment is home:
            continue
        shutil.move(str(segment.directory), str(segments_dir / segment.directory.name))

    metrics_text = "".join(json.dumps(row) + "\n" for row in merged_rows)
    (home.directory / "metrics.jsonl").write_text(metrics_text, encoding="utf-8")
    resumes_text = "".join(json.dumps(manifest) + "\n" for manifest in later_manifests)
    if resumes_text:
        existing_resumes.write_text(resumes_text, encoding="utf-8")
    (home.directory / LINEAGE_FILE).write_text(json.dumps(lineage, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    segments: list[RunSegment] = []
    for folder in RUN_FOLDERS:
        segments.extend(read_segments(folder))
    chains = build_chains(segments)

    one_shots = [
        chain for chain in chains if not chain.trained_segments and chain.home.target_epochs is None
    ]
    training_chains = [chain for chain in chains if chain not in one_shots]
    for chain in training_chains:
        print(describe(chain, chains))
    mergeable = [chain for chain in training_chains if len(chain.segments) > 1]
    print(
        f"\n{len(segments)} run directories: {len(training_chains)} training chains "
        f"({len(mergeable)} to merge), {len(one_shots)} one-shot runs left untouched"
    )

    if not args.apply:
        print("dry run: pass --apply to merge")
        return
    for chain in mergeable:
        apply_merge(chain, chains)
    print(f"merged {len(mergeable)} chains")


if __name__ == "__main__":
    main()
