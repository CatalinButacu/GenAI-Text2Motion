# python scripts/maintenance/backfill_mlflow.py
from __future__ import annotations

import json
from pathlib import Path

import mlflow

TRAIN_LOGS = Path("runs/logs/train")
STORE = Path("runs/logs/mlflow.db")
EXPERIMENT = "tokenizer"


def numeric_items(record: dict) -> dict[str, float]:
    prefix = str(record["split"]) + "_" if "split" in record else ""
    skipped = {"logged_utc", "epoch", "split"}
    return {
        prefix + key: float(value)
        for key, value in record.items()
        if key not in skipped and isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def run_params(folder: Path) -> dict[str, str]:
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    config = manifest.get("config", {})
    return {
        "run_name": manifest["name"],
        "git_commit": str(manifest.get("git_commit", "")),
        "seed": str(config.get("seed", "")),
        "precision": str(config.get("precision", "")),
        "started_utc": str(manifest.get("started_utc", "")),
    }


def backfill(folder: Path) -> int:
    records = [
        json.loads(line)
        for line in (folder / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    with mlflow.start_run(run_name=folder.name):
        mlflow.log_params(run_params(folder))
        for record in records:
            mlflow.log_metrics(numeric_items(record), step=int(record["epoch"]))
    return len(records)


def main() -> None:
    STORE.parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{STORE.resolve().as_posix()}")
    mlflow.set_experiment(EXPERIMENT)
    folders = sorted(
        folder
        for folder in TRAIN_LOGS.iterdir()
        if folder.is_dir()
        and (folder / "metrics.jsonl").is_file()
        and (folder / "manifest.json").is_file()
        and "_tokenizer_" in folder.name
    )
    logged = set(mlflow.search_runs(experiment_names=[EXPERIMENT])["tags.mlflow.runName"].dropna())
    pending = [folder for folder in folders if folder.name not in logged]
    total = 0
    for folder in pending:
        total += backfill(folder)
    print(
        f"runs found {len(folders)}  already logged {len(folders) - len(pending)}  new {len(pending)}  metric records {total}  store {STORE}"
    )


if __name__ == "__main__":
    main()
