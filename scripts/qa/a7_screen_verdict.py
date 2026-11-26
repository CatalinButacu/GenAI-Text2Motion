# python scripts/qa/a7_screen_verdict.py verdict --out docs/paper/pics/a7_screen_validation.csv
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from text2motion.common.engine import CleanCrossEntropyProgress

LOGS = Path("runs/logs/train")
BACKBONES = ("transformer", "mamba")
CONTROL = "p1"
REPLAY = "a7_replay"
ARMS = ("a7_step50", "a7_clipfrozen")
STOP_EPOCH = 6
LAG_EPOCH = 4
GAP_RATIO = 0.7
REPLAY_TOLERANCE = 0.005
EXTEND_ARM = "a7_step50"
EXTEND_EPOCHS = (8, 10, 12)
FID_RATIO = 0.8
R1_MARGIN = 0.02
CONTROL_PATIENCE = 3
FIELDS = ("train_ce_clean", "val_ce", "ce_gap", "fid", "r_top1")


def run_curve(backbone: str, tag: str) -> dict[int, dict[str, float]]:
    name = f"generator_{backbone}_w320_{tag}.pt"
    folders = [
        folder
        for folder in sorted(LOGS.glob(f"*_generator_{backbone}"))
        if json.loads((folder / "manifest.json").read_text(encoding="utf-8"))["extra"].get(
            "ckpt_name"
        )
        == name
    ]
    if len(folders) != 1:
        raise RuntimeError(f"{len(folders)} run folders for {name} in {LOGS}")
    epochs: dict[int, dict[str, float]] = {}
    for line in (folders[0] / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if "epoch" not in record or record.get("split", "val") != "val":
            continue
        values = {key: float(record[key]) for key in FIELDS if key in record}
        if "val_ce" in values or "fid" in values:
            epochs.setdefault(int(record["epoch"]), {}).update(values)
    return epochs


def matched_control_gap(control: dict[int, dict[str, float]], val_ce: float) -> float:
    falling = []
    best = float("inf")
    for epoch in sorted(control):
        if "val_ce" not in control[epoch]:
            continue
        if control[epoch]["val_ce"] >= best:
            break
        best = control[epoch]["val_ce"]
        falling.append((control[epoch]["val_ce"], control[epoch]["ce_gap"]))
    points = sorted(falling)
    if not points[0][0] <= val_ce <= points[-1][0]:
        return float("nan")
    return float(np.interp(val_ce, [p[0] for p in points], [p[1] for p in points]))


def replay_check() -> bool:
    agree = True
    for backbone in BACKBONES:
        control = run_curve(backbone, CONTROL)[1]
        replay = run_curve(backbone, REPLAY)[1]
        for key in ("train_ce_clean", "val_ce"):
            delta = abs(replay[key] - control[key])
            agree &= delta <= REPLAY_TOLERANCE
            print(
                f"{backbone} epoch 1 {key}: P1 {control[key]:.4f} replay {replay[key]:.4f} delta {delta:.4f}"
            )
    print("REPLAY MATCH" if agree else "REPLAY MISMATCH")
    return agree


def arm_row(arm: str, backbone: str) -> dict[str, object]:
    control = run_curve(backbone, CONTROL)
    run = run_curve(backbone, arm)
    final = run[STOP_EPOCH]
    matched = matched_control_gap(control, final["val_ce"])
    ratio = final["ce_gap"] / matched if np.isfinite(matched) and matched > 0 else float("nan")
    lag_ok = final["val_ce"] <= control[LAG_EPOCH]["val_ce"]
    return {
        "arm": arm,
        "backbone": backbone,
        "val_ce": final["val_ce"],
        "ce_gap": final["ce_gap"],
        "control_gap_at_same_val_ce": matched,
        "gap_ratio": ratio,
        "control_val_ce_epoch4": control[LAG_EPOCH]["val_ce"],
        "lag_ok": lag_ok,
        "val_fid": final.get("fid", float("nan")),
        "control_val_fid": control[STOP_EPOCH].get("fid", float("nan")),
        "val_r1": final.get("r_top1", float("nan")),
        "control_val_r1": control[STOP_EPOCH].get("r_top1", float("nan")),
        "pass": bool(np.isfinite(ratio) and ratio <= GAP_RATIO and lag_ok),
    }


def mean_over(curve: dict[int, dict[str, float]], key: str) -> float:
    return float(np.mean([curve[epoch][key] for epoch in EXTEND_EPOCHS]))


def extend_row(backbone: str) -> dict[str, object]:
    control = run_curve(backbone, CONTROL)
    run = run_curve(backbone, EXTEND_ARM)
    fid, control_fid = mean_over(run, "fid"), mean_over(control, "fid")
    r1, control_r1 = mean_over(run, "r_top1"), mean_over(control, "r_top1")
    return {
        "arm": EXTEND_ARM,
        "backbone": backbone,
        "epochs": "/".join(map(str, EXTEND_EPOCHS)),
        "val_fid_mean": fid,
        "control_val_fid_mean": control_fid,
        "fid_ratio": fid / control_fid,
        "val_r1_mean": r1,
        "control_val_r1_mean": control_r1,
        "pass": fid <= FID_RATIO * control_fid and r1 >= control_r1 - R1_MARGIN,
    }


def stop_epoch_at(curve: dict[int, dict[str, float]], patience: int) -> int | None:
    progress = CleanCrossEntropyProgress(best=float("inf"), stale=0)
    epochs = sorted(epoch for epoch in curve if "val_ce" in curve[epoch])
    for epoch in epochs:
        progress = progress.observe(curve[epoch]["val_ce"])
        if progress.stale >= patience:
            return epoch
    return None


def best_epoch(curve: dict[int, dict[str, float]], key: str, last: int) -> int:
    return min(
        (epoch for epoch in curve if key in curve[epoch] and epoch <= last),
        key=lambda e: curve[e][key],
    )


def final_row(backbone: str) -> dict[str, object]:
    control = run_curve(backbone, CONTROL)
    run = run_curve(backbone, EXTEND_ARM)
    control_last = max(control)
    stop = stop_epoch_at(run, CONTROL_PATIENCE)
    window = max(run) if stop is None else stop
    fid_epoch = best_epoch(run, "fid", window)
    control_fid_epoch = best_epoch(control, "fid", control_last)
    ce_epoch = best_epoch(run, "val_ce", window)
    control_ce_epoch = best_epoch(control, "val_ce", control_last)
    full_fid_epoch = best_epoch(run, "fid", max(run))
    return {
        "arm": EXTEND_ARM,
        "backbone": backbone,
        "patience3_stop_reached": stop is not None,
        "patience3_stop_epoch": window,
        "best_val_fid": run[fid_epoch]["fid"],
        "best_val_fid_epoch": fid_epoch,
        "best_val_fid_r1": run[fid_epoch]["r_top1"],
        "control_best_val_fid": control[control_fid_epoch]["fid"],
        "control_best_val_fid_epoch": control_fid_epoch,
        "control_best_val_fid_r1": control[control_fid_epoch]["r_top1"],
        "best_val_ce_epoch": ce_epoch,
        "control_best_val_ce_epoch": control_ce_epoch,
        "patience6_last_epoch": max(run),
        "patience6_best_val_fid": run[full_fid_epoch]["fid"],
        "patience6_best_val_fid_epoch": full_fid_epoch,
        "pass": run[fid_epoch]["fid"] < control[control_fid_epoch]["fid"]
        and ce_epoch > control_ce_epoch,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("replay", "verdict", "extend", "final"))
    parser.add_argument(
        "--out", type=Path, default=Path("docs/paper/pics/a7_screen_validation.csv")
    )
    arguments = parser.parse_args()
    if arguments.mode == "replay":
        raise SystemExit(0 if replay_check() else 1)
    if arguments.mode == "extend":
        rows = [extend_row(backbone) for backbone in BACKBONES]
    elif arguments.mode == "final":
        rows = [final_row(backbone) for backbone in BACKBONES]
    else:
        rows = [arm_row(arm, backbone) for arm in ARMS for backbone in BACKBONES]
    with arguments.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)
    for arm in sorted({str(row["arm"]) for row in rows}):
        verdict = all(row["pass"] for row in rows if row["arm"] == arm)
        print(f"{arm}: {'PASS' if verdict else 'FAIL'}")


if __name__ == "__main__":
    main()
