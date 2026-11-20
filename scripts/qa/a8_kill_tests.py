# python scripts/qa/a8_kill_tests.py kt1
from __future__ import annotations

import argparse

from a7_screen_verdict import BACKBONES, CONTROL, run_curve

PRIOR = "a8_prior"
ARMS = ("a8_f0", "a8_f1", "a8_f2")
KT_EPOCH = 2
KT1_DROP = 0.05
KT2_CE_MARGIN = 0.10
KT2_R1_MARGIN = 0.05


def prior_learns(backbone: str) -> bool:
    curve = run_curve(backbone, PRIOR)
    drop = curve[1]["val_ce"] - curve[KT_EPOCH]["val_ce"]
    print(
        f"KT1 {backbone}: unconditional val CE {curve[1]['val_ce']:.4f} -> {curve[KT_EPOCH]['val_ce']:.4f}, drop {drop:.4f}"
    )
    return drop >= KT1_DROP


def arm_survives(arm: str, backbone: str) -> bool:
    control = run_curve(backbone, CONTROL)[KT_EPOCH]
    run = run_curve(backbone, arm)[KT_EPOCH]
    ce_ok = run["val_ce"] <= control["val_ce"] + KT2_CE_MARGIN
    r1_ok = run["r_top1"] >= control["r_top1"] - KT2_R1_MARGIN
    print(
        f"KT2 {arm} {backbone}: val CE {run['val_ce']:.4f} (P1 {control['val_ce']:.4f}), "
        f"R@1 {run['r_top1']:.3f} (P1 {control['r_top1']:.3f})"
    )
    return ce_ok and r1_ok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("test", choices=("kt1", "kt2"))
    arguments = parser.parse_args()
    if arguments.test == "kt1":
        learns = [prior_learns(backbone) for backbone in BACKBONES]
        print("KT1 PASS" if all(learns) else "KT1 FAIL")
        raise SystemExit(0 if all(learns) else 1)
    for arm in ARMS:
        alive = [arm_survives(arm, backbone) for backbone in BACKBONES]
        print(f"{arm}: {'SURVIVES' if all(alive) else 'KILLED'}")


if __name__ == "__main__":
    main()
