import importlib.util
import math
import sys
from pathlib import Path
from types import MappingProxyType

VERDICT_SCRIPT = Path("scripts/qa/a7_screen_verdict.py")
CONTROL = MappingProxyType(
    {
        1: {"val_ce": 6.0, "ce_gap": 0.0},
        2: {"val_ce": 5.0, "ce_gap": 0.1},
        3: {"fid": 1.0},
        4: {"val_ce": 4.0, "ce_gap": 0.3},
        5: {"val_ce": 4.2, "ce_gap": 0.9},
        6: {"val_ce": 3.0, "ce_gap": 2.0},
    }
)


def verdict_module(monkeypatch):
    spec = importlib.util.spec_from_file_location("a7_screen_verdict", VERDICT_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_the_control_gap_is_read_at_the_same_val_ce_on_the_falling_curve(monkeypatch):
    verdict = verdict_module(monkeypatch)
    assert math.isclose(verdict.matched_control_gap(CONTROL, 4.5), 0.2)
    assert math.isclose(verdict.matched_control_gap(CONTROL, 4.0), 0.3)


def test_a_val_ce_outside_the_falling_curve_has_no_control_gap(monkeypatch):
    verdict = verdict_module(monkeypatch)
    assert math.isnan(verdict.matched_control_gap(CONTROL, 3.5))
    assert math.isnan(verdict.matched_control_gap(CONTROL, 6.5))


def extend_curve(fid: float, r1: float) -> dict[int, dict[str, float]]:
    return {epoch: {"fid": fid, "r_top1": r1} for epoch in (8, 10, 12)}


def test_the_extension_passes_only_with_a_lower_fid_and_a_kept_r1(monkeypatch):
    verdict = verdict_module(monkeypatch)
    cases = (((0.79, 0.40), True), ((0.81, 0.40), False), ((0.5, 0.37), False))
    for (fid, r1), expected in cases:
        curves = {
            verdict.CONTROL: extend_curve(1.0, 0.40),
            verdict.EXTEND_ARM: extend_curve(fid, r1),
        }
        monkeypatch.setattr(verdict, "run_curve", lambda backbone, tag, curves=curves: curves[tag])
        assert verdict.extend_row("mamba")["pass"] is expected


def test_the_patience_stop_counts_ties_as_stale_epochs(monkeypatch):
    verdict = verdict_module(monkeypatch)
    values = (5.0, 4.0, 4.0, 4.1, 3.9, 3.9, 3.9, 4.0, 3.0)
    curve = {epoch: {"val_ce": value} for epoch, value in enumerate(values, start=1)}
    assert verdict.stop_epoch_at(curve, 3) == 8
    assert verdict.stop_epoch_at(curve, 4) is None


def kill_module(monkeypatch):
    monkeypatch.syspath_prepend(str(VERDICT_SCRIPT.parent))
    spec = importlib.util.spec_from_file_location(
        "a8_kill_tests", VERDICT_SCRIPT.parent / "a8_kill_tests.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_kt2_kills_an_arm_with_a_worse_val_ce_or_a_lost_r1(monkeypatch):
    kill = kill_module(monkeypatch)
    control = {2: {"val_ce": 5.0, "r_top1": 0.30}}
    cases = (((5.09, 0.26), True), ((5.11, 0.30), False), ((5.0, 0.24), False))
    for (val_ce, r1), expected in cases:
        curves = {kill.CONTROL: control, "a8_f1": {2: {"val_ce": val_ce, "r_top1": r1}}}
        monkeypatch.setattr(kill, "run_curve", lambda backbone, tag, curves=curves: curves[tag])
        assert kill.arm_survives("a8_f1", "mamba") is expected


def test_kt1_needs_a_falling_unconditional_val_ce(monkeypatch):
    kill = kill_module(monkeypatch)
    for second, expected in ((5.94, True), (5.96, False)):
        curve = {1: {"val_ce": 6.0}, 2: {"val_ce": second}}
        monkeypatch.setattr(kill, "run_curve", lambda backbone, tag, curve=curve: curve)
        assert kill.prior_learns("mamba") is expected
