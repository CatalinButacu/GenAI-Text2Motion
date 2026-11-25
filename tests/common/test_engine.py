from contextlib import contextmanager

import pytest

from text2motion.common.engine import CleanCrossEntropyProgress, restore_clean_ce, run_epochs


def test_run_epochs_stops_on_the_first_reason():
    seen = []

    @contextmanager
    def guard(epoch):
        yield

    run_epochs(
        2,
        10,
        lambda epoch: {"epoch": float(epoch)},
        lambda epoch, means: seen.append(epoch) or ("stop" if epoch == 4 else ""),
        guard,
    )

    assert seen == [2, 3, 4]


def test_restore_keeps_the_saved_counter_and_fresh_selection_resets_it():
    state = {"clean_ce_progress": {"best": 1.5, "stale": 3}}

    kept = restore_clean_ce(state, 8, False, 10)
    reset = restore_clean_ce(state, 8, True, 10)

    assert kept == CleanCrossEntropyProgress(best=1.5, stale=3)
    assert reset == CleanCrossEntropyProgress(best=float("inf"), stale=0)


def test_observe_counts_ties_as_stale_and_rejects_nan():
    progress = CleanCrossEntropyProgress(best=2.0, stale=0)

    assert progress.observe(2.0).stale == 1
    assert progress.observe(1.0) == CleanCrossEntropyProgress(best=1.0, stale=0)
    with pytest.raises(ValueError):
        progress.observe(float("nan"))
