from dataclasses import replace

import pytest

from text2motion.generation.contracts import Backbone, GeneratorConfig
from text2motion.generation.model.spec import (
    TWIN_PARAMETER_TOLERANCE,
    check_twin_parameter_match,
    twin_parameter_counts,
)

CONFIG = GeneratorConfig()
CODEBOOK_SIZE = 1000
NUM_CODEBOOKS = 6


def test_counts_cover_both_backbones():
    counts = twin_parameter_counts(CONFIG, CODEBOOK_SIZE, NUM_CODEBOOKS)

    assert set(counts) == set(Backbone)
    assert all(count > 0 for count in counts.values())


def test_matched_twins_pass_and_return_counts():
    counts = check_twin_parameter_match(CONFIG, CODEBOOK_SIZE, NUM_CODEBOOKS)
    gap = abs(counts[Backbone.MAMBA] - counts[Backbone.TRANSFORMER]) / min(counts.values())

    assert gap <= TWIN_PARAMETER_TOLERANCE


def test_unmatched_twins_fail_loud():
    unmatched = replace(CONFIG, mamba_n_layers=CONFIG.mamba_n_layers + 4)

    with pytest.raises(ValueError, match="twin parameter counts differ"):
        check_twin_parameter_match(unmatched, CODEBOOK_SIZE, NUM_CODEBOOKS)
