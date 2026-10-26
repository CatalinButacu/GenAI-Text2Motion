from pathlib import Path

import pytest

from text2motion.app.config import load_config

EXPERIMENT_CONFIGS = sorted(
    [Path("configs/default.yaml"), Path("configs/aws.yaml")]
    + list(Path("configs/generator").glob("*.yaml"))
    + list(Path("configs/tokenizer").glob("*.yaml"))
)


def test_a_misspelled_root_block_is_refused_with_the_nearest_key(tmp_path):
    config = tmp_path / "typo.yaml"
    config.write_text("generatr:\n  d_model: 768\n", encoding="utf-8")

    with pytest.raises(ValueError, match="'generatr' \\(did you mean 'generator'\\?\\)"):
        load_config(config)


def test_the_legacy_representation_block_is_still_accepted(tmp_path):
    config = tmp_path / "legacy.yaml"
    config.write_text("hml3d:\n  dim: 263\n  num_joints: 22\n  fps: 20\n", encoding="utf-8")

    assert load_config(config).generator is not None


@pytest.mark.parametrize("path", EXPERIMENT_CONFIGS, ids=lambda path: path.stem)
def test_every_experiment_config_uses_only_known_root_keys(path):
    load_config(path)
