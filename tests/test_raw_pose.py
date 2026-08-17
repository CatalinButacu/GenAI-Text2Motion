import os
from pathlib import Path

import numpy as np
import pytest

from text2motion.app.config import load_config


@pytest.mark.slow
def test_smplx_amass_extractor_real_file():
    cfg = load_config("configs/default.yaml")
    smplx_dir = cfg.paths.smplx_models
    amass_dir = cfg.paths.amass_dir
    if not smplx_dir or not Path(smplx_dir).exists():
        pytest.skip("SMPL-X models unavailable")
    if not amass_dir or not Path(amass_dir).exists():
        pytest.skip("AMASS unavailable")

    from text2motion.motion.preparation import AmassPoseExtractor

    ext = AmassPoseExtractor(Path(smplx_dir))

    src = None
    for root, _, files in os.walk(amass_dir):
        npz = [f for f in files if f.endswith(".npz")]
        if npz:
            src = Path(root) / npz[0]
            break
    if src is None:
        pytest.skip("no AMASS npz found")

    joints = ext.amass_to_pose(src)
    assert joints is not None
    assert joints.ndim == 3 and joints.shape[1:] == (22, 3)
    assert np.isfinite(joints).all()
