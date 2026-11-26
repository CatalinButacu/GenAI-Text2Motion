from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from text2motion.app import training
from text2motion.app.bootstrap import ApplicationContext
from text2motion.app.cli_requests import PrepareRequest
from text2motion.motion.contracts import PreparationRequest, PreparationStage


def app(official: Path) -> ApplicationContext:
    request = PreparationRequest(out_dir=official, index_csv=Path("index.csv"))
    return cast(
        ApplicationContext, SimpleNamespace(create_hml3d_preparation_request=lambda: request)
    )


def test_the_preparation_stages_write_into_the_regeneration_folder(tmp_path, monkeypatch):
    written = []
    monkeypatch.setattr(
        training,
        "run_pipeline",
        lambda request, stage: written.append((Path(request.out_dir), stage)),
    )
    official = tmp_path / "official"
    regen = tmp_path / "regen"

    training.run_prepare(
        app(official), PrepareRequest(stage=PreparationStage.JOINTS, regen_dir=regen)
    )

    assert written == [(regen, PreparationStage.JOINTS)]


def test_the_training_dataset_is_refused_as_the_regeneration_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(training, "run_pipeline", lambda *args: None)
    official = tmp_path / "official"

    with pytest.raises(ValueError, match="training dataset"):
        training.run_prepare(
            app(official), PrepareRequest(stage=PreparationStage.JOINTS, regen_dir=official)
        )
