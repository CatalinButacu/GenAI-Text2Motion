from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from text2motion.app.requests.data import PrepareRequest
from text2motion.app.runtime.context import ApplicationContext
from text2motion.app.stages import data as data_stage
from text2motion.motion.contracts import PreparationRequest, PreparationStage


def app(official: Path) -> ApplicationContext:
    request = PreparationRequest(out_dir=official, index_csv=Path("index.csv"))
    return cast(
        ApplicationContext, SimpleNamespace(create_hml3d_preparation_request=lambda: request)
    )


def test_the_preparation_stages_write_into_the_regeneration_folder(tmp_path, monkeypatch):
    written = []
    monkeypatch.setattr(
        data_stage,
        "run_pipeline",
        lambda request, stage: written.append((Path(request.out_dir), stage)),
    )
    official = tmp_path / "official"
    regen = tmp_path / "regen"

    data_stage.run_prepare(
        app(official), PrepareRequest(stage=PreparationStage.JOINTS, regen_dir=regen)
    )

    assert written == [(regen, PreparationStage.JOINTS)]


def test_the_training_dataset_is_refused_as_the_regeneration_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(data_stage, "run_pipeline", lambda *args: None)
    official = tmp_path / "official"

    with pytest.raises(ValueError, match="training dataset"):
        data_stage.run_prepare(
            app(official), PrepareRequest(stage=PreparationStage.JOINTS, regen_dir=official)
        )
