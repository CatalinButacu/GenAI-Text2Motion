from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from text2motion.app.bootstrap import ApplicationContext
from text2motion.app.commands import data
from text2motion.motion.contracts import PreparationRequest, PreparationStage, PrepareRequest


def __app(official: Path) -> ApplicationContext:
    request = PreparationRequest(out_dir=official, index_csv=Path("index.csv"))
    return cast(
        ApplicationContext, SimpleNamespace(create_hml3d_preparation_request=lambda: request)
    )


def test_the_preparation_stages_write_into_the_regeneration_folder(tmp_path, monkeypatch):
    written = []
    monkeypatch.setattr(
        data,
        "run_hml3d_preparation",
        lambda request, stage, adopt: written.append(Path(request.out_dir)),
    )
    official = tmp_path / "official"
    regen = tmp_path / "regen"

    data.run_prepare(__app(official), PrepareRequest(stage=PreparationStage.INDEX, regen_dir=regen))

    assert written == [regen]


def test_the_training_dataset_is_refused_as_the_regeneration_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "run_hml3d_preparation", lambda *args: None)
    official = tmp_path / "official"

    with pytest.raises(ValueError, match="training dataset"):
        data.run_prepare(
            __app(official), PrepareRequest(stage=PreparationStage.INDEX, regen_dir=official)
        )
