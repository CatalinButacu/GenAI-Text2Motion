from __future__ import annotations

from pathlib import Path

from text2motion.common.runner import Step, run_steps
from text2motion.dataset.contracts import DatasetPaths
from text2motion.dataset.core.operations import sha256_file
from text2motion.dataset.sources.humanml3d.build import (
    build_joint_clips,
    encode_hml3d_features,
    fit_hml3d_statistics,
    write_clock_table,
)
from text2motion.motion.contracts import PreparationRequest, PreparationStage


def request_from_paths(paths: DatasetPaths, fps: int, fill: bool = True) -> PreparationRequest:
    return PreparationRequest(
        out_dir=paths.hml3d_at(fps),
        amass_dir=paths.amass_raw,
        smplx_models=paths.smplx_models,
        index_csv=paths.hml3d_official / "index.csv",
        clock_csv=paths.clock_csv,
        fps=fps,
        fill=fill,
    )


def recipe_for(request: PreparationRequest) -> dict[str, object]:
    if request.clock_csv is None or request.index_csv is None:
        raise ValueError("clock_csv and index_csv must be set to name the recipe")
    return {
        "fps": request.fps,
        "fill": request.fill,
        "clock_sha256": sha256_file(request.clock_csv),
        "index_sha256": sha256_file(request.index_csv),
    }


def all_steps(request: PreparationRequest) -> list[Step]:
    steps = [
        Step("clock", (), "clock", lambda inputs, out: write_clock_table(request)),
        Step("joints", ("clock",), "joints", lambda inputs, out: build_joint_clips(request)),
        Step(
            "features",
            ("joints",),
            "features",
            lambda inputs, out: encode_hml3d_features(request),
        ),
    ]
    if request.fill:
        steps.append(
            Step("stats", ("features",), "stats", lambda inputs, out: fit_hml3d_statistics(request))
        )
    return steps


def stage_step(request: PreparationRequest, stage: PreparationStage) -> Step:
    for step in all_steps(request):
        if step.name == stage.value:
            return Step(step.name, (), step.output, step.run)
    raise ValueError(f"stage {stage.value} is not run when fill is off")


def run_pipeline(
    request: PreparationRequest, stage: PreparationStage = PreparationStage.ALL
) -> dict[str, Path]:
    if stage is PreparationStage.ALL:
        steps = all_steps(request)
    else:
        steps = [stage_step(request, stage)]
    return run_steps(steps, Path(request.out_dir), {"recipe": recipe_for(request)})
