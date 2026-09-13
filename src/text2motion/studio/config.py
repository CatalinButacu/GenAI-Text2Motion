from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from text2motion.motion.model import Gender

_Color = tuple[float, float, float, float]
_Pair = tuple[float, float]


@dataclass(frozen=True)
class SmplxFitConfig:
    model_dir: str
    device: str = "cpu"
    threads: int = 4
    gender: Gender = Gender.NEUTRAL
    num_betas: int = 10
    stage1_iters: int = 150
    stage2_iters: int = 300
    lr: float = 0.05
    w_smooth: float = 0.05
    w_reg: float = 5e-4
    progress_every: int = 10
    verbose_every: int = 100
    yaw_candidates_deg: tuple[float, ...] = (0.0, 90.0, 180.0, 270.0)


@dataclass(frozen=True)
class SmplxFitSchedule:
    first_stage1_iters: int = 60
    first_stage2_iters: int = 150
    continuation_stage1_iters: int = 0
    continuation_stage2_iters: int = 25
    retrofit_window: int = 16


@dataclass(frozen=True)
class StreamDisplayConfig:
    window_frames: int = 600
    fit_lookahead_chunks: int = 8


@dataclass(frozen=True)
class LiveGenerationConfig:
    steps: int = 20
    debounce_seconds: float = 0.35


@dataclass(frozen=True)
class SceneAppearanceConfig:
    skin_color: _Color = (0.86, 0.72, 0.61, 1.0)
    sky_color: _Color = (240 / 255, 182 / 255, 182 / 255, 1.0)


@dataclass(frozen=True)
class PanelLayoutConfig:
    left_width: float = 0.20
    right_width: float = 0.24
    prompt_height: float = 0.11
    editor_height: float = 0.52
    playback_height: float = 0.30


@dataclass(frozen=True)
class StudioThemeConfig:
    window_rounding: float = 8.0
    child_rounding: float = 6.0
    frame_rounding: float = 5.0
    grab_rounding: float = 5.0
    popup_rounding: float = 6.0
    scrollbar_rounding: float = 6.0
    window_border_size: float = 0.0
    window_padding: _Pair = (12.0, 10.0)
    frame_padding: _Pair = (8.0, 4.0)
    item_spacing: _Pair = (8.0, 6.0)
    section_label_color: _Color = (0.55, 0.75, 1.0, 1.0)
    colors: dict[str, _Color] = field(
        default_factory=lambda: {
            "window_background": (0.09, 0.10, 0.13, 0.94),
            "title_background": (0.07, 0.08, 0.11, 1.0),
            "title_background_active": (0.12, 0.19, 0.32, 1.0),
            "frame_background": (0.16, 0.18, 0.23, 1.0),
            "frame_background_hovered": (0.22, 0.25, 0.32, 1.0),
            "frame_background_active": (0.26, 0.30, 0.38, 1.0),
            "button": (0.20, 0.34, 0.60, 1.0),
            "button_hovered": (0.26, 0.44, 0.78, 1.0),
            "button_active": (0.30, 0.52, 0.92, 1.0),
            "header": (0.18, 0.26, 0.42, 1.0),
            "header_hovered": (0.24, 0.34, 0.54, 1.0),
            "header_active": (0.28, 0.40, 0.62, 1.0),
            "slider_grab": (0.42, 0.62, 0.98, 1.0),
            "slider_grab_active": (0.55, 0.72, 1.0, 1.0),
            "check_mark": (0.42, 0.66, 1.0, 1.0),
            "separator": (0.25, 0.28, 0.36, 1.0),
            "plot_lines": (0.42, 0.66, 1.0, 1.0),
            "plot_histogram": (0.42, 0.66, 1.0, 1.0),
        }
    )


@dataclass(frozen=True)
class StudioConfig:
    fit: SmplxFitConfig = field(default_factory=lambda: SmplxFitConfig(model_dir=""))
    schedule: SmplxFitSchedule = field(default_factory=SmplxFitSchedule)
    live: LiveGenerationConfig = field(default_factory=LiveGenerationConfig)
    display: StreamDisplayConfig = field(default_factory=StreamDisplayConfig)
    appearance: SceneAppearanceConfig = field(default_factory=SceneAppearanceConfig)
    layout: PanelLayoutConfig = field(default_factory=PanelLayoutConfig)
    theme: StudioThemeConfig = field(default_factory=StudioThemeConfig)
    model_registry: Path = Path("configs/demo_models.yaml")

    @property
    def genders(self) -> list[str]:
        return [gender.value for gender in Gender]
