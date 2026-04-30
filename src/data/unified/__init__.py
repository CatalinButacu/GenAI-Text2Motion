from .unified_enrich import enrichUnified  # noqa: F401
from .unified_load import loadAmass, loadArctic, loadHumanml3d, preprocessMotion  # noqa: F401
from .unified_split import splitSamples  # noqa: F401


def buildSourcesBuffer(cfg, minFrames: int, resampleFn, qualityFn, tposeFn,
                       maxLength: int | None = 200) -> list[dict]:
    """Load and enrich all enabled data sources into a flat buffer. Shared by dataset and trainer.

    maxLength: if set, every motion is cropped to this many frames at ingest time.
    Default 200 caps memory; pass None to keep full clips (only safe for small subsets).
    """
    args = (minFrames, resampleFn, qualityFn, tposeFn, maxLength)
    buf: list[dict] = []

    if cfg.amass.enabled:
        loadAmass(buf, cfg.amass, *args)

    if cfg.arctic.enabled:
        loadArctic(buf, cfg.arctic, *args)

    humanml3dCfg = getattr(cfg, "humanml3d", None)
    if humanml3dCfg is not None and humanml3dCfg.enabled:
        loadHumanml3d(buf, humanml3dCfg, *args)

    enrichUnified(buf)

    return buf
