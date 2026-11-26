import hashlib

from text2motion.app.runtime.context import ApplicationBootstrap
from text2motion.generation.model.text_encoder import FrozenTextStates


def test_attach_text_cache_builds_once_then_loads_by_caption_set(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    config = tmp_path / "config.yaml"
    config.write_text(
        f"device: cpu\npaths:\n  cache_dir: {cache_dir.as_posix()}\n", encoding="utf-8"
    )
    app = ApplicationBootstrap.from_config(config)
    encoder = app.models.load_text_encoder()
    built, loaded = [], []

    def fake_build(enc, captions, batch_size, path):
        built.append((list(captions), batch_size, path))
        path.write_bytes(b"cache")
        return object()

    def fake_load(enc, path):
        loaded.append(path)
        return object()

    monkeypatch.setattr(FrozenTextStates, "build", fake_build)
    monkeypatch.setattr(FrozenTextStates, "load", fake_load)

    captions = ["a person walks", "a man jumps"]
    digest = hashlib.sha256("\n".join(sorted(captions)).encode("utf-8")).hexdigest()[:12]
    expected = cache_dir / (
        f"clip_frozen_{encoder.cfg.model_id.replace('/', '_')}"
        f"_keep{encoder.frozen_layer_count}_b8_{digest}.npy"
    )

    first = app.models.attach_text_cache(encoder, captions, 8)
    assert built == [(captions, 8, expected)]
    assert loaded == []
    assert encoder.frozen_cache is first

    second = app.models.attach_text_cache(encoder, list(reversed(captions)), 8)
    assert built == [(captions, 8, expected)]
    assert loaded == [expected]
    assert encoder.frozen_cache is second

    app.models.attach_text_cache(encoder, [*captions, "a child hops"], 8)
    assert len(built) == 2
    assert built[1][2] != expected
    assert built[1][2].parent == cache_dir
