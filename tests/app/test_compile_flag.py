from typing import cast

import torch

from text2motion.app.stages import generator


def test_compile_stays_off_without_cuda():
    module = torch.nn.Linear(2, 2)
    assert generator.compile_generator(cast(generator.MotionTokenGenerator, module), "cpu") is False


def test_compile_stays_off_without_triton(monkeypatch):
    monkeypatch.setattr(generator.importlib.util, "find_spec", lambda name: None)
    module = torch.nn.Linear(2, 2)
    assert (
        generator.compile_generator(cast(generator.MotionTokenGenerator, module), "cuda") is False
    )


def test_compile_keeps_the_checkpoint_keys(monkeypatch):
    monkeypatch.setattr(generator.importlib.util, "find_spec", lambda name: object())
    module = torch.nn.Linear(2, 2)
    keys = list(module.state_dict())
    assert generator.compile_generator(cast(generator.MotionTokenGenerator, module), "cuda") is True
    assert list(module.state_dict()) == keys
