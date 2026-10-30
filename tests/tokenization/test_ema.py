import pytest
import torch
from torch import nn

from text2motion.tokenization.ema import ExponentialMovingAverage


def __small_model() -> tuple[nn.Module, nn.Linear]:
    torch.manual_seed(0)
    first = nn.Linear(4, 3)
    second = nn.Linear(3, 2)
    second.bias.requires_grad_(False)
    return nn.Sequential(first, second), first


def test_update_follows_the_ema_formula_for_trained_parameters_only():
    model, _ = __small_model()
    ema = ExponentialMovingAverage(model, decay=0.9)
    expected = {name: value.clone() for name, value in ema.shadow.items()}
    for step in range(3):
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(step + 1.0)
        ema.update(model)
        parameters = dict(model.named_parameters())
        for name in expected:
            expected[name] = 0.9 * expected[name] + 0.1 * parameters[name].detach()
    assert "1.bias" not in ema.shadow
    for name, value in expected.items():
        torch.testing.assert_close(ema.shadow[name], value)


def test_copy_to_then_restore_gives_back_the_trained_weights():
    model, first = __small_model()
    ema = ExponentialMovingAverage(model, decay=0.5)
    with torch.no_grad():
        first.weight.add_(1.0)
    trained = first.weight.detach().clone()
    ema.update(model)
    ema.copy_to(model)
    torch.testing.assert_close(first.weight, ema.shadow["0.weight"])
    ema.restore(model)
    torch.testing.assert_close(first.weight, trained)


def test_update_with_a_model_that_lost_a_parameter_raises():
    model, _ = __small_model()
    ema = ExponentialMovingAverage(model, decay=0.9)
    with pytest.raises(KeyError):
        ema.update(nn.Sequential(nn.Linear(4, 3)))


def test_a_decay_outside_zero_to_one_raises():
    model, _ = __small_model()
    with pytest.raises(ValueError, match="decay"):
        ExponentialMovingAverage(model, decay=1.5)


def test_a_second_copy_to_before_restore_raises():
    model, _ = __small_model()
    ema = ExponentialMovingAverage(model, decay=0.9)
    ema.copy_to(model)
    with pytest.raises(RuntimeError, match="before restore"):
        ema.copy_to(model)


def test_restore_without_copy_to_raises():
    model, _ = __small_model()
    with pytest.raises(RuntimeError, match="without copy_to"):
        ExponentialMovingAverage(model, decay=0.9).restore(model)


@pytest.mark.parametrize("problem", ["missing", "unexpected", "shape", "not_finite"])
def test_loading_a_bad_ema_state_raises(problem):
    model, _ = __small_model()
    ema = ExponentialMovingAverage(model, decay=0.9)
    state = {name: value.clone() for name, value in ema.shadow.items()}
    if problem == "missing":
        state.pop("0.weight")
    if problem == "unexpected":
        state["extra"] = torch.zeros(1)
    if problem == "shape":
        state["0.weight"] = torch.zeros(1)
    if problem == "not_finite":
        state["0.weight"][0, 0] = float("nan")
    with pytest.raises((KeyError, ValueError)):
        ema.load_state_dict(state)


@pytest.mark.parametrize("average_buffers", [False, True])
def test_floating_buffers_are_averaged_only_on_request(average_buffers):
    model, _ = __small_model()
    codebook = torch.zeros(2)
    model.register_buffer("codebook", codebook)
    model.register_buffer("counter", torch.zeros(1, dtype=torch.long))
    ema = ExponentialMovingAverage(model, decay=0.5, average_buffers=average_buffers)
    codebook.fill_(4.0)
    ema.update(model)
    assert "counter" not in ema.shadow
    assert ("codebook" in ema.shadow) == average_buffers
    if not average_buffers:
        return
    torch.testing.assert_close(ema.shadow["codebook"], torch.full((2,), 2.0))
    ema.copy_to(model)
    torch.testing.assert_close(codebook, torch.full((2,), 2.0))
    ema.restore(model)
    torch.testing.assert_close(codebook, torch.full((2,), 4.0))
