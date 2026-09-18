import pytest
import torch

from vcm.train.architectures import BCResNet, DSCNN

MODEL_CLASSES = [DSCNN, BCResNet]


@pytest.mark.parametrize("model_cls", MODEL_CLASSES)
def test_forward_pass_output_shape_matches_num_classes(model_cls):
    model = model_cls(num_classes=20)
    x = torch.randn(4, 40, 301)  # matches real extract_log_mel output shape
    out = model(x)
    assert out.shape == (4, 20)


@pytest.mark.parametrize("model_cls", MODEL_CLASSES)
def test_forward_pass_single_example(model_cls):
    model = model_cls(num_classes=5)
    x = torch.randn(1, 40, 301)
    out = model(x)
    assert out.shape == (1, 5)


@pytest.mark.parametrize("model_cls", MODEL_CLASSES)
def test_model_is_small(model_cls):
    model = model_cls(num_classes=20)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params < 100_000  # "tiny" per the architecture doc's own framing


@pytest.mark.parametrize("model_cls", MODEL_CLASSES)
def test_handles_smaller_time_dimension(model_cls):
    # sanity check neither model assumes an exact frame count
    model = model_cls(num_classes=20)
    x = torch.randn(2, 40, 151)  # e.g. what the old 1.5s window produced
    out = model(x)
    assert out.shape == (2, 20)


@pytest.mark.parametrize("model_cls", MODEL_CLASSES)
def test_backward_pass_produces_gradients(model_cls):
    model = model_cls(num_classes=20)
    x = torch.randn(3, 40, 301)
    labels = torch.tensor([0, 5, 19])
    loss = torch.nn.functional.cross_entropy(model(x), labels)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
