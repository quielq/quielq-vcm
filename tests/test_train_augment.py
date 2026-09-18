import torch

from vcm.train.augment import spec_augment


def test_spec_augment_preserves_shape():
    features = torch.randn(40, 301)
    augmented = spec_augment(features)
    assert augmented.shape == features.shape


def test_spec_augment_does_not_mutate_input():
    features = torch.randn(40, 301)
    original = features.clone()
    spec_augment(features)
    assert torch.equal(features, original)


def test_spec_augment_actually_changes_something():
    torch.manual_seed(0)
    features = torch.randn(40, 301)
    augmented = spec_augment(features, freq_mask_param=8, time_mask_param=40, num_freq_masks=2, num_time_masks=2)
    assert not torch.equal(features, augmented)


def test_spec_augment_handles_tiny_input_without_crashing():
    features = torch.randn(2, 5)
    augmented = spec_augment(features, freq_mask_param=8, time_mask_param=40)
    assert augmented.shape == features.shape
