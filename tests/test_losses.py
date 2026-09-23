import torch

from vcm.train.losses import (
    CONFUSABLE_GROUPS,
    ConfusablePairLoss,
    build_confusable_mask,
    effective_number_weights,
)

LABELS = ("VOLUME_UP", "VOLUME_DOWN", "TEMPERATURE", "LIGHT_ON", "LIGHT_OFF", "CALL")


def test_build_confusable_mask_flags_group_members_both_directions():
    mask = build_confusable_mask(LABELS)
    idx = {label: i for i, label in enumerate(LABELS)}
    assert mask[idx["VOLUME_UP"], idx["VOLUME_DOWN"]]
    assert mask[idx["VOLUME_DOWN"], idx["VOLUME_UP"]]
    assert mask[idx["VOLUME_UP"], idx["TEMPERATURE"]]
    assert mask[idx["LIGHT_ON"], idx["LIGHT_OFF"]]
    assert mask[idx["LIGHT_OFF"], idx["LIGHT_ON"]]


def test_build_confusable_mask_no_self_loops():
    mask = build_confusable_mask(LABELS)
    assert not mask.diagonal().any()


def test_build_confusable_mask_excludes_unrelated_labels():
    mask = build_confusable_mask(LABELS)
    idx = {label: i for i, label in enumerate(LABELS)}
    assert not mask[idx["CALL"], idx["VOLUME_UP"]]
    assert not mask[idx["VOLUME_UP"], idx["LIGHT_ON"]]
    assert not mask[idx["CALL"]].any()


def test_build_confusable_mask_ignores_missing_labels():
    small_labels = ("VOLUME_UP", "VOLUME_DOWN", "CALL")
    mask = build_confusable_mask(small_labels, CONFUSABLE_GROUPS)
    idx = {label: i for i, label in enumerate(small_labels)}
    assert mask[idx["VOLUME_UP"], idx["VOLUME_DOWN"]]
    assert mask.sum() == 2


def test_alpha_zero_matches_plain_cross_entropy():
    torch.manual_seed(0)
    weights = torch.ones(len(LABELS))
    mask = build_confusable_mask(LABELS)
    logits = torch.randn(8, len(LABELS))
    targets = torch.randint(0, len(LABELS), (8,))

    plain = torch.nn.functional.cross_entropy(logits, targets, weight=weights)
    confusable = ConfusablePairLoss(weights, mask, alpha=0.0)(logits, targets)
    assert torch.allclose(plain, confusable)


def test_alpha_positive_penalizes_confusable_misprediction_more():
    weights = torch.ones(len(LABELS))
    mask = build_confusable_mask(LABELS)
    idx = {label: i for i, label in enumerate(LABELS)}
    criterion = ConfusablePairLoss(weights, mask, alpha=2.0)

    targets = torch.tensor([idx["VOLUME_UP"]])

    # Same margin over the target class, but one puts leftover mass on the
    # confusable class (VOLUME_DOWN), the other on an unrelated class (CALL).
    logits_confusable = torch.zeros(1, len(LABELS))
    logits_confusable[0, idx["VOLUME_UP"]] = 2.0
    logits_confusable[0, idx["VOLUME_DOWN"]] = 2.0

    logits_unrelated = torch.zeros(1, len(LABELS))
    logits_unrelated[0, idx["VOLUME_UP"]] = 2.0
    logits_unrelated[0, idx["CALL"]] = 2.0

    loss_confusable = criterion(logits_confusable, targets)
    loss_unrelated = criterion(logits_unrelated, targets)
    assert loss_confusable > loss_unrelated


def test_focal_gamma_zero_matches_plain_cross_entropy():
    torch.manual_seed(0)
    weights = torch.ones(len(LABELS))
    mask = build_confusable_mask(LABELS)
    logits = torch.randn(8, len(LABELS))
    targets = torch.randint(0, len(LABELS), (8,))

    plain = torch.nn.functional.cross_entropy(logits, targets, weight=weights)
    focal_off = ConfusablePairLoss(weights, mask, alpha=0.0, gamma=0.0)(logits, targets)
    assert torch.allclose(plain, focal_off)


def test_focal_gamma_downweights_confident_correct_example_more():
    weights = torch.ones(len(LABELS))
    mask = build_confusable_mask(LABELS)
    idx = {label: i for i, label in enumerate(LABELS)}
    criterion_plain = ConfusablePairLoss(weights, mask, alpha=0.0, gamma=0.0)
    criterion_focal = ConfusablePairLoss(weights, mask, alpha=0.0, gamma=2.0)

    targets = torch.tensor([idx["CALL"]])

    # Confident-correct prediction: focal loss should shrink this example's
    # loss much more than an unconfident-correct one, relative to plain CE.
    logits_confident = torch.zeros(1, len(LABELS))
    logits_confident[0, idx["CALL"]] = 10.0

    logits_unconfident = torch.zeros(1, len(LABELS))
    logits_unconfident[0, idx["CALL"]] = 0.5

    plain_confident = criterion_plain(logits_confident, targets)
    focal_confident = criterion_focal(logits_confident, targets)
    plain_unconfident = criterion_plain(logits_unconfident, targets)
    focal_unconfident = criterion_focal(logits_unconfident, targets)

    # Both should shrink relative to plain CE (loss never increases from
    # the (1-p_t)^gamma <= 1 factor), but the confident example shrinks
    # by a much larger relative amount.
    ratio_confident = focal_confident / plain_confident
    ratio_unconfident = focal_unconfident / plain_unconfident
    assert ratio_confident < ratio_unconfident


def test_effective_number_weights_favors_rare_classes():
    counts = torch.tensor([8691.0, 3524.0, 396.0])  # TEMPERATURE, VOLUME_UP, CALL-like ratios
    weights = effective_number_weights(counts, beta=0.999)
    assert weights[2] > weights[1] > weights[0]  # rarest class gets the highest weight


def test_effective_number_weights_mean_is_normalized():
    counts = torch.tensor([8691.0, 3524.0, 396.0, 1000.0])
    weights = effective_number_weights(counts, beta=0.999)
    assert torch.allclose(weights.mean(), torch.tensor(1.0), atol=1e-4)


def test_effective_number_weights_uniform_counts_gives_uniform_weights():
    counts = torch.full((5,), 1000.0)
    weights = effective_number_weights(counts, beta=0.999)
    assert torch.allclose(weights, torch.ones(5), atol=1e-4)


def test_label_smoothing_zero_matches_plain_cross_entropy():
    torch.manual_seed(0)
    weights = torch.ones(len(LABELS))
    mask = build_confusable_mask(LABELS)
    logits = torch.randn(8, len(LABELS))
    targets = torch.randint(0, len(LABELS), (8,))

    plain = torch.nn.functional.cross_entropy(logits, targets, weight=weights)
    smoothed_off = ConfusablePairLoss(weights, mask, alpha=0.0, label_smoothing=0.0)(logits, targets)
    assert torch.allclose(plain, smoothed_off)


def test_label_smoothing_positive_differs_from_plain_cross_entropy():
    torch.manual_seed(0)
    weights = torch.ones(len(LABELS))
    mask = build_confusable_mask(LABELS)
    logits = torch.randn(8, len(LABELS))
    targets = torch.randint(0, len(LABELS), (8,))

    plain = torch.nn.functional.cross_entropy(logits, targets, weight=weights)
    smoothed = ConfusablePairLoss(weights, mask, alpha=0.0, label_smoothing=0.1)(logits, targets)
    assert not torch.allclose(plain, smoothed)


def test_label_smoothing_reduces_loss_for_a_confident_correct_prediction():
    # Label smoothing penalizes over-confidence, so a very confident correct
    # prediction should score a *higher* loss under smoothing than without
    # it (it's being told not to be that sure), not lower.
    weights = torch.ones(len(LABELS))
    mask = build_confusable_mask(LABELS)
    idx = {label: i for i, label in enumerate(LABELS)}
    targets = torch.tensor([idx["CALL"]])
    logits = torch.zeros(1, len(LABELS))
    logits[0, idx["CALL"]] = 10.0

    plain = ConfusablePairLoss(weights, mask, alpha=0.0, label_smoothing=0.0)(logits, targets)
    smoothed = ConfusablePairLoss(weights, mask, alpha=0.0, label_smoothing=0.1)(logits, targets)
    assert smoothed > plain
