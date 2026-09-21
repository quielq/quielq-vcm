import torch

from vcm.train.losses import CONFUSABLE_GROUPS, ConfusablePairLoss, build_confusable_mask

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
