from vcm.dataset.sources.dataset_schema import (
    FIXED_INTENTS,
    INTENT_LABELS,
    SLOTTED_INTENTS,
    generate_phrases,
)


def test_intent_labels_cover_all_19():
    assert len(INTENT_LABELS) == 19
    assert len(FIXED_INTENTS) == 13
    assert len(SLOTTED_INTENTS) == 6


def test_generate_phrases_count_matches_option_b_richness():
    phrases = generate_phrases()
    # 13 fixed labels x 3 variations, 6 slotted labels x 3 templates x 3 values
    assert len(phrases) == 13 * 3 + 6 * 3 * 3


def test_every_label_appears_in_generated_phrases():
    labels_seen = {p.label for p in generate_phrases()}
    assert labels_seen == set(INTENT_LABELS)


def test_slotted_phrases_have_no_leftover_placeholder():
    for phrase in generate_phrases():
        if phrase.is_slotted:
            assert "{" not in phrase.text and "}" not in phrase.text
            assert phrase.slot_value is not None
            assert phrase.slot_value in phrase.text


def test_fixed_phrases_are_not_marked_slotted():
    for phrase in generate_phrases():
        if not phrase.is_slotted:
            assert phrase.slot_value is None


def test_alarm_values_are_plain_strings_not_time_objects():
    # Regression guard for the cell-type bug found in the live sheet
    # (Excel auto-converted ALARM's example times to datetime.time
    # objects) — this hand-captured copy must not reintroduce it.
    for value in SLOTTED_INTENTS["ALARM"]["values"]:
        assert isinstance(value, str)
