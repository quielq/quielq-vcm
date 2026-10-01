import csv
import re
from pathlib import Path

from vcm.dataset.sources.dataset_schema import (
    FIXED_INTENTS,
    INTENT_LABELS,
    SLOTTED_INTENTS,
    export_csv,
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
    # Every value in this module is a Python string literal by
    # construction; this just guards that ALARM specifically stays that
    # way, since spreadsheet tools are prone to auto-typing time-like text.
    for value in SLOTTED_INTENTS["ALARM"]["values"]:
        assert isinstance(value, str)


def test_export_csv_roundtrips_to_the_same_phrases(tmp_path):
    csv_path = tmp_path / "dataset_schema.csv"
    export_csv(csv_path)

    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    phrases = generate_phrases()
    assert len(rows) == len(phrases)
    for row, phrase in zip(rows, phrases):
        assert row["label"] == phrase.label
        assert row["phrase"] == phrase.text
        assert row["type"] == ("slotted" if phrase.is_slotted else "fixed")
        assert row["slot_value"] == (phrase.slot_value or "")
        # every cell reads back as a plain str via csv, never a time/date type
        assert all(isinstance(v, str) for v in row.values())


def test_matches_the_final_class_schema():
    """The class's final Dataset Schema (Option B, agreed 2026-10-01), as
    exported from the sheet: one row per intent, 3 variations and, for
    slotted intents, 3 values."""
    path = Path(__file__).resolve().parents[1] / "data/dataset_schema/final_dataset_schema.csv"
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["Label"] for r in rows] == list(INTENT_LABELS)
    for r in rows:
        variations = tuple(r[f"Variation {i}"] for i in (1, 2, 3))
        if r["Type"] == "Fixed":
            assert FIXED_INTENTS[r["Label"]] == variations
        else:
            spec = SLOTTED_INTENTS[r["Label"]]
            assert tuple(re.sub(r"\{[^}]+\}", "{}", t) for t in spec["templates"]) == tuple(
                re.sub(r"\{[^}]+\}", "{}", v) for v in variations
            )
            assert spec["values"] == tuple(r[f"Value {i}"] for i in (1, 2, 3))
