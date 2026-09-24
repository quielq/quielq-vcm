from vcm.train.transcripts import BLANK, MAX_TARGET_WORDS, UNK, build_vocab, encode_targets, pad_target, tokenize


def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("Turn the volume up. What's 5 o'clock?") == ["turn", "the", "volume", "up", "what's", "5", "o'clock"]


def test_vocab_uses_train_split_only_and_reserves_blank_unk():
    transcripts = {
        "a": ("train", "lights on"),
        "b": ("train", "lights off"),
        "c": ("test", "secret secret secret"),
    }
    vocab = build_vocab(transcripts, min_count=1)
    assert vocab[BLANK] == "<blank>" and vocab[UNK] == "<unk>"
    assert "lights" in vocab and "secret" not in vocab


def test_encode_maps_rare_words_to_unk():
    transcripts = {"a": ("train", "lights on"), "b": ("train", "lights off")}
    vocab = build_vocab(transcripts, min_count=2)  # only "lights" survives
    targets = encode_targets(transcripts, vocab)
    assert targets["a"].tolist() == [vocab.index("lights"), UNK]


def test_pad_target_fixed_size():
    import torch

    padded, length = pad_target(torch.tensor([3, 4]))
    assert padded.shape == (MAX_TARGET_WORDS,) and length == 2
