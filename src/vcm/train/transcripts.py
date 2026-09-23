"""Word-level CTC targets from the ASR-cascade's Whisper transcripts
(data/cascade_transcripts.csv, Experiment 26) — for an auxiliary,
training-only CTC head (EXPERIMENTS.md Experiment 30).

The idea (Lugosch et al., "Speech Model Pre-training for End-to-End
Spoken Language Understanding", Interspeech 2019): supervising a small
audio model's per-frame features with *which words were said, in what
order* forces it to encode word content rather than coarse acoustic
texture — a much richer signal than Experiment 27's 19-class soft
labels. The CTC head is discarded after training; the deployed model
is still audio -> intent, never produces text, and never runs ASR.

Word-level, not character-level: CRNN's per-frame features are ~80ms
apart (stride 8 at 10ms hops), too coarse for ~12-15 chars/sec speech
but ample for ~3 words/sec. This project's command vocabulary is
small — words seen >= 3 times in train transcripts give 2,170 types
covering 97.6% of tokens; the rest map to <unk>.
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

import torch

BLANK = 0
UNK = 1
MAX_TARGET_WORDS = 48  # longest train transcript, in words
_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def load_transcripts(csv_path: Path) -> dict[str, tuple[str, str]]:
    """audio_path -> (split, transcript)."""
    with Path(csv_path).open(newline="") as f:
        return {r["audio_path"]: (r["split"], r["transcript"]) for r in csv.DictReader(f)}


def build_vocab(transcripts: dict[str, tuple[str, str]], min_count: int = 3) -> list[str]:
    """Index 0 is the CTC blank, 1 is <unk>. Built from the train split only."""
    counts = Counter(w for split, text in transcripts.values() if split == "train" for w in tokenize(text))
    words = sorted(w for w, n in counts.items() if n >= min_count)
    return ["<blank>", "<unk>", *words]


def encode_targets(transcripts: dict[str, tuple[str, str]], vocab: list[str]) -> dict[str, torch.Tensor]:
    """audio_path -> word-index tensor (no blanks), truncated to MAX_TARGET_WORDS.
    Paths absent here (e.g. unknown_background, which was never transcribed)
    should get an empty target: "no words" is the right CTC target for noise."""
    index = {w: i for i, w in enumerate(vocab)}
    return {
        path: torch.tensor([index.get(w, UNK) for w in tokenize(text)][:MAX_TARGET_WORDS], dtype=torch.long)
        for path, (_, text) in transcripts.items()
    }


def pad_target(target: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Fixed-size (MAX_TARGET_WORDS,) tensor + true length, so the default
    DataLoader collate works; CTCLoss ignores everything past the length."""
    padded = torch.zeros(MAX_TARGET_WORDS, dtype=torch.long)
    padded[: len(target)] = target
    return padded, len(target)
