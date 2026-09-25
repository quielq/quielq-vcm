#!/usr/bin/env python
"""Rank wake-word candidates by how often similar-sounding phoneme sequences
occur in (a) this project's 62K Whisper transcripts of real command speech
and (b) common English words. A proxy for false-trigger risk, not a
detector evaluation.

Method: every transcript is turned into one continuous phoneme stream
(CMUdict, stress removed, across word boundaries, so "car in a" can match
"carina"). A candidate "hits" an utterance if some window of the stream
(candidate length -1..+1 phonemes) is within 1 phoneme edit of it.
Lower hit rates = fewer everyday sound-alikes.

How "Hey Kiwi" was chosen; results are in EXPERIMENTS.md ("Wake-word
selection"). Needs its own small environment (not a project dependency):
    python3 -m venv ~/ww-venv && ~/ww-venv/bin/pip install cmudict wordfreq rapidfuzz
    ~/ww-venv/bin/python scripts/wakeword_confusability.py data/cascade_transcripts.csv
"""

import csv
import re
import sys
from collections import Counter

import cmudict
from rapidfuzz.distance import Levenshtein
from wordfreq import zipf_frequency

TRANSCRIPTS = sys.argv[1] if len(sys.argv) > 1 else "data/cascade_transcripts.csv"
MAX_EDITS = 1

# ARPAbet, stress stripped. Carina isn't in CMUdict; Corinne gets both
# common pronunciations (kuh-REEN, kuh-RIN).
CANDIDATES = {
    "Kernel": [["K", "ER", "N", "AH", "L"]],
    "Carina": [["K", "AH", "R", "IY", "N", "AH"]],
    "Cronin": [["K", "R", "OW", "N", "IH", "N"]],
    "Conan": [["K", "OW", "N", "AH", "N"]],
    "Crayon": [["K", "R", "EY", "AA", "N"], ["K", "R", "EY", "AH", "N"]],
    "Cory": [["K", "AO", "R", "IY"]],
    "Corinne": [["K", "ER", "IY", "N"], ["K", "ER", "IH", "N"]],
    "Kiwi": [["K", "IY", "W", "IY"]],
    "Orly": [["AO", "R", "L", "IY"]],
    "Nini": [["N", "IY", "N", "IY"]],
    "Ellie": [["EH", "L", "IY"]],
}
HEY = ["HH", "EY"]

cmu = cmudict.dict()
PHONES = sorted({re.sub(r"\d", "", p) for prons in cmu.values() for pron in prons for p in pron})
CODE = {p: chr(0x100 + i) for i, p in enumerate(PHONES)}  # one char per phoneme -> fast C Levenshtein


def enc(phones):
    return "".join(CODE[p] for p in phones)


def word_phones(word):
    prons = cmu.get(word)
    return [re.sub(r"\d", "", p) for p in prons[0]] if prons else None


def utterance_stream(text):
    """Phoneme string plus, per phoneme, the index of the word it came from.
    Out-of-vocabulary words break the stream (no match spans them)."""
    segments, cur, cur_words = [], [], []
    words = re.findall(r"[a-z']+", text.lower())
    for wi, w in enumerate(words):
        ph = word_phones(w)
        if ph is None:
            if cur:
                segments.append((enc(cur), cur_words))
            cur, cur_words = [], []
            continue
        cur += ph
        cur_words += [wi] * len(ph)
    if cur:
        segments.append((enc(cur), cur_words))
    return segments, words


def hits_in(segments, words, target):
    L = len(target)
    for stream, widx in segments:
        for n in (L - 1, L, L + 1):
            for i in range(0, len(stream) - n + 1):
                if Levenshtein.distance(stream[i : i + n], target, score_cutoff=MAX_EDITS) <= MAX_EDITS:
                    return " ".join(words[widx[i] : widx[i + n - 1] + 1])
    return None


def main():
    with open(TRANSCRIPTS, newline="") as f:
        texts = [r["transcript"] for r in csv.DictReader(f) if r["transcript"].strip()]
    streams = [utterance_stream(t) for t in texts]
    print(f"{len(texts)} transcripts; match = within {MAX_EDITS} phoneme edit, window length L-1..L+1\n")

    common = [(w, zipf_frequency(w, "en")) for w in cmu if zipf_frequency(w, "en") >= 3.0]
    rows = []
    for name, prons in CANDIDATES.items():
        targets = [enc(p) for p in prons]
        hey_targets = [enc(HEY + p) for p in prons]
        bare, hey, examples = 0, 0, Counter()
        for segs, words in streams:
            span = next((s for t in targets if (s := hits_in(segs, words, t))), None)
            if span:
                bare += 1
                examples[span] += 1
            if any(hits_in(segs, words, t) for t in hey_targets):
                hey += 1
        alikes = sorted(
            (
                (w, z)
                for w, z in common
                if w.rstrip("'s") != name.lower()
                and any(
                    Levenshtein.distance(enc(word_phones(w)), t) <= MAX_EDITS for t in targets
                )
            ),
            key=lambda x: -x[1],
        )
        per_million = sum(10 ** (z - 3) for _, z in alikes)  # zipf = log10(frequency per billion words)
        rows.append((name, prons, bare, hey, examples, alikes, per_million))

    n = len(texts)
    print(f"{'wake word':<10}{'phonemes':>9}{'hits/1k utts':>14}{'with Hey /1k':>14}{'common sound-alikes':>21}{'their freq/million words':>26}")
    for name, prons, bare, hey, _, alikes, pm in sorted(rows, key=lambda r: (r[2], r[6])):
        print(f"{name:<10}{len(prons[0]):>9}{1000 * bare / n:>14.2f}{1000 * hey / n:>14.2f}{len(alikes):>21}{pm:>26.1f}")
    print()
    for name, prons, bare, hey, examples, alikes, pm in sorted(rows, key=lambda r: (r[2], r[6])):
        print(f"== {name}  {' / '.join(' '.join(p) for p in prons)}")
        print(f"   top matching spans in transcripts: {examples.most_common(8)}")
        print(f"   most frequent common-English sound-alikes: {[w for w, _ in alikes[:12]]}")


if __name__ == "__main__":
    main()
