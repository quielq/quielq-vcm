#!/usr/bin/env python
"""Voice the targeted phrasings in vcm.dataset.sources.targeted_synth with
Chatterbox TTS, cloning real speakers (EXPERIMENTS.md Experiment 31).

Runs in a separate environment from training (Chatterbox pins its own
torch/transformers versions):

    python3 -m venv ~/tts-venv && ~/tts-venv/bin/pip install chatterbox-tts soundfile

Three stages, each resumable:

    # 1. Reference voices: per real speaker, ~6-10s of their own clips,
    #    trimmed and concatenated. Train-split clips are cloned from
    #    train-split speakers only and test-split from test-split, so the
    #    held-out test clips are unseen voices.
    ~/tts-venv/bin/python scripts/generate_targeted_synthetic.py refs

    # 2. Synthesis, sharded so several processes share one GPU:
    for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=2 nohup ~/tts-venv/bin/python \\
        scripts/generate_targeted_synthetic.py synth --shard $i --num-shards 4 & done

    # 3. QA with faster-whisper, in the *project* venv (see qa below):
    .venv/bin/python scripts/generate_targeted_synthetic.py qa

Only FSC and Timers-and-Such speakers are cloned: their speaker_id is a
real per-person ID. SLURP's is per recording, which can't guarantee
speaker-disjoint splits.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vcm.dataset.sources.targeted_synth import (  # noqa: E402
    BATCHES,
    batch_phrases,
    normalize_for_qa,
    schema_phrases,
    word_error_rate,
)
from vcm.slots import parse_slot  # noqa: E402

# Reference voices are shared by every batch (built once, under round1's folder).
REFS_ROOT = REPO_ROOT / "data/external/targeted_synth/refs"
OUT_ROOT = REPO_ROOT / BATCHES["round1"]["out_dir"]  # reset by main() from --batch
CLIPS_PER_LABEL = BATCHES["round1"]["clips"]
BATCH = "round1"
REF_SOURCES = ("fsc", "timers_and_such")
REF_MIN_S, REF_MAX_S = 6.0, 10.0
TARGET_SR = 16000
QA_MAX_WER = 0.2
# For the verb minimal pairs, a clip only counts if Whisper hears the
# intended verb: a TTS "stop" that sounds like "start" (the Whisper
# confusion found in the STOP audit) would teach exactly the wrong thing.
_MEDIA_VERBS = {"PAUSE": {"pause", "hold"}, "STOP": {"stop", "turn"}, "PLAY_MUSIC": {"play", "start", "put"}}


WAKE_DURATION_S = (0.4, 2.5)
WAKE_MIN_SPEECH_LEVEL = 0.01


def _plausible_wake_clip(path: str) -> bool:
    from vcm.audio.features import speech_level

    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    duration = len(audio) / sr
    return WAKE_DURATION_S[0] <= duration <= WAKE_DURATION_S[1] and speech_level(audio, sr) >= WAKE_MIN_SPEECH_LEVEL


def _first_verb(words: list[str], verbs: set[str]) -> str | None:
    """First verb-set word, skipping polite prefixes like "please" / "can you"."""
    return next((w for w in words if w in verbs), None)


def _trim(audio: np.ndarray, sr: int, top_db: float = 30.0) -> np.ndarray:
    """Energy trim without librosa (not installed in the TTS venv)."""
    frame = int(0.02 * sr)
    if len(audio) < frame:
        return audio
    n = len(audio) // frame
    energy = 10 * np.log10(np.mean(audio[: n * frame].reshape(n, frame) ** 2, axis=1) + 1e-10)
    voiced = np.where(energy > energy.max() - top_db)[0]
    return audio[voiced[0] * frame : (voiced[-1] + 1) * frame] if len(voiced) else audio


def build_refs(manifest: Path, seed: int) -> None:
    by_speaker: dict[tuple[str, str], list[str]] = defaultdict(list)
    with manifest.open(newline="") as f:
        for r in csv.DictReader(f):
            if r["source"] in REF_SOURCES and r["split"] in ("train", "test"):
                by_speaker[(r["split"], r["speaker_id"])].append(r["audio_path"])
    rng = random.Random(seed)
    n_written = 0
    for (split, speaker), paths in sorted(by_speaker.items()):
        out = REFS_ROOT / split / f"{speaker}.wav"
        if out.exists():
            continue
        rng.shuffle(paths)
        pieces, total, sr = [], 0.0, None
        for p in paths:
            audio, clip_sr = sf.read(p, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr is None:
                sr = clip_sr
            if clip_sr != sr:
                continue
            audio = _trim(audio, sr)
            pieces += [audio, np.zeros(int(0.25 * sr), dtype="float32")]
            total += len(audio) / sr
            if total >= REF_MIN_S:
                break
        if total < REF_MIN_S:
            continue
        ref = np.concatenate(pieces)[: int(REF_MAX_S * sr)]
        ref = 0.9 * ref / (np.abs(ref).max() + 1e-9)
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out, ref, sr)
        n_written += 1
    counts = {s: len(list((REFS_ROOT / s).glob("*.wav"))) for s in ("train", "test")}
    print(f"wrote {n_written} new reference voices; total per split: {counts}")


def plan(seed: int) -> list[dict]:
    """Deterministic job list: every phrase of a label gets (nearly) equal
    coverage, each clip a random speaker and random prosody settings. The
    class schema's phrasings stay first in the cycle (only the extras are
    shuffled), so they're always covered even if a label's clip budget is
    smaller than its phrase count."""
    rng = random.Random(seed)
    jobs = []
    all_phrases = batch_phrases(BATCH)
    for split, per_label in CLIPS_PER_LABEL.items():
        speakers = sorted(p.stem for p in (REFS_ROOT / split).glob("*.wav"))
        for label, n in per_label.items():
            n_schema = len(schema_phrases(label)) if BATCH == "round1" else 0
            extras = list(all_phrases[label][n_schema:])
            rng.shuffle(extras)
            phrases = list(all_phrases[label][:n_schema]) + extras
            for i in range(n):
                text, slot_value = phrases[i % len(phrases)]
                jobs.append(
                    {
                        "id": f"{label}_{split}_{i:04d}",
                        "split": split,
                        "label": label,
                        "text": text,
                        "slot_value": slot_value,
                        "speaker_id": rng.choice(speakers),
                        "exaggeration": round(rng.uniform(0.3, 0.7), 2),
                        "cfg_weight": round(rng.uniform(0.3, 0.6), 2),
                    }
                )
    return jobs


def synth(shard: int, num_shards: int, seed: int, limit: int | None = None) -> None:
    import torch
    import torchaudio.functional as AF
    from chatterbox.tts import ChatterboxTTS

    torch.manual_seed(seed + shard)
    model = ChatterboxTTS.from_pretrained(device="cuda")
    t0 = time.time()
    jobs = plan(seed)[shard::num_shards][:limit]
    meta_path = OUT_ROOT / f"synth_shard{shard}.csv"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if meta_path.exists():
        with meta_path.open(newline="") as f:
            done = {r["id"] for r in csv.DictReader(f)}
    fields = ["id", "audio_path", "label", "text", "slot_value", "speaker_id", "split", "exaggeration", "cfg_weight"]
    with meta_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not done:
            writer.writeheader()
        for k, job in enumerate(jobs):
            if job["id"] in done:
                continue
            ref = REFS_ROOT / job["split"] / f"{job['speaker_id']}.wav"
            sentence = job["text"][0].upper() + job["text"][1:] + "."
            wav = model.generate(
                sentence,
                audio_prompt_path=str(ref),
                exaggeration=job["exaggeration"],
                cfg_weight=job["cfg_weight"],
            )
            wav = AF.resample(wav.cpu(), model.sr, TARGET_SR).squeeze(0).numpy()
            out = OUT_ROOT / "audio" / job["label"] / f"{job['id']}.wav"
            out.parent.mkdir(parents=True, exist_ok=True)
            sf.write(out, wav, TARGET_SR, subtype="PCM_16")
            writer.writerow({**{k2: job[k2] for k2 in fields if k2 in job}, "audio_path": str(out)})
            f.flush()
            if k % 50 == 0:
                print(f"shard {shard}: {k}/{len(jobs)}", flush=True)
    print(f"shard {shard}: done ({len(jobs)} jobs, {time.time() - t0:.0f}s)", flush=True)


def qa() -> None:
    """Transcribe every clip with faster-whisper base (English) and keep it
    only if the transcript matches the intended text (WER <= QA_MAX_WER
    after number normalization) and, for PAUSE/STOP/PLAY_MUSIC, starts with
    the intended verb."""
    from faster_whisper import WhisperModel

    rows = []
    for shard_csv in sorted(OUT_ROOT.glob("synth_shard*.csv")):
        with shard_csv.open(newline="") as f:
            rows += list(csv.DictReader(f))
    whisper = WhisperModel("base", device="cuda", compute_type="float16")
    for i, r in enumerate(rows):
        segments, _ = whisper.transcribe(r["audio_path"], language="en", beam_size=5)
        heard = " ".join(s.text for s in segments).strip()
        ref, hyp = normalize_for_qa(r["text"]), normalize_for_qa(heard)
        wer = word_error_rate(ref, hyp)
        verbs = _MEDIA_VERBS.get(r["label"])
        verb_ok = verbs is None or (_first_verb(hyp, verbs) is not None and _first_verb(hyp, verbs) == _first_verb(ref, verbs))
        passed = wer <= QA_MAX_WER and verb_ok
        if BATCH == "slots2":
            # The slot value is the point of this batch: Whisper must hear the
            # intended value. WER is looser because "A.M."/"AM"/"in the
            # morning" transcribe inconsistently.
            passed = parse_slot(r["label"], heard) == r["slot_value"] and wer <= 0.34
        elif BATCH == "wakeword":
            # A negative only has to not sound like the wake word; a WER check
            # rejected most two-word negatives over a single misheard name.
            heard_wake = "hey kiwi" in " ".join(hyp)
            if r["label"] == "WAKE":
                # Listening confirmed the clips say "hey kiwi" even when
                # Whisper-base hears "Thank you" (EXPERIMENTS.md Experiment 33),
                # so positives are checked on the audio instead: a plausible
                # length and actual speech, not silence or a runaway generation.
                passed = _plausible_wake_clip(r["audio_path"])
            else:
                passed = not heard_wake
        r.update(whisper_text=heard, wer=f"{wer:.3f}", qa_pass=str(passed))
        if i % 500 == 0:
            print(f"qa {i}/{len(rows)}", flush=True)
    out = OUT_ROOT / "manifest.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    by = defaultdict(lambda: [0, 0])
    for r in rows:
        by[(r["split"], r["label"])][0] += r["qa_pass"] == "True"
        by[(r["split"], r["label"])][1] += 1
    for (split, label), (ok, n) in sorted(by.items()):
        print(f"{split:<6}{label:<12}{ok:>5}/{n:<5} passed ({ok / n:.1%})")
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["refs", "synth", "qa", "plan"])
    parser.add_argument(
        "--batch",
        choices=sorted(BATCHES),
        default="round1",
        help="Which phrase set to generate (vcm.dataset.sources.targeted_synth.BATCHES): round1 = "
        "Experiment 31's clips, slots2 = slot-value coverage, wakeword = Hey Kiwi. Each writes to "
        "its own folder; reference voices are shared.",
    )
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "data/dataset_manifest.csv")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="synth: only the first N jobs of this shard (smoke test)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    global OUT_ROOT, CLIPS_PER_LABEL, BATCH
    BATCH = args.batch
    OUT_ROOT = REPO_ROOT / BATCHES[BATCH]["out_dir"]
    CLIPS_PER_LABEL = BATCHES[BATCH]["clips"]
    if args.stage == "refs":
        build_refs(args.manifest, args.seed)
    elif args.stage == "plan":
        jobs = plan(args.seed)
        print(f"{len(jobs)} jobs; first: {jobs[0]}")
    elif args.stage == "synth":
        synth(args.shard, args.num_shards, args.seed, args.limit)
    else:
        qa()


if __name__ == "__main__":
    main()
