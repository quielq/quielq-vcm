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
    CLIPS_PER_LABEL,
    PHRASES,
    normalize_for_qa,
    schema_phrases,
    word_error_rate,
)

OUT_ROOT = REPO_ROOT / "data/external/targeted_synth"
REF_SOURCES = ("fsc", "timers_and_such")
REF_MIN_S, REF_MAX_S = 6.0, 10.0
TARGET_SR = 16000
QA_MAX_WER = 0.2
# For the verb minimal pairs, a clip only counts if Whisper hears the
# intended verb: a TTS "stop" that sounds like "start" (the Whisper
# confusion found in the STOP audit) would teach exactly the wrong thing.
_MEDIA_VERBS = {"PAUSE": {"pause", "hold"}, "STOP": {"stop", "turn"}, "PLAY_MUSIC": {"play", "start", "put"}}


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
            if r["source"] in REF_SOURCES and r["split"] in CLIPS_PER_LABEL:
                by_speaker[(r["split"], r["speaker_id"])].append(r["audio_path"])
    rng = random.Random(seed)
    n_written = 0
    for (split, speaker), paths in sorted(by_speaker.items()):
        out = OUT_ROOT / "refs" / split / f"{speaker}.wav"
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
    counts = {s: len(list((OUT_ROOT / "refs" / s).glob("*.wav"))) for s in CLIPS_PER_LABEL}
    print(f"wrote {n_written} new reference voices; total per split: {counts}")


def plan(seed: int) -> list[dict]:
    """Deterministic job list: every phrase of a label gets (nearly) equal
    coverage, each clip a random speaker and random prosody settings. The
    class schema's phrasings stay first in the cycle (only the extras are
    shuffled), so they're always covered even if a label's clip budget is
    smaller than its phrase count."""
    rng = random.Random(seed)
    jobs = []
    for split, per_label in CLIPS_PER_LABEL.items():
        speakers = sorted(p.stem for p in (OUT_ROOT / "refs" / split).glob("*.wav"))
        for label, n in per_label.items():
            n_schema = len(schema_phrases(label))
            extras = list(PHRASES[label][n_schema:])
            rng.shuffle(extras)
            phrases = list(PHRASES[label][:n_schema]) + extras
            for i in range(n):
                text = phrases[i % len(phrases)]
                jobs.append(
                    {
                        "id": f"{label}_{split}_{i:04d}",
                        "split": split,
                        "label": label,
                        "text": text,
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
    done = set()
    if meta_path.exists():
        with meta_path.open(newline="") as f:
            done = {r["id"] for r in csv.DictReader(f)}
    fields = ["id", "audio_path", "label", "text", "speaker_id", "split", "exaggeration", "cfg_weight"]
    with meta_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not done:
            writer.writeheader()
        for k, job in enumerate(jobs):
            if job["id"] in done:
                continue
            ref = OUT_ROOT / "refs" / job["split"] / f"{job['speaker_id']}.wav"
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
        verb_ok = verbs is None or (bool(hyp) and hyp[0] in verbs and hyp[0] == ref[0])
        r.update(whisper_text=heard, wer=f"{wer:.3f}", qa_pass=str(wer <= QA_MAX_WER and verb_ok))
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
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "data/dataset_manifest.csv")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="synth: only the first N jobs of this shard (smoke test)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
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
