"""Google Speech Commands v2 background-noise clips -> unknown_background class.

Source: the official archive's `_background_noise_/` folder — 6 long
(~1 minute) recordings (white/pink noise, a running tap, a dude
"miaowing", a dishwasher, an exercise bike), not the 35 keyword classes.
This is the dataset Section 3/9 already names as the intended source for
the explicit "unknown/background" class the taxonomy requires, but it
was never actually pulled into the pipeline until now.

GSC's own training convention (and the one used here) is to draw random
fixed-length crops from these long files rather than use them whole,
since the model needs many independent-looking negative examples, not
six long ones.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from vcm.dataset.manifest import ManifestRow

WINDOW_S = 1.5  # matches vcm.audio.features.WINDOW_S
SAMPLE_RATE = 16_000  # matches vcm.audio.capture.SAMPLE_RATE
BACKGROUND_LABEL = "unknown_background"  # matches vcm.taxonomy.UNKNOWN_BACKGROUND


def _assign_split(rng: np.random.Generator) -> str:
    r = rng.random()
    if r < 0.8:
        return "train"
    if r < 0.9:
        return "val"
    return "test"


def chop_background_noise(
    source_dir: Path,
    out_dir: Path,
    clips_per_file: int = 100,
    seed: int = 0,
    window_s: float = WINDOW_S,
    sample_rate: int = SAMPLE_RATE,
) -> list[ManifestRow]:
    """Crop `clips_per_file` random fixed-length clips from each .wav in
    `source_dir`, write them to `out_dir`, and return manifest rows."""
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    window_len = int(window_s * sample_rate)

    rows: list[ManifestRow] = []
    for wav_path in sorted(Path(source_dir).glob("*.wav")):
        audio, sr = sf.read(wav_path)
        if sr != sample_rate:
            raise ValueError(f"{wav_path} is {sr}Hz, expected {sample_rate}Hz")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if len(audio) < window_len:
            raise ValueError(f"{wav_path} ({len(audio)} samples) is shorter than one window ({window_len} samples)")

        max_start = len(audio) - window_len
        for i in range(clips_per_file):
            start = int(rng.integers(0, max_start + 1))
            clip = audio[start : start + window_len]
            out_path = out_dir / f"{wav_path.stem}_{i}.wav"
            sf.write(out_path, clip, sample_rate)
            rows.append(
                ManifestRow(
                    audio_path=str(out_path),
                    label=BACKGROUND_LABEL,
                    source="gsc_background",
                    is_synthetic=False,
                    speaker_id="background_noise",
                    split=_assign_split(rng),
                )
            )
    return rows
