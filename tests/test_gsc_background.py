import numpy as np
import soundfile as sf

from vcm.dataset.sources.gsc_background import BACKGROUND_LABEL, chop_background_noise


def _write_noise_wav(path, duration_s=5.0, sample_rate=16000, seed=0):
    rng = np.random.default_rng(seed)
    audio = rng.uniform(-0.1, 0.1, size=int(duration_s * sample_rate)).astype("float32")
    sf.write(path, audio, sample_rate)


def test_chop_background_noise_produces_expected_count(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _write_noise_wav(source_dir / "noise_a.wav")
    _write_noise_wav(source_dir / "noise_b.wav", seed=1)

    rows = chop_background_noise(source_dir, tmp_path / "out", clips_per_file=10, seed=0)

    assert len(rows) == 20  # 2 files x 10 clips
    assert all(r.label == BACKGROUND_LABEL for r in rows)
    assert all(r.source == "gsc_background" for r in rows)
    assert all(r.is_synthetic is False for r in rows)


def test_chop_background_noise_writes_real_files_of_correct_length(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _write_noise_wav(source_dir / "noise.wav", duration_s=5.0, sample_rate=16000)

    rows = chop_background_noise(source_dir, tmp_path / "out", clips_per_file=3, seed=0, window_s=1.5, sample_rate=16000)

    for row in rows:
        audio, sr = sf.read(row.audio_path)
        assert sr == 16000
        assert len(audio) == int(1.5 * 16000)


def test_chop_background_noise_assigns_all_three_splits_given_enough_clips():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        source_dir = Path(td) / "source"
        source_dir.mkdir()
        _write_noise_wav(source_dir / "noise.wav", duration_s=10.0)
        rows = chop_background_noise(source_dir, Path(td) / "out", clips_per_file=200, seed=0)
        splits = {r.split for r in rows}
        assert splits == {"train", "val", "test"}


def test_chop_background_noise_rejects_wrong_sample_rate(tmp_path):
    import pytest

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    sf.write(source_dir / "noise.wav", np.zeros(8000, dtype="float32"), 8000)

    with pytest.raises(ValueError):
        chop_background_noise(source_dir, tmp_path / "out", clips_per_file=1, sample_rate=16000)


def test_chop_background_noise_rejects_clip_shorter_than_window(tmp_path):
    import pytest

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    sf.write(source_dir / "tooshort.wav", np.zeros(100, dtype="float32"), 16000)

    with pytest.raises(ValueError):
        chop_background_noise(source_dir, tmp_path / "out", clips_per_file=1)
