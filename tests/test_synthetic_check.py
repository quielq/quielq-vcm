from pathlib import Path

from vcm.dataset.qa.synthetic_check import match_confidence, screen_batch, screen_clip


class _FakeTranscriber:
    def __init__(self, canned: dict[str, str]):
        self._canned = canned

    def transcribe(self, audio_path: Path) -> str:
        return self._canned[str(audio_path)]


def test_match_confidence_identical_text_is_perfect():
    assert match_confidence("Lights off", "lights off") == 1.0  # case-insensitive


def test_match_confidence_unrelated_text_is_low():
    assert match_confidence("Lights off", "play some music please") < 0.5


def test_screen_clip_passes_above_threshold():
    transcriber = _FakeTranscriber({"clip.wav": "lights off"})
    result = screen_clip(Path("clip.wav"), "Lights off", transcriber, threshold=0.8)
    assert result.passed
    assert result.confidence == 1.0


def test_screen_clip_fails_below_threshold():
    transcriber = _FakeTranscriber({"clip.wav": "completely different words"})
    result = screen_clip(Path("clip.wav"), "Lights off", transcriber, threshold=0.8)
    assert not result.passed


def test_screen_batch_flags_only_low_confidence_clips():
    transcriber = _FakeTranscriber({"good.wav": "play music", "bad.wav": "xyz"})
    results = screen_batch(
        [(Path("good.wav"), "Play music"), (Path("bad.wav"), "Play music")],
        transcriber,
        threshold=0.8,
    )
    passed = {r.audio_path: r.passed for r in results}
    assert passed["good.wav"] is True
    assert passed["bad.wav"] is False
