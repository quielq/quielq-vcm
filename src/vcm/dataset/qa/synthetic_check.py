"""Generic QA gate for synthetic command audio, any generator.

Generalizes a transcribe-and-compare pattern shared within the class:
transcribe each generated clip and check whether the transcription
matches the phrase it was supposed to say, producing a confidence score
so low-confidence clips get routed to human spot-check instead of being
blindly trusted. Deliberately generator-agnostic, it screens output from
any synthesis pipeline (voice-cloning, precedented public datasets, or
anything else), since none of them should skip this check.

The transcription step here is an **offline dataset-curation tool only**
(Section 9's own framing) — it never runs at inference time; the deployed
VCM classifies audio directly into an intent, no transcription step.
"""

# Acknowledgment: this generalizes a transcribe-and-compare QA tool built
# and shared by a classmate as part of the class's collective dataset
# effort — see DATASET.md's Acknowledgments section.

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Protocol


class Transcriber(Protocol):
    def transcribe(self, audio_path: Path) -> str: ...


class FasterWhisperTranscriber:
    """Real backend — requires `pip install faster-whisper` (not a default
    dependency here, it pulls in a multi-hundred-MB model on first use)."""

    def __init__(self, model_size: str = "small") -> None:
        from faster_whisper import WhisperModel  # type: ignore[import-not-found]

        self._model = WhisperModel(model_size)

    def transcribe(self, audio_path: Path) -> str:
        segments, _ = self._model.transcribe(str(audio_path))
        return " ".join(segment.text for segment in segments).strip()


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def match_confidence(expected_text: str, transcribed_text: str) -> float:
    """0.0-1.0 similarity between the intended phrase and what was transcribed."""
    return SequenceMatcher(None, _normalize(expected_text), _normalize(transcribed_text)).ratio()


@dataclass(frozen=True)
class QaResult:
    audio_path: str
    expected_text: str
    transcribed_text: str
    confidence: float
    passed: bool


def screen_clip(
    audio_path: Path,
    expected_text: str,
    transcriber: Transcriber,
    threshold: float = 0.8,
) -> QaResult:
    transcribed = transcriber.transcribe(audio_path)
    confidence = match_confidence(expected_text, transcribed)
    return QaResult(
        audio_path=str(audio_path),
        expected_text=expected_text,
        transcribed_text=transcribed,
        confidence=confidence,
        passed=confidence >= threshold,
    )


def screen_batch(
    pairs: list[tuple[Path, str]],
    transcriber: Transcriber,
    threshold: float = 0.8,
) -> list[QaResult]:
    """Screen a batch of (audio_path, expected_text) pairs. Results with
    passed=False are the ones to route to human spot-check."""
    return [screen_clip(path, text, transcriber, threshold) for path, text in pairs]
