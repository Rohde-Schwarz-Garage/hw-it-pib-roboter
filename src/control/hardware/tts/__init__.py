"""Local text-to-speech engines and shared contracts."""

from __future__ import annotations

from pathlib import Path

from .common import AudioClip, PlaybackCancelledError, SpeakerError, TtsEngine, TtsError
from .espeak import EspeakEngine
from .piper import PiperEngine


def create_tts_engine(source: str, language: str = "de", model_dir: str | Path | None = None, model: str | None = None) -> TtsEngine:
    """Create a speech engine from a source identifier and language."""
    if not isinstance(source, str):
        raise TtsError("Speech engine source must be a string.")
    normalized = source.strip().lower()
    if normalized == "espeak-ng":
        return EspeakEngine(language=language)
    if normalized == "piper":
        return PiperEngine(language, model_dir, model)
    raise TtsError("Unsupported speech engine source.")


__all__ = [
    "AudioClip",
    "EspeakEngine",
    "PiperEngine",
    "PlaybackCancelledError",
    "SpeakerError",
    "TtsEngine",
    "TtsError",
    "create_tts_engine",
]
