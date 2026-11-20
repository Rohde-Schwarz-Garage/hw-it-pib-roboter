"""eSpeak NG text-to-speech engine."""

from __future__ import annotations

import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Lock

import espeakng

from .common import (
    AudioClip,
    PlaybackCancelledError,
    TtsError,
    encode_pcm,
    normalize_tts_language,
    raise_if_cancelled,
    read_tts_output,
    terminate_process,
    wait_for_process,
)

_ESPEAK_SYNTHESIS_LOCK: Lock = Lock()


@dataclass
class _EspeakRuntime:
    """Process-wide eSpeak voice list; only touched while _ESPEAK_SYNTHESIS_LOCK is held."""

    voices: tuple[str, ...] = ()


_ESPEAK_RUNTIME: _EspeakRuntime = _EspeakRuntime()


def _initialize_espeak() -> _EspeakRuntime:
    """Cache the voice list exposed by the espeakng wrapper."""
    if not _ESPEAK_RUNTIME.voices:
        _ESPEAK_RUNTIME.voices = tuple(voice.lower() for voice in espeakng.Speaker.list_voices())
    return _ESPEAK_RUNTIME


def _espeak_voice_known(voices: tuple[str, ...], language: str) -> bool:
    """Check a language against the voice codes exposed by espeakng."""
    base = language.split("+", 1)[0].lower()
    if not base or not base.isascii():
        return False
    return base in voices


def _run_espeak_command(language: str, text: str, output_path: Path, cancel: Event | None) -> None:
    """Run the wrapper-generated WAV export command with cancellation and timeout checks."""
    command = espeakng.Speaker(voice=language).generate_command(text, export_path=str(output_path))
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_for_process(process, cancel)
    finally:
        terminate_process(process)


def _run_espeak_synthesis(language: str, text: str, cancel: Event | None) -> tuple[int, int, bytes]:
    """Run one WAV-exporting eSpeak command while its global state is locked."""
    runtime = _initialize_espeak()
    if not _espeak_voice_known(runtime.voices, language):
        raise TtsError("Speech engine could not select the configured language.")
    raise_if_cancelled(cancel)
    with TemporaryDirectory() as output_directory:
        output_path = Path(output_directory) / "speech.wav"
        _run_espeak_command(language, text, output_path, cancel)
        rate, frames = read_tts_output(str(output_path))
    return 1, rate, frames


def _synthesize_espeak(language: str, text: str, cancel: Event | None) -> AudioClip:
    """Synthesize and encode PCM while translating library errors."""
    raise_if_cancelled(cancel)
    try:
        with _ESPEAK_SYNTHESIS_LOCK:
            return encode_pcm(*_run_espeak_synthesis(language, text, cancel))
    except (PlaybackCancelledError, TtsError):
        raise
    except (OSError, RuntimeError, SystemError, TypeError, ValueError, wave.Error) as err:
        raise TtsError("Speech engine failed to synthesize audio.") from err


class EspeakEngine:
    """Synthesize speech synchronously via the local eSpeak NG library."""

    def __init__(self, language: str = "de") -> None:
        """Select an eSpeak language or voice."""
        self._language: str = normalize_tts_language(language)

    @property
    def source(self) -> str:
        """Return the eSpeak NG library source name."""
        return "espeak-ng"

    @property
    def language(self) -> str:
        """Return the selected eSpeak language or voice."""
        return self._language

    def synthesize_cancellable(self, text: str, cancel: Event | None) -> AudioClip:
        """Synthesize eSpeak audio with callback-based cancellation and bounds."""
        return _synthesize_espeak(self._language, text, cancel)
