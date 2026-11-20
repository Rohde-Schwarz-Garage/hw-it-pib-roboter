"""Shared contracts and helpers for local text-to-speech engines."""

from __future__ import annotations

import subprocess
import wave
from contextlib import suppress
from dataclasses import dataclass
from io import BytesIO
from threading import Event
from time import monotonic
from typing import Protocol

MAX_AUDIO_SIZE_BYTES: int = 50 * 1024 * 1024
SYNTHESIS_TIMEOUT_S: float = 15.0


class SpeakerError(Exception):
    """Base error for speaker operations."""


class TtsError(SpeakerError):
    """A speech engine could not be configured or synthesize audio."""


class PlaybackCancelledError(SpeakerError):
    """A request was cancelled on purpose by stop() or skip() before it completed."""


@dataclass(frozen=True)
class AudioClip:
    """A complete WAV stream containing PCM audio."""

    wav_bytes: bytes


class TtsEngine(Protocol):
    """Convert plain text to playable audio with cooperative cancellation."""

    @property
    def source(self) -> str:
        """Return the source identifier accepted by create_tts_engine()."""
        ...

    @property
    def language(self) -> str:
        """Return the synthesis language."""
        ...

    def synthesize_cancellable(self, text: str, cancel: Event | None) -> AudioClip:
        """Synthesize one utterance, raising PlaybackCancelledError once ``cancel`` is set."""
        ...


def normalize_tts_language(language: str) -> str:
    """Validate and normalize a speech language."""
    if not isinstance(language, str):
        raise TtsError("Invalid speech language.")
    normalized = language.strip()
    if not normalized or normalized.startswith("-"):
        raise TtsError("Invalid speech language.")
    return normalized


def raise_if_cancelled(cancel: Event | None) -> None:
    """Raise PlaybackCancelledError if the request has been cancelled."""
    if cancel is not None and cancel.is_set():
        raise PlaybackCancelledError("Playback cancelled.")


def encode_pcm(channels: int, rate: int, frames: bytes) -> AudioClip:
    """Write a canonical WAV header matching the actual PCM frame count."""
    with BytesIO() as output:
        with wave.open(output, "wb") as writer:
            writer.setparams((channels, 2, rate, 0, "NONE", "not compressed"))
            writer.writeframes(frames)
        return AudioClip(output.getvalue())


def wait_for_process(process: subprocess.Popen[bytes], cancel: Event | None) -> None:
    """Wait for a speech process while honoring cancellation and the synthesis deadline."""
    deadline = monotonic() + SYNTHESIS_TIMEOUT_S
    while True:
        raise_if_cancelled(cancel)
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise TtsError("Speech engine timed out.")
        try:
            if process.wait(timeout=min(remaining, 0.05)) != 0:
                raise TtsError("Speech engine failed to synthesize audio.")
            return
        except subprocess.TimeoutExpired:
            continue


def terminate_process(process: subprocess.Popen[bytes]) -> None:
    """Terminate a still-running speech process and reap it."""
    if process.poll() is None:
        with suppress(OSError):
            process.terminate()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            with suppress(OSError):
                process.kill()
            process.wait()


def read_tts_frames(reader: wave.Wave_read) -> tuple[int, bytes]:
    """Read bounded mono 16-bit PCM from an opened speech-engine WAV file."""
    channels, rate, sample_width = reader.getnchannels(), reader.getframerate(), reader.getsampwidth()
    if channels != 1 or sample_width != 2 or reader.getnframes() * channels * sample_width > MAX_AUDIO_SIZE_BYTES:
        raise TtsError("Speech engine output exceeds the size limit or has an invalid format.")
    frames = reader.readframes(reader.getnframes() + 1)
    if not frames or len(frames) % 2:
        raise TtsError("Speech engine returned invalid audio.")
    return rate, frames


def read_tts_output(path: str) -> tuple[int, bytes]:
    """Read bounded mono 16-bit PCM written by a speech engine."""
    try:
        with wave.open(path, "rb") as reader:
            return read_tts_frames(reader)
    except TtsError:
        raise
    except (wave.Error, EOFError, OSError, ValueError, RuntimeError) as err:
        raise TtsError("Speech engine returned invalid audio.") from err
