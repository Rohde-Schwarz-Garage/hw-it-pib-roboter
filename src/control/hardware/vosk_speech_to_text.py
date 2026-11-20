"""Bilingual German/English Vosk speech-to-text adapter."""

from __future__ import annotations

import json
import logging
import threading
import wave
from collections import deque
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Literal

import numpy as np
from vosk import KaldiRecognizer, Model

from control.hardware.microphone import MicrophoneError, StreamingSpeechToText, Utterance

Language = Literal["en", "de"]
_WAV_CHUNK_FRAMES: int = 8000
_logger: logging.Logger = logging.getLogger(__name__)


class VoskSpeechToText(StreamingSpeechToText):
    """Recognize English and German utterances with two local Vosk models.

    Both models hear the same audio. When either one detects the end of an utterance, both are
    finalized together, and the result with the higher mean word confidence becomes the utterance.
    All methods are thread-safe.
    """

    def __init__(self, english_model_path: Path, german_model_path: Path, max_pending_utterances: int = 256) -> None:
        """Load both local models; no network access or model download is attempted.

        Args:
            english_model_path: Directory of the English Vosk model.
            german_model_path: Directory of the German Vosk model.
            max_pending_utterances: Finished utterances kept until popped; the oldest are dropped beyond this.

        Raises:
            FileNotFoundError: If a model directory does not exist.
            MicrophoneError: If Vosk cannot load a model from an existing directory.
            ValueError: If ``max_pending_utterances`` is not positive.
        """
        if max_pending_utterances <= 0:
            raise ValueError(f"max_pending_utterances must be positive, got {max_pending_utterances}.")
        self._models: dict[Language, Model] = {"en": _load_model(english_model_path), "de": _load_model(german_model_path)}
        self._recognizers: dict[Language, KaldiRecognizer] = {}
        self._pending_utterances: deque[Utterance] = deque(maxlen=max_pending_utterances)
        self._detected_language: Language | None = None
        self._lock: threading.Lock = threading.Lock()

    def transcribe(self, audio: np.ndarray, sample_rate_hz: int) -> str:
        """Transcribe a complete recording.

        Args:
            audio: One-dimensional int16 PCM samples.
            sample_rate_hz: Sample rate of ``audio``.

        Returns:
            Space-joined text of all recognized utterances.

        Raises:
            TypeError: If ``audio`` is not one-dimensional int16.
            ValueError: If ``sample_rate_hz`` is not positive.
            MicrophoneError: If a live session is active; it is left untouched.
        """
        _validate_audio(audio)
        return self._transcribe_chunks(sample_rate_hz, [audio])

    def transcribe_wav(self, path: Path) -> str:
        """Transcribe an uncompressed 16-bit PCM WAV file; multiple channels are downmixed.

        Args:
            path: WAV file to transcribe.

        Returns:
            Space-joined text of all recognized utterances.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            MicrophoneError: If the file is not a supported WAV file, or a live session is active.
        """
        wav_path = Path(path)
        if not wav_path.is_file():
            raise FileNotFoundError(f"WAV file does not exist: {wav_path}")
        try:
            with wave.open(str(wav_path), "rb") as wav_file:
                _validate_wav(wav_file)
                return self._transcribe_chunks(wav_file.getframerate(), _read_wav_chunks(wav_file))
        except (wave.Error, EOFError) as error:
            raise MicrophoneError(f"Invalid WAV file {wav_path}: {error}") from error

    def start_session(self, sample_rate_hz: int) -> None:
        """Start a live session.

        Args:
            sample_rate_hz: Sample rate of the audio passed to ``accept_audio``.

        Raises:
            ValueError: If ``sample_rate_hz`` is not positive.
            MicrophoneError: If a session is already active.
        """
        with self._lock:
            self._start_session(sample_rate_hz)

    def accept_audio(self, audio: np.ndarray) -> None:
        """Feed a chunk to both recognizers; an utterance is queued when one of them detects its end.

        Args:
            audio: One-dimensional int16 PCM samples.

        Raises:
            TypeError: If ``audio`` is not one-dimensional int16; the session stays usable.
            MicrophoneError: If no session is active.
        """
        with self._lock:
            self._accept_audio(audio)

    def get_partial_text(self) -> str:
        """Return the provisional text of the utterance currently being spoken.

        Returns:
            The partial transcript with more words, or an empty string.

        Raises:
            MicrophoneError: If no session is active.
        """
        with self._lock:
            self._require_session()
            partials = {language: _partial_text(recognizer) for language, recognizer in self._recognizers.items()}
            return partials[self._pick_language({language: len(text.split()) for language, text in partials.items()})]

    def pop_final_utterances(self) -> list[Utterance]:
        """Remove and return all finished utterances, oldest first; empty if there are none."""
        with self._lock:
            utterances = list(self._pending_utterances)
            self._pending_utterances.clear()
            return utterances

    def finish_session(self) -> str:
        """Finalize the current utterance and end the session, also if this raises.

        Returns:
            Space-joined text of all utterances not yet popped.

        Raises:
            MicrophoneError: If no session is active.
        """
        with self._lock:
            return self._finish_session()

    def cancel_session(self) -> None:
        """Discard the active session and its pending utterances; does nothing if none is active."""
        with self._lock:
            self._reset_session()

    def get_detected_language(self) -> str | None:
        """Return the language of the latest utterance, or ``None`` if nothing was recognized."""
        with self._lock:
            return self._detected_language

    def _transcribe_chunks(self, sample_rate_hz: int, chunks: Iterable[np.ndarray]) -> str:
        """Run a complete one-shot session; any failure ends it so the engine stays usable."""
        with self._lock:
            self._start_session(sample_rate_hz)
            try:
                for chunk in chunks:
                    self._accept_audio(chunk)
            except Exception:
                self._reset_session()
                raise
            return self._finish_session()

    def _start_session(self, sample_rate_hz: int) -> None:
        """Create fresh recognizers; the caller holds the lock."""
        if self._recognizers:
            raise MicrophoneError("A Vosk recognition session is already active.")
        if sample_rate_hz <= 0:
            raise ValueError(f"sample_rate_hz must be positive, got {sample_rate_hz}.")
        self._recognizers = {language: _create_recognizer(model, sample_rate_hz) for language, model in self._models.items()}
        self._pending_utterances.clear()
        self._detected_language = None

    def _accept_audio(self, audio: np.ndarray) -> None:
        """Feed both recognizers; the caller holds the lock."""
        self._require_session()
        _validate_audio(audio)
        frames = np.ascontiguousarray(audio, dtype="<i2").tobytes()
        endpoints = {language: recognizer.AcceptWaveform(frames) for language, recognizer in self._recognizers.items()}
        if any(endpoints.values()):
            # Flushing the other model too keeps both recognizers aligned on the next utterance.
            payloads = {language: rec.Result() if endpoints[language] else rec.FinalResult() for language, rec in self._recognizers.items()}
            self._queue_utterance(payloads)

    def _finish_session(self) -> str:
        """Queue the last utterance, drain the queue and always end the session; the caller holds the lock."""
        self._require_session()
        try:
            self._queue_utterance({language: recognizer.FinalResult() for language, recognizer in self._recognizers.items()})
            return " ".join(utterance.text for utterance in self._pending_utterances)
        finally:
            self._reset_session()

    def _queue_utterance(self, payloads: dict[Language, str]) -> None:
        """Queue the more confident of two simultaneous results, unless both are empty."""
        results = {language: _parse_result(payload) for language, payload in payloads.items()}
        language = self._pick_language({language: _confidence_score(result) for language, result in results.items()})
        text = _result_text(results[language])
        if not text:
            return
        if len(self._pending_utterances) == self._pending_utterances.maxlen:
            _logger.warning("Dropping the oldest Vosk utterance because the pending queue is full.")
        self._pending_utterances.append(Utterance(text=text, language=language))
        self._detected_language = language

    def _pick_language(self, scores: dict[Language, float]) -> Language:
        """Return the higher-scoring language; a tie keeps the latest language, then English."""
        if scores["en"] == scores["de"]:
            return self._detected_language or "en"
        return "en" if scores["en"] > scores["de"] else "de"

    def _require_session(self) -> None:
        """Raise when no session is active."""
        if not self._recognizers:
            raise MicrophoneError("No Vosk recognition session is active.")

    def _reset_session(self) -> None:
        """Drop the recognizers and pending utterances."""
        self._recognizers = {}
        self._pending_utterances.clear()


def _load_model(path: Path) -> Model:
    """Load a Vosk model, translating Vosk's generic load error into ``MicrophoneError``."""
    model_path = Path(path)
    if not model_path.is_dir():
        raise FileNotFoundError(f"Vosk model directory does not exist: {model_path}")
    try:
        return Model(str(model_path))
    except Exception as error:
        raise MicrophoneError(f"Failed to load Vosk model at {model_path}: {error}") from error


def _create_recognizer(model: Model, sample_rate_hz: int) -> KaldiRecognizer:
    """Create a recognizer that reports per-word confidences."""
    recognizer = KaldiRecognizer(model, sample_rate_hz)
    recognizer.SetWords(True)
    recognizer.SetPartialWords(True)
    return recognizer


def _validate_audio(audio: np.ndarray) -> None:
    """Raise ``TypeError`` unless ``audio`` is a one-dimensional int16 array."""
    if not isinstance(audio, np.ndarray) or audio.dtype != np.int16 or audio.ndim != 1:
        raise TypeError("Vosk audio input must be a one-dimensional int16 array.")


def _validate_wav(wav_file: wave.Wave_read) -> None:
    """Reject WAV files that are compressed, not 16-bit, or have no sample rate or channels."""
    if wav_file.getcomptype() != "NONE" or wav_file.getsampwidth() != 2:
        raise MicrophoneError("WAV input must contain uncompressed 16-bit PCM audio.")
    if wav_file.getframerate() <= 0 or wav_file.getnchannels() <= 0:
        raise MicrophoneError("WAV input must have a positive sample rate and channel count.")


def _read_wav_chunks(wav_file: wave.Wave_read) -> Iterator[np.ndarray]:
    """Yield the WAV file as mono int16 chunks."""
    while frames := wav_file.readframes(_WAV_CHUNK_FRAMES):
        yield _decode_wav_frames(frames, wav_file.getnchannels())


def _decode_wav_frames(frames: bytes, channel_count: int) -> np.ndarray:
    """Decode 16-bit PCM frames to mono, dropping a trailing incomplete frame of a truncated file."""
    frame_bytes = 2 * channel_count
    samples = np.frombuffer(frames[: len(frames) - len(frames) % frame_bytes], dtype="<i2")
    if channel_count == 1:
        return samples.astype(np.int16, copy=False)
    return samples.reshape(-1, channel_count).astype(np.int32).mean(axis=1).astype(np.int16)


def _partial_text(recognizer: KaldiRecognizer) -> str:
    """Return the trimmed provisional text of a recognizer."""
    partial = _parse_result(recognizer.PartialResult()).get("partial", "")
    return partial.strip() if isinstance(partial, str) else ""


def _result_text(result: dict[str, object]) -> str:
    """Return the trimmed final text of a recognizer result."""
    text = result.get("text", "")
    return text.strip() if isinstance(text, str) else ""


def _confidence_score(result: dict[str, object]) -> float:
    """Return the mean word confidence; 0.0 for text without confidences and -1.0 for no text."""
    words = result.get("result", [])
    confidences = (
        [word["conf"] for word in words if isinstance(word, dict) and isinstance(word.get("conf"), int | float)] if isinstance(words, list) else []
    )
    if confidences:
        return sum(confidences) / len(confidences)
    return 0.0 if _result_text(result) else -1.0


def _parse_result(payload: str) -> dict[str, object]:
    """Parse a Vosk JSON result.

    Raises:
        MicrophoneError: If the result is not a JSON object.
    """
    result = json.loads(payload)
    if not isinstance(result, dict):
        raise MicrophoneError("Vosk returned an invalid recognition result.")
    return result
