"""Bilingual German/English Vosk speech-to-text adapter."""

from __future__ import annotations

import json
import threading
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from vosk import KaldiRecognizer, Model

from .microphone import MicrophoneError, StreamingSpeechToText

Language = Literal["en", "de"]


@dataclass
class _RecognizerState:
    """Recognition state and accumulated evidence for one language."""

    language: Language
    recognizer: KaldiRecognizer
    committed: list[str] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)


class VoskSpeechToText(StreamingSpeechToText):
    """Run English and German Vosk models and select the stronger transcript."""

    def __init__(self, english_model_path: Path, german_model_path: Path) -> None:
        """Load both local Vosk models; no network access or model download is attempted."""
        self._models: dict[Language, Model] = {
            "en": Model(self._model_path(english_model_path)),
            "de": Model(self._model_path(german_model_path)),
        }
        self._states: dict[Language, _RecognizerState] = {}
        self._lock = threading.Lock()
        self._detected_language: Language | None = None

    def transcribe(self, audio: np.ndarray, sample_rate_hz: int) -> str:
        """Transcribe a complete mono audio array after it's done recording. NOT LIVE."""
        with self._lock:
            self._start_session(sample_rate_hz)
            self._accept_audio(audio)
            return self._finish_session()

    def transcribe_wav(self, path: Path) -> str:
        """Transcribe a 16-bit PCM WAV file, downmixing multiple channels."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"WAV file does not exist: {path}")
        with wave.open(str(path), "rb") as wav_file:
            return self._transcribe_wav_file(wav_file)

    def start_session(self, sample_rate_hz: int) -> None:
        """Start a new live recognition session."""
        with self._lock:
            self._start_session(sample_rate_hz)

    def accept_audio(self, audio: np.ndarray) -> None:
        """Feed a mono int16 audio chunk to both language recognizers."""
        with self._lock:
            self._accept_audio(audio)

    def get_partial_text(self) -> str:
        """Return the strongest current partial transcript."""
        with self._lock:
            self._require_session()
            language = self._select_language(partial=True)
            self._detected_language = language
            return self._full_text(self._states[language], partial=True)

    def finish_session(self) -> str:
        """Finish recognition and return the stronger final transcript."""
        with self._lock:
            return self._finish_session()

    def cancel_session(self) -> None:
        """Discard the active session."""
        with self._lock:
            self._states.clear()
            self._detected_language = None

    def get_detected_language(self) -> str | None:
        """Return the language selected for the latest transcript."""
        with self._lock:
            return self._detected_language

    def _start_session(self, sample_rate_hz: int) -> None:
        """Initialize recognizers without acquiring the session lock."""
        if self._states:
            raise MicrophoneError("A Vosk recognition session is already active.")
        if sample_rate_hz <= 0:
            raise ValueError(f"sample_rate_hz must be positive, got {sample_rate_hz}.")
        self._states = {language: _RecognizerState(language, KaldiRecognizer(model, sample_rate_hz)) for language, model in self._models.items()}
        for state in self._states.values():
            state.recognizer.SetWords(True)
            state.recognizer.SetPartialWords(True)
        self._detected_language = None

    def _accept_audio(self, audio: np.ndarray) -> None:
        """Process audio without acquiring the session lock."""
        self._require_session()
        if audio.dtype != np.int16 or audio.ndim != 1:
            raise TypeError("Vosk audio input must be a one-dimensional int16 array.")
        frames = np.ascontiguousarray(audio, dtype="<i2").tobytes()
        for state in self._states.values():
            if state.recognizer.AcceptWaveform(frames):
                self._add_result(state, state.recognizer.Result())

    def _finish_session(self) -> str:
        """Finalize recognition without acquiring the session lock."""
        self._require_session()
        for state in self._states.values():
            self._add_result(state, state.recognizer.FinalResult())
        language = self._select_language(partial=False)
        text = self._full_text(self._states[language], partial=False)
        self._detected_language = language
        self._states.clear()
        return text

    def _add_result(self, state: _RecognizerState, payload: str) -> None:
        """Accumulate text and confidence values from a Vosk result."""
        result = _parse_result(payload)
        text = str(result.get("text", "")).strip()
        if text:
            state.committed.append(text)
        for word in result.get("result", []):
            confidence = word.get("conf")
            if isinstance(confidence, int | float):
                state.confidences.append(float(confidence))

    def _select_language(self, *, partial: bool) -> Language:
        """Select English or German from transcript strength."""
        if partial:
            lengths = {language: len(self._full_text(state, partial=True).split()) for language, state in self._states.items()}
            return max(lengths, key=lambda language: (lengths[language], language == self._detected_language))
        scores = {language: self._confidence_score(state) for language, state in self._states.items()}
        return max(scores, key=scores.get)

    def _full_text(self, state: _RecognizerState, *, partial: bool) -> str:
        """Combine committed text with an optional partial result."""
        text_parts = list(state.committed)
        if partial:
            partial_result = _parse_result(state.recognizer.PartialResult())
            partial_text = str(partial_result.get("partial", "")).strip()
            if partial_text:
                text_parts.append(partial_text)
        return " ".join(text_parts)

    @staticmethod
    def _confidence_score(state: _RecognizerState) -> float:
        """Calculate the language selection score."""
        if state.confidences:
            return sum(state.confidences) / len(state.confidences)
        text = " ".join(state.committed)
        return -1.0 if not text else 0.0

    @staticmethod
    def _model_path(path: Path) -> str:
        """Validate and convert a Vosk model directory."""
        model_path = Path(path)
        if not model_path.is_dir():
            raise FileNotFoundError(f"Vosk model directory does not exist: {model_path}")
        return str(model_path)

    def _require_session(self) -> None:
        """Raise when no recognition session is active."""
        if not self._states:
            raise MicrophoneError("No Vosk recognition session is active.")

    def _transcribe_wav_file(self, wav_file: wave.Wave_read) -> str:
        """Stream an open WAV file into both recognizers."""
        if wav_file.getcomptype() != "NONE" or wav_file.getsampwidth() != 2:
            raise MicrophoneError("WAV input must contain uncompressed 16-bit PCM audio.")
        with self._lock:
            self._start_session(wav_file.getframerate())
            while frames := wav_file.readframes(8000):
                self._accept_audio(_decode_wav_frames(frames, wav_file.getnchannels()))
            return self._finish_session()


def _decode_wav_frames(frames: bytes, channel_count: int) -> np.ndarray:
    """Decode WAV PCM and downmix it to mono int16 samples."""
    samples = np.frombuffer(frames, dtype="<i2")
    if channel_count == 1:
        return samples.astype(np.int16, copy=False)
    return samples.reshape(-1, channel_count).astype(np.int32).mean(axis=1).astype(np.int16)


def _parse_result(payload: str) -> dict[str, object]:
    """Parse and validate a JSON result returned by Vosk."""
    result = json.loads(payload)
    if not isinstance(result, dict):
        raise MicrophoneError("Vosk returned an invalid recognition result.")
    return result
