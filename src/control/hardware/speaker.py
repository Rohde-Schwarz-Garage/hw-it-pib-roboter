"""Synchronous speaker output adapter."""

from __future__ import annotations

import math
import subprocess
import unicodedata
import wave
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from threading import Condition, Event, Lock
from typing import Protocol

import sounddevice

from .tts import AudioClip, SpeakerError, TtsEngine, create_tts_engine
from .tts.common import MAX_AUDIO_SIZE_BYTES, raise_if_cancelled

MAX_TEXT_LENGTH: int = 500
OUTPUT_CHUNK_FRAMES: int = 1024
INVISIBLE_CATEGORIES: frozenset[str] = frozenset({"Cc", "Cf", "Cs"})
_AUDIO_BACKEND_ERRORS: tuple[type[Exception], ...] = (OSError, RuntimeError, TypeError, ValueError, sounddevice.PortAudioError)


class AudioDeviceError(SpeakerError):
    """The playback device or audio stream failed."""


class AudioFileError(SpeakerError):
    """Audio input is inaccessible or not a supported WAV file."""


class _OutputStream(Protocol):
    """Operations used from a sounddevice raw output stream."""

    def start(self) -> None:
        """Start output processing."""
        ...

    def stop(self) -> None:
        """Drain pending output and stop processing."""
        ...

    def abort(self) -> None:
        """Discard pending output and stop processing."""
        ...

    def close(self) -> None:
        """Release the stream and backend resources."""
        ...

    def write(self, data: bytes) -> bool:
        """Write interleaved PCM frames."""
        ...

    @property
    def latency(self) -> float:
        """Return the output latency in seconds."""
        ...


@dataclass
class _ActivePlayback:
    """Track a running output stream; only the playing thread calls into PortAudio."""

    stream: _OutputStream
    stopped: bool = False
    finished: Event = field(default_factory=Event)


@dataclass(eq=False)
class _Request:
    """One queued playback request; ordered by descending priority, then arrival."""

    priority: int
    seq: int
    cancel: Event = field(default_factory=Event)

    def order(self) -> tuple[int, int]:
        """Return the sort key: higher priority first, then first come first served."""
        return -self.priority, self.seq


def _read_pcm(data: bytes) -> tuple[int, int, int, bytes]:
    """Read bounded PCM frames and their source format from a WAV byte stream."""
    if len(data) > MAX_AUDIO_SIZE_BYTES:
        raise AudioFileError("WAV exceeds the size limit.")
    try:
        with wave.open(BytesIO(data), "rb") as reader:
            channels, rate, sample_width = reader.getnchannels(), reader.getframerate(), reader.getsampwidth()
            frames = reader.readframes(reader.getnframes() + 1)
    except (wave.Error, EOFError, OSError, ValueError, RuntimeError) as err:
        raise AudioFileError("Invalid PCM WAV stream.") from err
    return channels, rate, sample_width, frames


def _load_file(path: str | Path) -> AudioClip:
    """Read and validate a bounded regular local .wav file."""
    try:
        candidate = Path(path)
        resolved = candidate.resolve(strict=True)
        if candidate.suffix.lower() != ".wav" or not resolved.is_file():
            raise AudioFileError("Expected a regular .wav file.")
        with resolved.open("rb") as stream:
            data = stream.read(MAX_AUDIO_SIZE_BYTES + 1)
    except (OSError, RuntimeError, TypeError, ValueError) as err:
        raise AudioFileError("Cannot read WAV file.") from err
    return AudioClip(data)


def _clean_char(char: str) -> str:
    """Map whitespace to spaces and drop control or surrogate characters."""
    if char.isspace():
        return " "
    return "" if unicodedata.category(char) in INVISIBLE_CATEGORIES else char


def _clean_text(text: str) -> str:
    """Normalize speech input and reject non-strings, empty text or oversized text."""
    if not isinstance(text, str):
        raise ValueError("Speech input must be a string.")
    if len(text) > MAX_TEXT_LENGTH:
        raise ValueError("Speech input exceeds maximum length.")
    cleaned = " ".join("".join(map(_clean_char, text)).split())
    if not cleaned:
        raise ValueError("Speech input is empty.")
    return cleaned


def _raise_if_cancelled(cancel: Event | None) -> None:
    """Raise PlaybackCancelledError if the request has been cancelled."""
    raise_if_cancelled(cancel)


class SoundDevicePlayer:
    """Play clips synchronously through the default PortAudio output."""

    def __init__(self) -> None:
        """Use the system's default output device."""
        self._lock: Lock = Lock()
        self._start_lock: Lock = Lock()
        self._active: _ActivePlayback | None = None

    def _create_stream(self, channels: int, rate: int, sample_width: int) -> _ActivePlayback:
        """Create an inactive raw output stream."""
        dtype = "uint8" if sample_width == 1 else f"int{sample_width * 8}"
        try:
            stream = sounddevice.RawOutputStream(samplerate=rate, channels=channels, dtype=dtype)
        except _AUDIO_BACKEND_ERRORS as err:
            raise AudioDeviceError("Cannot open audio output stream.") from err
        return _ActivePlayback(stream)

    def _open_stream(self, channels: int, rate: int, sample_width: int) -> _ActivePlayback:
        """Open and start an output stream with the WAV clip's sample format."""
        active = self._create_stream(channels, rate, sample_width)
        try:
            active.stream.start()
        except _AUDIO_BACKEND_ERRORS as err:
            self._close_after_failure(active)
            raise AudioDeviceError("Cannot start audio output stream.") from err
        return active

    def _close_stream(self, active: _ActivePlayback) -> None:
        """Close one stream and translate backend failures."""
        try:
            active.stream.close()
        except _AUDIO_BACKEND_ERRORS as err:
            raise AudioDeviceError("Could not close audio output stream.") from err

    def _close_after_failure(self, active: _ActivePlayback) -> None:
        """Best-effort cleanup that preserves the original stream failure."""
        with suppress(*_AUDIO_BACKEND_ERRORS):
            active.stream.close()

    def _finish_stream(self, active: _ActivePlayback, *, drain: bool) -> None:
        """Drain or abort a stream and translate backend failures."""
        try:
            if drain:
                active.stream.stop()
            else:
                active.stream.abort()
        except _AUDIO_BACKEND_ERRORS as err:
            raise AudioDeviceError("Could not stop audio output stream.") from err

    def _replace_active(self) -> None:
        """Ask a previous playback to stop and wait until its thread has released the stream."""
        with self._lock:
            previous = self._active
            if previous is not None:
                previous.stopped = True
        if previous is not None:
            previous.finished.wait()

    def _start(self, channels: int, rate: int, sample_width: int, cancel: Event | None) -> _ActivePlayback | None:
        """Abort an existing output and start a replacement unless cancelled."""
        with self._start_lock:
            self._replace_active()
            with self._lock:
                if cancel is not None and cancel.is_set():
                    return None
                active = self._open_stream(channels, rate, sample_width)
                self._active = active
                return active

    def _interrupted(self, active: _ActivePlayback, cancel: Event | None) -> bool:
        """Return whether stop() or the cancel event asked the playing thread to end early."""
        with self._lock:
            return active.stopped or (cancel is not None and cancel.is_set())

    def _write(self, active: _ActivePlayback, channels: int, sample_width: int, frames: bytes, cancel: Event | None) -> bool:
        """Write frame-aligned chunks and return whether all frames were submitted."""
        chunk_bytes = OUTPUT_CHUNK_FRAMES * channels * sample_width
        for offset in range(0, len(frames), chunk_bytes):
            if self._interrupted(active, cancel):
                return False
            try:
                active.stream.write(frames[offset : offset + chunk_bytes])
            except _AUDIO_BACKEND_ERRORS as err:
                raise AudioDeviceError("Audio output failed during playback.") from err
        return True

    def _finalize_stream(self, active: _ActivePlayback, *, drain: bool) -> None:
        """Stop and close a stream while preserving the first backend failure."""
        failure: AudioDeviceError | None = None
        try:
            self._finish_stream(active, drain=drain)
        except AudioDeviceError as err:
            failure = err
        try:
            self._close_stream(active)
        except AudioDeviceError as err:
            failure = failure or err
        if failure is not None:
            raise failure

    def _release(self, active: _ActivePlayback) -> None:
        """Clear a finished stream and wake a replacement playback."""
        with self._lock:
            if self._active is active:
                self._active = None
            active.finished.set()

    def _capture_write(
        self, active: _ActivePlayback, channels: int, sample_width: int, frames: bytes, cancel: Event | None
    ) -> tuple[bool, BaseException | None]:
        """Write output while retaining any failure for cleanup and re-raise."""
        try:
            return self._write(active, channels, sample_width, frames, cancel), None
        except BaseException as err:
            return False, err

    def _play_active(self, active: _ActivePlayback, channels: int, sample_width: int, frames: bytes, cancel: Event | None) -> None:
        """Write, then drain or abort and close the stream, all on the playing thread."""
        completed, failure = self._capture_write(active, channels, sample_width, frames, cancel)
        try:
            self._finalize_stream(active, drain=completed and not self._interrupted(active, cancel))
        except AudioDeviceError as err:
            failure = failure or err
        finally:
            self._release(active)
        if failure is not None:
            raise failure

    def play(self, clip: AudioClip, cancel: Event | None = None) -> None:
        """Play a WAV clip synchronously; return quietly if cancelled or stopped."""
        channels, rate, sample_width, frames = _read_pcm(clip.wav_bytes)
        active = self._start(channels, rate, sample_width, cancel)
        if active is not None:
            # PortAudio's ALSA backend drops buffered frames on stop, so push the clip's tail out with silence.
            silence = 0x80 if sample_width == 1 else 0
            padding = bytes([silence]) * (math.ceil(2 * active.stream.latency * rate) * channels * sample_width)
            self._play_active(active, channels, sample_width, frames + padding, cancel)

    def stop(self) -> None:
        """Ask the active playback to abort; the playing thread performs the PortAudio calls."""
        with self._lock:
            if self._active is not None:
                self._active.stopped = True


def _check_volume(value: int) -> None:
    """Reject non-integer percentages and values outside 0 to 100."""
    if type(value) is not int or not 0 <= value <= 100:
        raise ValueError("Volume must be an integer between 0 and 100.")


def _set_system_volume(percent: int) -> None:
    """Set the PipeWire default output volume via WirePlumber."""
    try:
        subprocess.run(
            ["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", f"{percent / 100:.2f}"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as err:
        raise AudioDeviceError("Cannot set system output volume.") from err


def _check_priority(priority: int) -> None:
    """Reject priorities that are not non-negative integers."""
    if type(priority) is not int or priority < 0:
        raise ValueError("Priority must be a non-negative integer.")


def _produce_clip(request: _Request, produce: Callable[[_Request], AudioClip]) -> AudioClip:
    """Prefer a cancellation over an ordinary preparation error."""
    try:
        return produce(request)
    except Exception:
        _raise_if_cancelled(request.cancel)
        raise


class Speaker:
    """Synchronous speaker adapter supporting WAV files, TTS, volume, priorities, skip and stop.

    Overlapping requests are played one after another: higher priority first, equal priority in order
    of arrival. A running clip is never preempted by a later request; only stop() or skip() interrupt it,
    in which case the affected call raises PlaybackCancelledError.
    """

    def __init__(self, engine: TtsEngine | None = None) -> None:
        """Use German eSpeak NG and the default PortAudio output."""
        self._source, self._language = ("espeak-ng", "de") if engine is None else (engine.source, engine.language)
        self._engine: TtsEngine | None = engine
        self._player: SoundDevicePlayer = SoundDevicePlayer()
        self._lock: Lock = Lock()
        self._turn: Condition = Condition(self._lock)
        self._queue: list[_Request] = []
        self._next_seq: int = 0
        self._active: _Request | None = None
        self._volume: int = 50

    def _resolve_engine(self) -> TtsEngine:
        """Instantiate the configured TTS engine lazily on first speech request."""
        with self._lock:
            engine, source, language = self._engine, self._source, self._language
        if engine is None:
            engine = create_tts_engine(source, language)
            with self._lock:
                if self._engine is None and self._source == source and self._language == language:
                    self._engine, self._source = engine, engine.source
        return engine

    def _enqueue(self, priority: int) -> _Request:
        """Reserve a queue position at arrival time, before the clip is prepared."""
        with self._lock:
            self._next_seq += 1
            request = _Request(priority, self._next_seq)
            self._queue.append(request)
            return request

    def _is_next(self, request: _Request) -> bool:
        """Return whether the output is free and the request is first in queue order; lock must be held."""
        return self._active is None and min(self._queue, key=_Request.order) is request

    def _wait_turn(self, request: _Request) -> None:
        """Block until the request may play and claim the output; raise PlaybackCancelledError if it was cancelled."""
        with self._turn:
            self._turn.wait_for(lambda: request.cancel.is_set() or self._is_next(request))
            _raise_if_cancelled(request.cancel)
            self._active = request

    def _output(self, request: _Request, clip: AudioClip) -> None:
        """Play the claimed request; raise PlaybackCancelledError if it was cancelled before or during output."""
        try:
            self._player.play(clip, request.cancel)
        except AudioDeviceError:
            if not request.cancel.is_set():
                raise
        _raise_if_cancelled(request.cancel)

    def _release(self, request: _Request) -> None:
        """Remove a finished, failed or cancelled request and wake waiting requests."""
        with self._turn:
            if request in self._queue:
                self._queue.remove(request)
            if self._active is request:
                self._active = None
            self._turn.notify_all()

    def _run(self, priority: int, produce: Callable[[_Request], AudioClip]) -> None:
        """Queue a request, prepare its clip and play it when it is its turn."""
        _check_priority(priority)
        request = self._enqueue(priority)
        try:
            clip = _produce_clip(request, produce)
            self._wait_turn(request)
            self._output(request, clip)
        finally:
            self._release(request)

    def _synthesize(self, request: _Request, text: str) -> AudioClip:
        """Resolve the engine and synthesize with the request's cancellation event."""
        _raise_if_cancelled(request.cancel)
        engine = self._resolve_engine()
        return engine.synthesize_cancellable(text, request.cancel)

    def play(self, path: str | Path, priority: int = 0) -> None:
        """Validate and play a local WAV file synchronously.

        Args:
            path: Path of a 16-bit PCM ``.wav`` file.
            priority: Non-negative integer; higher values are played first. Defaults to the lowest priority 0.

        Raises:
            PlaybackCancelledError: The clip was cancelled by stop() or skip().
        """
        self._run(priority, lambda _: _load_file(path))

    def say(self, text: str, priority: int = 0) -> None:
        """Synthesize text and play the resulting speech synchronously.

        Args:
            text: Text to speak, at most ``MAX_TEXT_LENGTH`` characters.
            priority: Non-negative integer; higher values are played first. Defaults to the lowest priority 0.

        Raises:
            PlaybackCancelledError: The speech was cancelled by stop() or skip().
        """
        cleaned = _clean_text(text)
        self._run(priority, lambda request: self._synthesize(request, cleaned))

    def stop(self) -> None:
        """Stop active playback and cancel all queued requests, including those still being prepared."""
        with self._turn:
            for request in self._queue:
                request.cancel.set()
            self._queue.clear()
            try:
                self._player.stop()
            finally:
                self._turn.notify_all()

    def skip(self) -> None:
        """Stop only the clip that currently owns the output; queued requests continue in order.

        Requests that are still waiting or being synthesized are not affected. The skipped say()/play() call raises
        PlaybackCancelledError.
        """
        with self._lock:
            if self._active is None:
                return
            self._active.cancel.set()
            self._player.stop()

    def set_tts_source(self, source: str, language: str | None = None, *, model_dir: str | Path | None = None, model: str | None = None) -> None:
        """Switch engines, optionally selecting a language and Piper model directory and name."""
        with self._lock:
            selected_language = self._language if language is None else language
        engine = create_tts_engine(source, selected_language, model_dir, model)
        with self._lock:
            self._source, self._language, self._engine = engine.source, engine.language, engine

    def get_tts_source(self) -> str:
        """Return the configured speech engine source."""
        with self._lock:
            return self._source

    def get_tts_language(self) -> str:
        """Return the configured speech language."""
        with self._lock:
            return self._language

    def set_volume(self, volume_percent: int) -> None:
        """Set the global default output volume percentage."""
        _check_volume(volume_percent)
        with self._lock:
            _set_system_volume(volume_percent)
            self._volume = volume_percent

    def get_volume(self) -> int:
        """Return the configured global output volume percentage."""
        with self._lock:
            return self._volume
