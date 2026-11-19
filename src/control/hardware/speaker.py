"""Synchronous speaker output adapter."""

from __future__ import annotations

import json
import logging
import math
import os
import socket
import subprocess
import unicodedata
import wave
from collections.abc import Callable
from contextlib import closing, suppress
from dataclasses import dataclass, field
from http.client import HTTPConnection, HTTPException, HTTPResponse, HTTPSConnection
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Condition, Event, Lock
from time import monotonic
from typing import Protocol
from urllib.parse import SplitResult, urlsplit

import espeakng
import sounddevice

LOGGER: logging.Logger = logging.getLogger(__name__)
MAX_AUDIO_SIZE_BYTES: int = 50 * 1024 * 1024
MAX_TEXT_LENGTH: int = 500
HTTP_CHUNK_BYTES: int = 65536
SYNTHESIS_TIMEOUT_S: float = 15.0
OUTPUT_CHUNK_FRAMES: int = 1024
WAV_MEDIA_TYPES: frozenset[str] = frozenset({"audio/wav", "audio/wave", "audio/x-wav", "audio/vnd.wave"})
LOOPBACK_HOSTS: frozenset[str] = frozenset({"localhost", "127.0.0.1", "::1"})
INVISIBLE_CATEGORIES: frozenset[str] = frozenset({"Cc", "Cf", "Cs"})
_AUDIO_BACKEND_ERRORS: tuple[type[Exception], ...] = (OSError, RuntimeError, TypeError, ValueError, sounddevice.PortAudioError)


class SpeakerError(Exception):
    """Base error for speaker operations."""


class AudioDeviceError(SpeakerError):
    """The playback device or audio stream failed."""


class AudioFileError(SpeakerError):
    """Audio input is inaccessible or not a supported WAV file."""


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


def _encode_pcm(channels: int, rate: int, frames: bytes) -> AudioClip:
    """Write a canonical WAV header matching the actual PCM frame count."""
    with BytesIO() as output:
        with wave.open(output, "wb") as writer:
            writer.setparams((channels, 2, rate, 0, "NONE", "not compressed"))
            writer.writeframes(frames)
        return AudioClip(output.getvalue())


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


def _normalize_tts_language(language: str) -> str:
    """Validate and normalize an eSpeak language or HTTP service language."""
    if not isinstance(language, str):
        raise TtsError("Invalid speech language.")
    normalized = language.strip()
    if not normalized or normalized.startswith("-"):
        raise TtsError("Invalid speech language.")
    return normalized


def _raise_if_cancelled(cancel: Event | None) -> None:
    """Raise PlaybackCancelledError if the request has been cancelled."""
    if cancel is not None and cancel.is_set():
        raise PlaybackCancelledError("Playback cancelled.")


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


def _wait_for_espeak(process: subprocess.Popen[bytes], cancel: Event | None) -> None:
    """Wait for the eSpeak command while honoring cancellation and the synthesis deadline."""
    deadline = monotonic() + SYNTHESIS_TIMEOUT_S
    while True:
        _raise_if_cancelled(cancel)
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise TtsError("Speech engine timed out.")
        try:
            if process.wait(timeout=min(remaining, 0.05)) != 0:
                raise TtsError("Speech engine failed to synthesize audio.")
            return
        except subprocess.TimeoutExpired:
            continue


def _terminate_espeak(process: subprocess.Popen[bytes]) -> None:
    """Terminate a still-running eSpeak command and reap it."""
    if process.poll() is None:
        with suppress(OSError):
            process.terminate()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            with suppress(OSError):
                process.kill()
            process.wait()


def _read_espeak_frames(reader: wave.Wave_read) -> tuple[int, bytes]:
    """Read bounded mono 16-bit PCM from an opened eSpeak WAV file."""
    channels, rate, sample_width = reader.getnchannels(), reader.getframerate(), reader.getsampwidth()
    if channels != 1 or sample_width != 2 or reader.getnframes() * channels * sample_width > MAX_AUDIO_SIZE_BYTES:
        raise TtsError("Speech engine output exceeds the size limit or has an invalid format.")
    frames = reader.readframes(reader.getnframes() + 1)
    if not frames or len(frames) % 2:
        raise TtsError("Speech engine returned invalid audio.")
    return rate, frames


def _read_espeak_output(path: Path) -> tuple[int, bytes]:
    """Read bounded mono 16-bit PCM written by the eSpeak command."""
    try:
        with wave.open(str(path), "rb") as reader:
            return _read_espeak_frames(reader)
    except TtsError:
        raise
    except (wave.Error, EOFError, OSError, ValueError, RuntimeError) as err:
        raise TtsError("Speech engine returned invalid audio.") from err


def _run_espeak_command(language: str, text: str, output_path: Path, cancel: Event | None) -> None:
    """Run the wrapper-generated WAV export command with cancellation and timeout checks."""
    command = espeakng.Speaker(voice=language).generate_command(text, export_path=str(output_path))
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        _wait_for_espeak(process, cancel)
    finally:
        _terminate_espeak(process)


def _run_espeak_synthesis(language: str, text: str, cancel: Event | None) -> tuple[int, int, bytes]:
    """Run one WAV-exporting eSpeak command while its global state is locked."""
    runtime = _initialize_espeak()
    if not _espeak_voice_known(runtime.voices, language):
        raise TtsError("Speech engine could not select the configured language.")
    _raise_if_cancelled(cancel)
    with TemporaryDirectory() as output_directory:
        output_path = Path(output_directory) / "speech.wav"
        _run_espeak_command(language, text, output_path, cancel)
        rate, frames = _read_espeak_output(output_path)
    return 1, rate, frames


def _synthesize_espeak(language: str, text: str, cancel: Event | None) -> AudioClip:
    """Synthesize and encode PCM while translating library errors."""
    _raise_if_cancelled(cancel)
    try:
        with _ESPEAK_SYNTHESIS_LOCK:
            return _encode_pcm(*_run_espeak_synthesis(language, text, cancel))
    except (PlaybackCancelledError, TtsError):
        raise
    except (OSError, RuntimeError, SystemError, TypeError, ValueError, wave.Error) as err:
        raise TtsError("Speech engine failed to synthesize audio.") from err


class EspeakEngine:
    """Synthesize speech synchronously via the local eSpeak NG library."""

    def __init__(self, language: str = "de") -> None:
        """Select an eSpeak language or voice."""
        self._language: str = _normalize_tts_language(language)

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


def _endpoint(source: str) -> SplitResult:
    """Validate an absolute HTTP or HTTPS URL without credentials or fragments."""
    try:
        url = urlsplit(source)
        if url.scheme not in {"http", "https"} or not url.hostname or url.port == 0:
            raise ValueError
        if url.username is not None or url.password is not None or url.fragment:
            raise ValueError
        if any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in source):
            raise ValueError
        return url
    except ValueError as err:
        raise TtsError("Invalid HTTP speech endpoint.") from err


def _read_http_body(sock: socket.socket, response: HTTPResponse, deadline: float) -> bytes:
    """Read a bounded HTTP response body, re-arming the socket timeout to the remaining deadline per chunk."""
    chunks = bytearray()
    while len(chunks) <= MAX_AUDIO_SIZE_BYTES:
        if response.isclosed():
            return bytes(chunks)
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise TimeoutError("Speech response timed out.")
        sock.settimeout(remaining)
        block = response.read1(min(HTTP_CHUNK_BYTES, MAX_AUDIO_SIZE_BYTES + 1 - len(chunks)))
        if not block:
            return bytes(chunks)
        chunks.extend(block)
    raise TtsError("Speech response exceeds the size limit.")


def _response_audio(sock: socket.socket, response: HTTPResponse, deadline: float) -> AudioClip:
    """Validate HTTP status, media type and payload bounds without following redirects."""
    if not 200 <= response.status < 300 or response.headers.get_content_type() not in WAV_MEDIA_TYPES:
        raise TtsError("Speech service did not return a WAV response.")
    expected = response.length
    if expected is not None and expected > MAX_AUDIO_SIZE_BYTES:
        raise TtsError("Speech response exceeds the size limit.")
    data = _read_http_body(sock, response, deadline)
    if expected is not None and len(data) != expected:
        raise TtsError("Incomplete speech response.")
    return AudioClip(data)


class _HttpCall:
    """Send one HTTP synthesis request."""

    def __init__(self, url: SplitResult, target: str, body: bytes, headers: dict[str, str]) -> None:
        """Prepare the request."""
        self._url: SplitResult = url
        self._request: tuple[str, bytes, dict[str, str]] = (target, body, headers)

    def run(self, cancel: Event | None = None) -> AudioClip:
        """Return the clip; cancellation is checked before and after the blocking request."""
        _raise_if_cancelled(cancel)
        try:
            clip = self._send()
        except (HTTPException, OSError, ValueError) as err:
            raise TtsError(str(err)) from err
        _raise_if_cancelled(cancel)
        return clip

    def _send(self) -> AudioClip:
        """Connect, POST the request and read the validated WAV response; DNS lookup is not bounded by the timeout."""
        deadline = monotonic() + SYNTHESIS_TIMEOUT_S
        connection = HTTPSConnection if self._url.scheme == "https" else HTTPConnection
        target, body, headers = self._request
        with closing(connection(self._url.netloc, timeout=SYNTHESIS_TIMEOUT_S)) as client:
            client.connect()
            sock = client.sock
            if sock is None:
                raise OSError("Speech service did not provide a socket.")
            client.request("POST", target, body=body, headers=headers)
            with client.getresponse() as response:
                return _response_audio(sock, response, deadline)


class HttpEngine:
    """Synthesize speech synchronously by posting JSON to an HTTP endpoint."""

    def __init__(self, url: str, api_key_env: str | None = "TTS_API_KEY", language: str = "de") -> None:
        """Bind synthesis requests to one validated HTTP or HTTPS endpoint."""
        if not isinstance(url, str):
            raise TtsError("Invalid HTTP speech endpoint.")
        self._url: SplitResult = _endpoint(url)
        self._source: str = url
        self._api_key_env: str | None = api_key_env
        self._language: str = _normalize_tts_language(language)

    @property
    def source(self) -> str:
        """Return the endpoint URL."""
        return self._source

    @property
    def language(self) -> str:
        """Return the language sent with synthesis requests."""
        return self._language

    def _headers(self) -> dict[str, str]:
        """Include the optional bearer token only over HTTPS or loopback HTTP."""
        headers = {"Content-Type": "application/json", "Accept": "audio/wav"}
        key = os.environ.get(self._api_key_env) if self._api_key_env else None
        if key and (self._url.scheme == "https" or self._url.hostname in LOOPBACK_HOSTS):
            headers["Authorization"] = f"Bearer {key}"
        elif key:
            LOGGER.warning("API key withheld because the speech endpoint is not encrypted.")
        return headers

    def _target(self) -> str:
        """Build the HTTP origin-form request target without corrupting leading slashes."""
        path = self._url.path or "/"
        return f"{path}?{self._url.query}" if self._url.query else path

    def synthesize_cancellable(self, text: str, cancel: Event | None) -> AudioClip:
        """POST the utterance and return a WAV clip within the total deadline, with cancellation support."""
        body = json.dumps({"text": text, "language": self._language}).encode()
        return _HttpCall(self._url, self._target(), body, self._headers()).run(cancel)


def create_tts_engine(source: str, language: str = "de") -> EspeakEngine | HttpEngine:
    """Create an eSpeak or HTTP speech engine from a source identifier and language."""
    if not isinstance(source, str):
        raise TtsError("Speech engine source must be a string.")
    normalized = source.strip().lower()
    if normalized == "espeak-ng":
        return EspeakEngine(language=language)
    if "://" in normalized:
        return HttpEngine(source.strip(), language=language)
    raise TtsError("Unsupported speech engine source.")


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

    def set_tts_source(self, source: str, language: str | None = None) -> None:
        """Validate and switch the speech engine, optionally changing its language."""
        with self._lock:
            selected_language = self._language if language is None else language
        engine = create_tts_engine(source, selected_language)
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
