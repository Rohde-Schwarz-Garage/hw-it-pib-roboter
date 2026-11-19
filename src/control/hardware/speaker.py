"""Synchronous speaker output adapter."""

from __future__ import annotations

import json
import logging
import os
import signal
import socket
import struct
import subprocess
import sys
import unicodedata
import wave
from array import array
from collections.abc import Callable
from concurrent.futures import Future
from contextlib import closing
from dataclasses import dataclass, field
from enum import Enum
from http.client import HTTPConnection, HTTPException, HTTPResponse, HTTPSConnection
from io import BytesIO
from pathlib import Path
from threading import Condition, Event, Lock, Thread
from time import monotonic
from typing import Protocol
from urllib.parse import SplitResult, urlsplit

import espeak_ng

LOGGER: logging.Logger = logging.getLogger(__name__)
MAX_AUDIO_SIZE_BYTES: int = 50 * 1024 * 1024
MAX_TEXT_LENGTH: int = 500
HTTP_CHUNK_BYTES: int = 65536
SYNTHESIS_TIMEOUT_S: float = 15.0
STOP_TIMEOUT_S: float = 0.3
SYNTHESIS_POLL_S: float = 0.05
STREAMING_WAV_SIZES: frozenset[int] = frozenset({0x7FFFF000, 0x7FFFFFFF, 0xFFFFFFFF})
WAV_MEDIA_TYPES: frozenset[str] = frozenset({"audio/wav", "audio/wave", "audio/x-wav", "audio/vnd.wave"})
LOOPBACK_HOSTS: frozenset[str] = frozenset({"localhost", "127.0.0.1", "::1"})
INVISIBLE_CATEGORIES: frozenset[str] = frozenset({"Cc", "Cf", "Cs"})


class SpeakerError(Exception):
    """Base error for speaker operations."""


class AudioDeviceError(SpeakerError):
    """The playback device or its process failed."""


class AudioFileError(SpeakerError):
    """Audio input is inaccessible or not a supported WAV file."""


class TtsError(SpeakerError):
    """A speech engine could not be configured or synthesize audio."""


class _HttpWorkerCleanupError(TtsError):
    """A cancelled or timed-out HTTP synthesis worker could not be stopped."""


class VolumeError(SpeakerError):
    """A volume setting violates the hearing protection limits."""


class InputValidationError(SpeakerError):
    """Speech input is invalid."""


class CancelReason(Enum):
    """Why a say() or play() request ended without being played completely."""

    STOPPED = "stopped"
    SKIPPED = "skipped"
    VOLUME_LIMIT = "volume_limit"


class PlaybackCancelledError(SpeakerError):
    """A request was cancelled on purpose by stop(), skip() or a lowered volume limit before it completed."""

    def __init__(self, reason: CancelReason) -> None:
        """Store the cancellation reason."""
        super().__init__(f"Playback cancelled: {reason.value}.")
        self.reason: CancelReason = reason


@dataclass(frozen=True)
class AudioClip:
    """A complete WAV stream containing signed 16-bit PCM."""

    wav_bytes: bytes


class TtsEngine(Protocol):
    """Convert plain text to playable audio."""

    def synthesize(self, text: str) -> AudioClip:
        """Synthesize one utterance."""
        ...


class AudioPlayer(Protocol):
    """Play one clip synchronously and allow interruption."""

    def play(self, clip: AudioClip, cancel: Event | None = None) -> None:
        """Play audio until completion or interruption.

        Implementations must check ``cancel`` atomically with respect to ``stop`` right before output
        starts and return without playing if it is set.
        """
        ...

    def stop(self) -> None:
        """Stop the current output, if any."""
        ...


@dataclass
class _ActivePlayback:
    """Track a running player subprocess and whether stop was requested."""

    process: subprocess.Popen[bytes]
    stopped: bool = False


@dataclass(eq=False)
class _Request:
    """One queued playback request; ordered by descending priority, then arrival."""

    priority: int
    seq: int
    cancel: Event = field(default_factory=Event)
    reason: CancelReason | None = None

    def order(self) -> tuple[int, int]:
        """Return the sort key: higher priority first, then first come first served."""
        return -self.priority, self.seq

    def cancel_with(self, reason: CancelReason) -> None:
        """Mark the request as cancelled; the first reason wins. The speaker lock must be held."""
        if not self.cancel.is_set():
            self.reason = reason
            self.cancel.set()

    def raise_if_cancelled(self) -> None:
        """Raise PlaybackCancelledError if the request has been cancelled."""
        if self.cancel.is_set():
            raise PlaybackCancelledError(self.reason or CancelReason.STOPPED)


def _pcm_data_size(data: bytes) -> int:
    """Validate format chunks and retain the declared PCM data byte length."""
    offset = 12
    while offset + 8 <= len(data):
        length = int.from_bytes(data[offset + 4 : offset + 8], "little")
        if data[offset : offset + 4] == b"data":
            return length
        if data[offset : offset + 4] == b"fmt ":
            encoding = data[offset + 8 : offset + 10]
            if encoding not in {b"\x01\x00", b"\xfe\xff"} or length < 16 or data[offset + 22 : offset + 24] != b"\x10\x00":
                raise AudioFileError("Expected 16-bit PCM.")
        offset += 8 + length + length % 2
    raise AudioFileError("Missing WAV data chunk.")


def _validate_frames(frames: bytes, channels: int, rate: int, expected: int, *, streaming: bool) -> None:
    """Reject empty, misaligned, out-of-range or truncated PCM payloads."""
    if not frames or len(frames) % (channels * 2) or not rate:
        raise AudioFileError("Invalid WAV frames or sample rate.")
    if channels * 2 > 0xFFFF or rate * channels * 2 > 0xFFFFFFFF:
        raise AudioFileError("Invalid WAV frame size or byte rate.")
    if len(frames) != expected and not (streaming and expected in STREAMING_WAV_SIZES):
        raise AudioFileError("Incomplete WAV payload.")


def _read_pcm(data: bytes, *, streaming: bool = False) -> tuple[int, int, bytes]:
    """Decode bounded 16-bit PCM frames from a WAV byte stream."""
    if len(data) > MAX_AUDIO_SIZE_BYTES:
        raise AudioFileError("WAV exceeds the size limit.")
    expected = _pcm_data_size(data)
    try:
        with wave.open(BytesIO(data), "rb") as reader:
            channels, rate = reader.getnchannels(), reader.getframerate()
            frames = reader.readframes(reader.getnframes() + 1)
    except (wave.Error, EOFError, OSError, ValueError, RuntimeError, struct.error) as err:
        raise AudioFileError("Invalid PCM WAV stream.") from err
    _validate_frames(frames, channels, rate, expected, streaming=streaming)
    return channels, rate, frames


def _encode_pcm(channels: int, rate: int, frames: bytes) -> AudioClip:
    """Write a canonical WAV header matching the actual PCM frame count."""
    with BytesIO() as output:
        with wave.open(output, "wb") as writer:
            writer.setparams((channels, 2, rate, 0, "NONE", "not compressed"))
            writer.writeframes(frames)
        return AudioClip(output.getvalue())


def _attenuate_samples(frames: bytes, percent: int) -> bytes:
    """Scale little-endian 16-bit signed PCM samples by a percentage."""
    samples = array("h", frames)
    if sys.byteorder != "little":
        samples.byteswap()
    for index, value in enumerate(samples):
        samples[index] = int(value * percent / 100)
    if sys.byteorder != "little":
        samples.byteswap()
    return samples.tobytes()


def _scale(clip: AudioClip, percent: int) -> AudioClip:
    """Return a canonical WAV clip scaled to the given volume percentage."""
    channels, rate, frames = _read_pcm(clip.wav_bytes)
    if percent == 0:
        frames = bytes(len(frames))
    elif percent != 100:
        frames = _attenuate_samples(frames, percent)
    return _encode_pcm(channels, rate, frames)


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
    _read_pcm(data)
    return AudioClip(data)


def _clean_char(char: str) -> str:
    """Map whitespace to spaces and drop control or surrogate characters."""
    if char.isspace():
        return " "
    return "" if unicodedata.category(char) in INVISIBLE_CATEGORIES else char


def _clean_text(text: str) -> str:
    """Normalize speech input and reject non-strings, empty text or oversized text."""
    if not isinstance(text, str):
        raise InputValidationError("Speech input must be a string.")
    if len(text) > MAX_TEXT_LENGTH:
        raise InputValidationError("Speech input exceeds maximum length.")
    cleaned = " ".join("".join(map(_clean_char, text)).split())
    if not cleaned:
        raise InputValidationError("Speech input is empty.")
    return cleaned


def _normalize_tts_language(language: str) -> str:
    """Validate and normalize an eSpeak language or HTTP service language."""
    if not isinstance(language, str):
        raise TtsError("Invalid speech language.")
    normalized = language.strip()
    if not normalized or normalized.startswith("-"):
        raise TtsError("Invalid speech language.")
    return normalized


def _tts_clip(data: bytes) -> AudioClip:
    """Validate synthesizer output and normalize streaming WAV headers."""
    try:
        return _encode_pcm(*_read_pcm(data, streaming=True))
    except AudioFileError as err:
        raise TtsError("Speech engine returned invalid audio.") from err


def _check_synthesis_cancel(cancel: Event | None) -> None:
    """Raise when synthesis has been cancelled."""
    if cancel is not None and cancel.is_set():
        raise PlaybackCancelledError(CancelReason.STOPPED)


def _communicate_http_worker(process: subprocess.Popen[bytes], data: bytes, cancel: Event | None, deadline: float) -> tuple[bytes, bytes]:
    """Poll the HTTP synthesis worker while observing cancellation and the total deadline."""
    pending: bytes | None = data
    while True:
        _check_synthesis_cancel(cancel)
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise TtsError("Speech engine timed out.")
        try:
            return process.communicate(pending, timeout=min(remaining, SYNTHESIS_POLL_S))
        except subprocess.TimeoutExpired:
            pending = None


def _kill_http_worker(process: subprocess.Popen[bytes]) -> None:
    """Force-stop an active HTTP synthesis worker without risking a reaped process group."""
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    elif process.poll() is None:
        process.kill()


def _cleanup_http_worker(process: subprocess.Popen[bytes]) -> None:
    """Stop the HTTP synthesis worker tree and reap it within a bounded cleanup period."""
    if process.returncode is not None:
        return
    try:
        _kill_http_worker(process)
        process.communicate(timeout=STOP_TIMEOUT_S)
    except subprocess.TimeoutExpired as err:
        raise _HttpWorkerCleanupError("HTTP synthesis worker cleanup timed out.") from err
    except (OSError, ValueError, subprocess.SubprocessError) as err:
        raise _HttpWorkerCleanupError("Could not stop HTTP synthesis worker.") from err


def _http_worker_result(process: subprocess.Popen[bytes], data: bytes, cancel: Event | None, deadline: float) -> bytes:
    """Collect a successful HTTP worker result or translate its failure."""
    output, stderr = _communicate_http_worker(process, data, cancel, deadline)
    _check_synthesis_cancel(cancel)
    if process.returncode:
        detail = stderr.decode("utf-8", errors="replace").strip()
        raise TtsError(f"Speech engine process failed: {detail or process.returncode}")
    return output


def _run_http_worker(command: list[str], data: bytes, cancel: Event | None) -> bytes:
    """Run a bounded HTTP synthesis worker and reap it on cancellation or timeout."""
    _check_synthesis_cancel(cancel)
    deadline = monotonic() + SYNTHESIS_TIMEOUT_S
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=os.name == "posix",
        )
        try:
            return _http_worker_result(process, data, cancel, deadline)
        finally:
            _cleanup_http_worker(process)
    except (OSError, ValueError, subprocess.SubprocessError) as err:
        raise TtsError("HTTP synthesis worker failed.") from err


_ESPEAK_SYNTHESIS_LOCK: Lock = Lock()


class _EspeakCapture:
    """Collect bounded 16-bit PCM from the eSpeak synthesis callback."""

    def __init__(self, cancel: Event | None) -> None:
        """Prepare one cancellable synchronous synthesis capture."""
        self._cancel: Event | None = cancel
        self._deadline: float = monotonic() + SYNTHESIS_TIMEOUT_S
        self._frames: bytearray = bytearray()
        self._error: SpeakerError | None = None

    def __call__(self, wav: bytes, num_samples: int, event: object) -> int:
        """Collect an audio chunk, returning one to stop eSpeak on cancellation or failure."""
        del num_samples, event
        try:
            _check_synthesis_cancel(self._cancel)
            if monotonic() >= self._deadline:
                raise TtsError("Speech engine timed out.")
            if len(self._frames) + len(wav) > MAX_AUDIO_SIZE_BYTES:
                raise TtsError("Speech engine output exceeds the size limit.")
            self._frames.extend(wav)
            return 0
        except (PlaybackCancelledError, TtsError) as err:
            self._error = err
            return 1

    def raise_if_failed(self) -> None:
        """Re-raise an error recorded by the callback, if any."""
        if self._error is not None:
            raise self._error

    def result(self) -> bytes:
        """Return complete PCM frames or re-raise an error recorded by the callback."""
        self.raise_if_failed()
        if not self._frames or len(self._frames) % 2:
            raise TtsError("Speech engine returned invalid audio.")
        return bytes(self._frames)


def _discard_espeak_audio(wav: bytes, num_samples: int, event: object) -> int:
    """Discard callbacks left over after a completed synchronous request."""
    del wav, num_samples, event
    return 0


def _run_espeak_synthesis(language: str, text: str, capture: _EspeakCapture) -> tuple[int, int, bytes]:
    """Run one callback-driven eSpeak request while its global state is locked."""
    sample_rate = espeak_ng.initialize(output=espeak_ng.espeak_AUDIO_OUTPUT.AUDIO_OUTPUT_SYNCHRONOUS)
    if not espeak_ng.set_voice_by_properties(name=language):
        raise TtsError("Speech engine could not select the configured language.")
    espeak_ng.set_synth_callback(capture)
    try:
        espeak_ng.synth(text)
    finally:
        espeak_ng.set_synth_callback(_discard_espeak_audio)
    return 1, sample_rate, capture.result()


def _synthesize_espeak(language: str, text: str, cancel: Event | None) -> AudioClip:
    """Synthesize and encode PCM while translating library errors."""
    _check_synthesis_cancel(cancel)
    capture = _EspeakCapture(cancel)
    try:
        with _ESPEAK_SYNTHESIS_LOCK:
            return _encode_pcm(*_run_espeak_synthesis(language, text, capture))
    except (PlaybackCancelledError, TtsError):
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as err:
        capture.raise_if_failed()
        raise TtsError("Speech engine failed to synthesize audio.") from err


class EspeakEngine:
    """Synthesize speech synchronously via the local eSpeak NG library."""

    def __init__(self, voice: str = "de", binary: str = "espeak-ng", language: str | None = None) -> None:
        """Select a language or voice while retaining the legacy binary argument."""
        del binary
        self._source: str = "espeak-ng"
        self._language: str = _normalize_tts_language(voice if language is None else language)

    @property
    def source(self) -> str:
        """Return the eSpeak NG library source name."""
        return self._source

    @property
    def language(self) -> str:
        """Return the selected eSpeak language or voice."""
        return self._language

    def synthesize(self, text: str) -> AudioClip:
        """Synthesize eSpeak audio and return a validated WAV clip."""
        return self.synthesize_cancellable(text, None)

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
    return _tts_clip(data)


class _HttpCall:
    """Run an HTTP request in a process so DNS and connection attempts can be stopped."""

    def __init__(self, url: SplitResult, target: str, body: bytes, headers: dict[str, str]) -> None:
        """Prepare the request."""
        self._url: SplitResult = url
        self._request: tuple[str, bytes, dict[str, str]] = (target, body, headers)

    def run(self, cancel: Event | None = None) -> AudioClip:
        """Return the clip or stop and reap the worker on cancellation or timeout."""
        target, body, headers = self._request
        request = json.dumps({"url": self._url.geturl(), "target": target, "body": body.decode(), "headers": headers}).encode()
        command = [sys.executable, "-I", "-B", str(Path(__file__).resolve()), "--tts-http-worker"]
        result = _run_http_worker(command, request, cancel)
        if not result:
            raise TtsError("Speech service returned no result.")
        if result[:1] != b"\x00":
            raise TtsError(result[1:].decode("utf-8", errors="replace"))
        return AudioClip(result[1:])

    def _send(self) -> AudioClip:
        """Connect, POST the request and read the validated WAV response within the total deadline."""
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


def _http_worker() -> None:
    """Read a private request from stdin and return a tagged result on stdout."""
    try:
        request = json.loads(sys.stdin.buffer.read())
        call = _HttpCall(_endpoint(request["url"]), request["target"], request["body"].encode(), request["headers"])
        result = b"\x00" + call._send().wav_bytes
    except (TtsError, HTTPException, OSError, ValueError) as err:
        result = b"\x01" + str(err).encode("utf-8", errors="replace")
    sys.stdout.buffer.write(result)


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

    def synthesize(self, text: str) -> AudioClip:
        """POST the utterance to the speech endpoint and return a WAV clip within the total deadline."""
        return self.synthesize_cancellable(text, None)

    def synthesize_cancellable(self, text: str, cancel: Event | None) -> AudioClip:
        """POST the utterance with cancellation support."""
        body = json.dumps({"text": text, "language": self._language}).encode()
        return _HttpCall(self._url, self._target(), body, self._headers()).run(cancel)


def create_tts_engine(source: str, language: str = "de") -> EspeakEngine | HttpEngine:
    """Create an eSpeak or HTTP speech engine from a source identifier and language."""
    if not isinstance(source, str):
        raise TtsError("Speech engine source must be a string.")
    normalized = source.strip().lower()
    if normalized in {"espeak", "espeak-ng"}:
        return EspeakEngine(binary=normalized, language=language)
    if "://" in normalized:
        return HttpEngine(source.strip(), language=language)
    raise TtsError("Unsupported speech engine source.")


def _engine_source(engine: TtsEngine | None) -> str:
    """Describe an engine the way set_tts_source accepts it; custom engines fall back to their class name."""
    if engine is None:
        return "espeak-ng"
    return getattr(engine, "source", type(engine).__name__)


def _normalize_command(command: str | list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Validate a player command given as program name, list or tuple and return it as a tuple."""
    args = [command] if isinstance(command, str) else command
    if not isinstance(args, (list, tuple)) or not args or not all(isinstance(part, str) for part in args) or not args[0].strip():
        raise AudioDeviceError("Invalid audio player command.")
    return tuple(args)


def _terminate(process: subprocess.Popen[bytes]) -> None:
    """Terminate a child process and escalate to kill if it does not exit promptly."""
    try:
        try:
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=STOP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    except (OSError, subprocess.SubprocessError) as err:
        raise AudioDeviceError("Could not stop audio player.") from err


def _write_clip(process: subprocess.Popen[bytes], payload: bytes) -> None:
    """Pipe the audio payload into the player and require a zero exit status."""
    try:
        with process:
            process.communicate(payload)
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, process.args)
    except (OSError, ValueError, subprocess.SubprocessError) as err:
        raise AudioDeviceError("Audio player failed during playback.") from err


class SubprocessPlayer:
    """Play clips synchronously on the default PipeWire output."""

    def __init__(self, command: str | list[str] | tuple[str, ...] | None = None) -> None:
        """Use PipeWire by default; an explicit command receives complete WAV data on stdin."""
        self._command: tuple[str, ...] | None = None if command is None else _normalize_command(command)
        self._lock: Lock = Lock()
        self._active: _ActivePlayback | None = None

    def _player_input(self, clip: AudioClip) -> tuple[tuple[str, ...], bytes]:
        """Send headerless samples to pw-play with the clip's actual timing and channel count."""
        if self._command is not None:
            return self._command, clip.wav_bytes
        channels, rate, frames = _read_pcm(clip.wav_bytes)
        return ("pw-play", "--raw", "--format=s16", f"--rate={rate}", f"--channels={channels}", "-"), frames

    def _stop_locked(self) -> None:
        """Stop and reap the active child process while holding the lock."""
        if self._active is not None:
            active = self._active
            active.stopped = True
            try:
                _terminate(active.process)
            except AudioDeviceError:
                active.stopped = False
                raise
            self._active = None

    def _start(self, command: tuple[str, ...], cancel: Event | None) -> _ActivePlayback | None:
        """Stop any running player process before spawning the replacement, unless cancelled.

        The cancel check shares the lock with stop(), so a cancellation set before stop() can never be missed.
        """
        with self._lock:
            self._stop_locked()
            if cancel is not None and cancel.is_set():
                return None
            try:
                proc = subprocess.Popen(list(command), stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except (OSError, ValueError) as err:
                raise AudioDeviceError(f"Cannot start audio player: {command[0]}.") from err
            self._active = _ActivePlayback(proc)
            return self._active

    def _finish(self, active: _ActivePlayback) -> None:
        """Clear the active slot when the matching playback completes."""
        with self._lock:
            if self._active is active:
                self._active = None

    def play(self, clip: AudioClip, cancel: Event | None = None) -> None:
        """Play a WAV clip synchronously; return quietly if cancelled or stopped intentionally."""
        command, payload = self._player_input(clip)
        active = self._start(command, cancel)
        if active is None:
            return
        try:
            _write_clip(active.process, payload)
        except AudioDeviceError:
            if not active.stopped:
                raise
        finally:
            self._finish(active)

    def stop(self) -> None:
        """Interrupt active playback, if any."""
        with self._lock:
            self._stop_locked()


def _check_volume(value: int, limits: tuple[int, int]) -> None:
    """Reject non-integer percentages and values outside the inclusive limits."""
    if type(value) is not int or not limits[0] <= value <= limits[1]:
        raise VolumeError("Volume must be an integer within the configured limits.")


def _check_priority(priority: int) -> None:
    """Reject priorities that are not non-negative integers."""
    if type(priority) is not int or priority < 0:
        raise InputValidationError("Priority must be a non-negative integer.")


def _custom_synthesis_worker(engine: TtsEngine, text: str, cancel: Event, result: Future[AudioClip]) -> None:
    """Publish a legacy engine's result or exception to its waiting caller."""
    try:
        result.set_result(engine.synthesize(text))
    except BaseException as err:
        result.set_exception(err)
        if cancel.is_set():
            LOGGER.warning("Custom speech engine failed after cancellation.", exc_info=err)


def _custom_synthesis(engine: TtsEngine, text: str, cancel: Event) -> AudioClip:
    """Wait for a legacy engine without blocking a cancelled caller."""
    _check_synthesis_cancel(cancel)
    result: Future[AudioClip] = Future()
    Thread(target=_custom_synthesis_worker, args=(engine, text, cancel, result), name="tts-custom", daemon=True).start()
    while not result.done():
        if cancel.wait(SYNTHESIS_POLL_S):
            _check_synthesis_cancel(cancel)
    _check_synthesis_cancel(cancel)
    return result.result()


def _produce_clip(request: _Request, produce: Callable[[_Request], AudioClip]) -> AudioClip:
    """Prefer a cancellation over an ordinary preparation error, but report failed cleanup."""
    try:
        return produce(request)
    except _HttpWorkerCleanupError:
        raise
    except Exception:
        request.raise_if_cancelled()
        raise


def _speaker_language(engine: TtsEngine | None, language: str | None) -> str:
    """Select a language that is compatible with an optional custom engine."""
    engine_language = getattr(engine, "language", None)
    if engine_language is not None:
        engine_language = _normalize_tts_language(engine_language)
    selected_language = _normalize_tts_language(language if language is not None else engine_language or "de")
    if engine_language is not None and selected_language != engine_language:
        raise TtsError("Speaker language does not match the speech engine language.")
    return selected_language


class Speaker:
    """Synchronous speaker adapter supporting WAV files, TTS, volume limits, priorities and stop.

    Overlapping requests are played one after another: higher priority first, equal priority in order
    of arrival. A running clip is never preempted by a later request; only stop(), skip() or a lowered
    volume limit interrupt it, in which case the affected call raises PlaybackCancelledError.
    """

    def __init__(self, engine: TtsEngine | None = None, player: AudioPlayer | None = None, language: str | None = None) -> None:
        """Use eSpeak NG and PipeWire by default, resolving eSpeak on first speech call.

        Raises:
            TtsError: ``language`` differs from the language of the given ``engine``.
        """
        self._source, self._language = _engine_source(engine), _speaker_language(engine, language)
        self._engine: TtsEngine | None = engine
        self._player: AudioPlayer = SubprocessPlayer() if player is None else player
        self._lock: Lock = Lock()
        self._turn: Condition = Condition(self._lock)
        self._queue: list[_Request] = []
        self._next_seq: int = 0
        self._active: _Request | None = None
        self._active_volume: int | None = None
        self._limits: tuple[int, int] = (0, 100)
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
            request.raise_if_cancelled()
            self._active = request

    def _prepare(self, request: _Request, clip: AudioClip) -> AudioClip:
        """Scale the clip outside the lock and commit its volume if it still respects the maximum limit."""
        with self._lock:
            volume = self._volume
        while True:
            scaled = _scale(clip, volume)
            with self._lock:
                request.raise_if_cancelled()
                if volume <= self._limits[1]:
                    self._active_volume = volume
                    return scaled
                volume = self._volume

    def _output(self, request: _Request, clip: AudioClip) -> None:
        """Play the claimed request; raise PlaybackCancelledError if it was cancelled before or during output."""
        scaled = self._prepare(request, clip)
        try:
            self._player.play(scaled, request.cancel)
        except AudioDeviceError:
            if not request.cancel.is_set():
                raise
        request.raise_if_cancelled()

    def _release(self, request: _Request) -> None:
        """Remove a finished, failed or cancelled request and wake waiting requests."""
        with self._turn:
            if request in self._queue:
                self._queue.remove(request)
            if self._active is request:
                self._active, self._active_volume = None, None
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
        """Use an engine's cancellation support or adapt a legacy engine."""
        request.raise_if_cancelled()
        engine = self._resolve_engine()
        request.raise_if_cancelled()
        synthesize = getattr(engine, "synthesize_cancellable", None)
        if callable(synthesize):
            return synthesize(text, request.cancel)
        return _custom_synthesis(engine, text, request.cancel)

    def play(self, path: str | Path, priority: int = 0) -> None:
        """Validate and play a local WAV file synchronously.

        Args:
            path: Path of a 16-bit PCM ``.wav`` file.
            priority: Non-negative integer; higher values are played first. Defaults to the lowest priority 0.

        Raises:
            PlaybackCancelledError: The clip was cancelled by stop(), skip() or a lowered volume limit; see ``reason``.
        """
        self._run(priority, lambda _: _load_file(path))

    def say(self, text: str, priority: int = 0) -> None:
        """Synthesize text and play the resulting speech synchronously.

        Args:
            text: Text to speak, at most ``MAX_TEXT_LENGTH`` characters.
            priority: Non-negative integer; higher values are played first. Defaults to the lowest priority 0.

        Raises:
            PlaybackCancelledError: The speech was cancelled by stop(), skip() or a lowered volume limit; see ``reason``.

        A custom engine without ``synthesize_cancellable(text, cancel)`` may continue running after cancellation.
        """
        cleaned = _clean_text(text)
        self._run(priority, lambda request: self._synthesize(request, cleaned))

    def stop(self) -> None:
        """Stop active playback and cancel all queued requests, including those still being prepared."""
        with self._turn:
            for request in self._queue:
                request.cancel_with(CancelReason.STOPPED)
            self._queue.clear()
            try:
                self._player.stop()
            finally:
                self._turn.notify_all()

    def skip(self) -> None:
        """Stop only the clip that currently owns the output; queued requests continue in order.

        Requests that are still waiting or being synthesized are not affected. The skipped say()/play() call raises
        PlaybackCancelledError with reason SKIPPED.

        """
        with self._lock:
            if self._active is None:
                return
            self._active.cancel_with(CancelReason.SKIPPED)
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
        """Set the playback volume percentage for subsequent clips."""
        with self._lock:
            _check_volume(volume_percent, self._limits)
            self._volume = volume_percent

    def get_volume(self) -> int:
        """Return the configured playback volume percentage."""
        with self._lock:
            return self._volume

    def set_volume_limits(self, min_percent: int, max_percent: int) -> None:
        """Update hearing protection limits, clamping volume and stopping the active clip if it is too loud.

        Queued requests are kept; they are scaled with the clamped volume when it is their turn.
        """
        _check_volume(min_percent, (0, 100))
        _check_volume(max_percent, (min_percent, 100))
        with self._lock:
            if self._active is not None and self._active_volume is not None and self._active_volume > max_percent:
                self._active.cancel_with(CancelReason.VOLUME_LIMIT)
                self._player.stop()
            self._limits = (min_percent, max_percent)
            self._volume = max(min_percent, min(self._volume, max_percent))

    def get_volume_limits(self) -> tuple[int, int]:
        """Return the inclusive minimum and maximum volume percentages."""
        with self._lock:
            return self._limits


if __name__ == "__main__" and sys.argv[1:] == ["--tts-http-worker"]:
    _http_worker()
