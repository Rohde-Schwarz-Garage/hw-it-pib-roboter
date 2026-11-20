"""Microphone input adapter for the Seeed ReSpeaker USB 4 Mic Array (6-channel firmware)."""

from __future__ import annotations

import logging
import math
import struct
import threading
import time
import wave
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sounddevice as sd
import usb.core
import usb.util

USB_VENDOR_ID: int = 0x2886
USB_PRODUCT_ID: int = 0x0018
USB_TIMEOUT_MS: int = 8000
SAMPLE_RATE_HZ: int = 16000
CHANNEL_COUNT: int = 6
PROCESSED_CHANNEL: int = 0
RAW_MIC_CHANNELS: tuple[int, ...] = (1, 2, 3, 4)
PLAYBACK_CHANNEL: int = 5
SAMPLE_DTYPE: str = "int16"
SAMPLE_WIDTH_BYTES: int = 2
FULL_SCALE: float = 32768.0
MIN_DBFS: float = -120.0
LED_COUNT: int = 12
MAX_LED_BRIGHTNESS: int = 31

_REQUEST_TYPE_OUT: int = usb.util.CTRL_OUT | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE
_REQUEST_TYPE_IN: int = usb.util.CTRL_IN | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE
_VENDOR_REQUEST: int = 0

_TUNING_READ_FLAG: int = 0x80
_TUNING_INT_FLAG: int = 0x40
_TUNING_RESPONSE_LENGTH: int = 8

_LED_INDEX: int = 0x1C
_LED_SET_COLOR: int = 0x01
_LED_LISTEN: int = 0x02
_LED_SPEAK: int = 0x03
_LED_THINK: int = 0x04
_LED_SPIN: int = 0x05
_LED_SHOW: int = 0x06
_LED_BRIGHTNESS: int = 0x20

_logger: logging.Logger = logging.getLogger(__name__)

AudioCallback = Callable[[np.ndarray], None]


class MicrophoneError(Exception):
    """Base error for all microphone failures."""


class DeviceNotFoundError(MicrophoneError):
    """Raised when the ReSpeaker array is not connected or not accessible."""


@dataclass(frozen=True)
class AudioConfig:
    """Capture settings for the audio backend."""

    sample_rate_hz: int = SAMPLE_RATE_HZ
    channel_count: int = CHANNEL_COUNT
    block_size_frames: int = 1024
    device_name: str = "ReSpeaker 4 Mic Array"


@dataclass(frozen=True)
class LedConfig:
    """Layout and colors of the LED ring; adjust to match the mounted board."""

    zero_deg_led_index: int = 0
    clockwise: bool = True
    wakeup_color: tuple[int, int, int] = (0, 0, 255)


# --- Audio capture -----------------------------------------------------------


class AudioBackend(ABC):
    """Abstract audio capture interface, so the capture library can be swapped or mocked."""

    @property
    @abstractmethod
    def config(self) -> AudioConfig:
        """Capture settings used by this backend."""

    @abstractmethod
    def open(self) -> None:
        """Open the input device and start capturing."""

    @abstractmethod
    def close(self) -> None:
        """Release the input device."""

    @abstractmethod
    def read(self, frame_count: int) -> np.ndarray:
        """Read a blocking chunk of audio.

        Args:
            frame_count: Number of frames to read.

        Returns:
            int16 array of shape (frame_count, channel_count).
        """


class SounddeviceBackend(AudioBackend):
    """Default audio backend based on the ``sounddevice`` library."""

    def __init__(self, config: AudioConfig) -> None:
        """Store the capture configuration; the device is opened in ``open``."""
        self._config: AudioConfig = config
        self._stream: sd.InputStream | None = None

    @property
    def config(self) -> AudioConfig:
        """Capture settings used by this backend."""
        return self._config

    def open(self) -> None:
        """Find the ReSpeaker input device by name and start an input stream.

        Raises:
            DeviceNotFoundError: If no matching input device exists.
            MicrophoneError: If PortAudio cannot open the stream.
        """
        if self._stream is not None:
            return
        try:
            stream = sd.InputStream(
                device=self._find_device_index(),
                channels=self._config.channel_count,
                samplerate=self._config.sample_rate_hz,
                blocksize=self._config.block_size_frames,
                dtype=SAMPLE_DTYPE,
            )
            _start_or_close(stream)
        except sd.PortAudioError as error:
            raise MicrophoneError(f"Could not open audio input: {error}") from error
        self._stream = stream

    def close(self) -> None:
        """Stop and close the input stream; does nothing if it is not open.

        The backend is considered closed afterwards even if PortAudio reports an error.

        Raises:
            MicrophoneError: If PortAudio fails to stop or close the stream.
        """
        stream, self._stream = self._stream, None
        if stream is None:
            return
        try:
            try:
                stream.stop()
            finally:
                stream.close()
        except sd.PortAudioError as error:
            raise MicrophoneError(f"Could not close audio input: {error}") from error

    def read(self, frame_count: int) -> np.ndarray:
        """Read a blocking chunk of all channels.

        Args:
            frame_count: Number of frames to read.

        Returns:
            int16 array of shape (frame_count, channel_count).

        Raises:
            MicrophoneError: If the backend is not open or PortAudio fails, e.g. after a USB unplug.
        """
        stream = self._stream
        if stream is None:
            raise MicrophoneError("Audio backend is not open.")
        try:
            data, overflowed = stream.read(frame_count)
        except sd.PortAudioError as error:
            raise MicrophoneError(f"Reading audio failed: {error}") from error
        if overflowed:
            # Usually means the consumer is slower than real time, so audio is being lost.
            _logger.warning("Audio input overflow; older samples were dropped.")
        return data

    def _find_device_index(self) -> int:
        for index, device in enumerate(sd.query_devices()):
            if self._config.device_name in device["name"] and device["max_input_channels"] >= self._config.channel_count:
                return index
        raise DeviceNotFoundError(f"No input device '{self._config.device_name}' with {self._config.channel_count} channels found.")


def _start_or_close(stream: sd.InputStream) -> None:
    # A stream that fails to start still holds the device; release it before propagating the error.
    try:
        stream.start()
    except BaseException:
        with suppress(sd.PortAudioError):
            stream.close()
        raise


# --- USB control (shared by tuning and LED ring) -----------------------------


class ReSpeakerUsbDevice:
    """Owns the single USB handle and serializes control transfers from tuning and LED ring."""

    def __init__(self, vendor_id: int = USB_VENDOR_ID, product_id: int = USB_PRODUCT_ID, timeout_ms: int = USB_TIMEOUT_MS) -> None:
        """Store USB ids and create the access lock."""
        self._vendor_id: int = vendor_id
        self._product_id: int = product_id
        self._timeout_ms: int = timeout_ms
        self._lock: threading.Lock = threading.Lock()
        self._device: usb.core.Device | None = None

    def open(self) -> None:
        """Find the device via pyusb; does nothing if it is already open.

        Raises:
            DeviceNotFoundError: If the array is not connected or no libusb backend is installed.
        """
        with self._lock:
            if self._device is None:
                self._device = _find_usb_device(self._vendor_id, self._product_id)

    def close(self) -> None:
        """Release the USB handle; does nothing if it is not open."""
        with self._lock:
            if self._device is not None:
                usb.util.dispose_resources(self._device)
                self._device = None

    def control_read(self, request: int, value: int, index: int, length: int) -> bytes:
        """Perform a locked vendor control IN transfer.

        Args:
            request: Vendor request code.
            value: ``wValue`` field of the setup packet.
            index: ``wIndex`` field of the setup packet.
            length: Number of bytes to read.

        Returns:
            Bytes returned by the device; may be shorter than ``length``.

        Raises:
            MicrophoneError: If the device is not open or the transfer fails.
        """
        with self._locked_device() as device:
            return bytes(device.ctrl_transfer(_REQUEST_TYPE_IN, request, value, index, length, self._timeout_ms))

    def control_write(self, request: int, value: int, index: int, data: bytes) -> None:
        """Perform a locked vendor control OUT transfer.

        Args:
            request: Vendor request code.
            value: ``wValue`` field of the setup packet.
            index: ``wIndex`` field of the setup packet.
            data: Payload to send.

        Raises:
            MicrophoneError: If the device is not open or the transfer fails.
        """
        with self._locked_device() as device:
            device.ctrl_transfer(_REQUEST_TYPE_OUT, request, value, index, data, self._timeout_ms)

    @contextmanager
    def _locked_device(self) -> Iterator[usb.core.Device]:
        with self._lock:
            if self._device is None:
                raise MicrophoneError("ReSpeaker USB device is not open.")
            try:
                yield self._device
            except usb.core.USBError as error:
                raise MicrophoneError(f"USB control transfer failed: {error}") from error


def _find_usb_device(vendor_id: int, product_id: int) -> usb.core.Device:
    try:
        device = usb.core.find(idVendor=vendor_id, idProduct=product_id)
    except usb.core.NoBackendError as error:
        raise DeviceNotFoundError("No libusb backend available.") from error
    if device is None:
        raise DeviceNotFoundError(f"ReSpeaker USB device {vendor_id:04x}:{product_id:04x} not found.")
    return device


@dataclass(frozen=True)
class _TuningParameter:
    parameter_id: int
    offset: int
    is_int: bool
    is_writable: bool


# Values from Seeed's usb_4_mic_array/tuning.py.
_TUNING_PARAMETERS: dict[str, _TuningParameter] = {
    "AECFREEZEONOFF": _TuningParameter(18, 7, is_int=True, is_writable=True),
    "AGCONOFF": _TuningParameter(19, 0, is_int=True, is_writable=True),
    "STATNOISEONOFF": _TuningParameter(19, 8, is_int=True, is_writable=True),
    "NONSTATNOISEONOFF": _TuningParameter(19, 11, is_int=True, is_writable=True),
    "VOICEACTIVITY": _TuningParameter(19, 32, is_int=True, is_writable=False),
    "DOAANGLE": _TuningParameter(21, 0, is_int=True, is_writable=False),
}


class ReSpeakerTuning:
    """Reads and writes XMOS tuning parameters (DOA, VAD, AGC, AEC, noise suppression)."""

    def __init__(self, device: ReSpeakerUsbDevice) -> None:
        """Use the shared USB device."""
        self._device: ReSpeakerUsbDevice = device

    def read_parameter(self, name: str) -> int | float:
        """Read a tuning parameter by name.

        Args:
            name: Parameter name, e.g. ``"DOAANGLE"``.

        Returns:
            The value as ``int`` for integer parameters, otherwise as ``float``.

        Raises:
            MicrophoneError: If the parameter is unknown, the transfer fails or the response is malformed.
        """
        parameter = _lookup_tuning_parameter(name)
        command = _TUNING_READ_FLAG | parameter.offset
        if parameter.is_int:
            command |= _TUNING_INT_FLAG
        response = self._device.control_read(_VENDOR_REQUEST, command, parameter.parameter_id, _TUNING_RESPONSE_LENGTH)
        if len(response) != _TUNING_RESPONSE_LENGTH:
            raise MicrophoneError(f"Tuning parameter '{name}': expected {_TUNING_RESPONSE_LENGTH} response bytes, got {len(response)}.")
        mantissa, exponent = struct.unpack("<ii", response)
        return mantissa if parameter.is_int else mantissa * 2.0**exponent

    def write_parameter(self, name: str, value: int | float) -> None:
        """Write a writable tuning parameter by name.

        Raises:
            MicrophoneError: If the parameter is unknown or read-only.
        """
        parameter = _lookup_tuning_parameter(name)
        if not parameter.is_writable:
            raise MicrophoneError(f"Tuning parameter '{name}' is read-only.")
        if parameter.is_int:
            payload = struct.pack("<iii", parameter.offset, int(value), 1)
        else:
            payload = struct.pack("<ifi", parameter.offset, float(value), 0)
        self._device.control_write(_VENDOR_REQUEST, 0, parameter.parameter_id, payload)

    def get_direction_deg(self) -> int:
        """Return the on-chip direction of arrival (``DOAANGLE``), 0-359 degrees."""
        return int(self.read_parameter("DOAANGLE"))

    def is_voice_active(self) -> bool:
        """Return the on-chip voice activity flag (``VOICEACTIVITY``, read-only)."""
        return bool(self.read_parameter("VOICEACTIVITY"))

    def set_agc_enabled(self, enabled: bool) -> None:
        """Enable or disable automatic gain control (``AGCONOFF``)."""
        self.write_parameter("AGCONOFF", int(enabled))

    def set_echo_cancellation_enabled(self, enabled: bool) -> None:
        """Enable or disable acoustic echo cancellation adaptation (``AECFREEZEONOFF``)."""
        # The chip flag is inverted: 1 freezes adaptation.
        self.write_parameter("AECFREEZEONOFF", int(not enabled))

    def set_noise_suppression_enabled(self, enabled: bool) -> None:
        """Enable or disable stationary and non-stationary noise suppression (``STATNOISEONOFF``, ``NONSTATNOISEONOFF``)."""
        self.write_parameter("STATNOISEONOFF", int(enabled))
        self.write_parameter("NONSTATNOISEONOFF", int(enabled))


def _lookup_tuning_parameter(name: str) -> _TuningParameter:
    try:
        return _TUNING_PARAMETERS[name]
    except KeyError as error:
        raise MicrophoneError(f"Unknown tuning parameter '{name}'.") from error


class LedRing:
    """Controls the 12-LED ring (pixel_ring protocol) for user feedback."""

    def __init__(self, device: ReSpeakerUsbDevice, config: LedConfig | None = None) -> None:
        """Use the shared USB device and the given LED layout (defaults to ``LedConfig()``)."""
        self._device: ReSpeakerUsbDevice = device
        self._config: LedConfig = config if config is not None else LedConfig()

    def wakeup(self, direction_deg: int) -> None:
        """Light the single LED closest to ``direction_deg`` using the configured layout."""
        step = round((direction_deg % 360) * LED_COUNT / 360)
        if not self._config.clockwise:
            step = -step
        led_index = (self._config.zero_deg_led_index + step) % LED_COUNT
        colors = [(0, 0, 0)] * LED_COUNT
        colors[led_index] = self._config.wakeup_color
        self._write(_LED_SHOW, bytes(channel for color in colors for channel in (*color, 0)))

    def listen(self) -> None:
        """Show the listening pattern."""
        self._write(_LED_LISTEN)

    def think(self) -> None:
        """Show the thinking pattern."""
        self._write(_LED_THINK)

    def speak(self) -> None:
        """Show the speaking pattern."""
        self._write(_LED_SPEAK)

    def spin(self) -> None:
        """Show a spinning pattern."""
        self._write(_LED_SPIN)

    def set_color(self, rgb: tuple[int, int, int]) -> None:
        """Set all LEDs to one color (each component 0-255)."""
        self._write(_LED_SET_COLOR, bytes((*rgb, 0)))

    def set_brightness(self, brightness: int) -> None:
        """Set LED brightness (0-31).

        Raises:
            ValueError: If ``brightness`` is out of range.
        """
        if not 0 <= brightness <= MAX_LED_BRIGHTNESS:
            raise ValueError(f"Brightness must be 0-{MAX_LED_BRIGHTNESS}, got {brightness}.")
        self._write(_LED_BRIGHTNESS, bytes([brightness]))

    def off(self) -> None:
        """Turn all LEDs off."""
        self.set_color((0, 0, 0))

    def _write(self, command: int, data: bytes = b"\x00") -> None:
        self._device.control_write(_VENDOR_REQUEST, command, _LED_INDEX, data)


# --- Pluggable speech engines ------------------------------------------------


class SpeechToText(ABC):
    """Pluggable speech-to-text engine."""

    @abstractmethod
    def transcribe(self, audio: np.ndarray, sample_rate_hz: int) -> str:
        """Convert mono audio to text."""


@dataclass(frozen=True)
class Utterance:
    """One finished utterance, delimited by the recognizer's end-of-speech detection.

    Attributes:
        text: Final recognized text; never empty.
        language: Language code selected for this utterance, e.g. ``"en"`` or ``"de"``.
    """

    text: str
    language: str


class StreamingSpeechToText(SpeechToText, ABC):
    """Speech-to-text engine that supports live, utterance-segmented recognition and WAV files.

    A session is split into utterances at detected speech endpoints. Each utterance gets its own
    language decision. Finished utterances are queued until they are drained with
    ``pop_final_utterances`` or returned by ``finish_session``.

    The one-shot calls ``transcribe`` and ``transcribe_wav`` use their own session. If one of them
    raises, it must leave no recognition session active so the engine stays usable. If a live
    session is already active, they raise ``MicrophoneError`` and leave that session untouched.
    """

    @abstractmethod
    def start_session(self, sample_rate_hz: int) -> None:
        """Start a new live recognition session."""

    @abstractmethod
    def accept_audio(self, audio: np.ndarray) -> None:
        """Add a mono int16 audio chunk to the active recognition session."""

    @abstractmethod
    def get_partial_text(self) -> str:
        """Return the provisional text of the utterance currently being spoken."""

    @abstractmethod
    def pop_final_utterances(self) -> list[Utterance]:
        """Remove and return all utterances finished since the last call, oldest first."""

    @abstractmethod
    def finish_session(self) -> str:
        """Finish the session and return the space-joined text of all utterances not yet popped."""

    @abstractmethod
    def cancel_session(self) -> None:
        """Discard the active live recognition session; does nothing if none is active."""

    @abstractmethod
    def get_detected_language(self) -> str | None:
        """Return the language of the latest utterance, or ``None`` if nothing was recognized."""

    @abstractmethod
    def transcribe_wav(self, path: Path) -> str:
        """Transcribe a PCM WAV file and return its text.

        A call that raises must leave no recognition session active, so the engine stays usable.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            MicrophoneError: If the file is not a supported WAV file, or a live session is already active;
                that session is left untouched.
        """


class WakeWordDetector(ABC):
    """Pluggable wake-word engine."""

    @abstractmethod
    def process(self, chunk: np.ndarray) -> bool:
        """Feed a mono audio chunk; return True when the wake word was detected."""

    @abstractmethod
    def reset(self) -> None:
        """Clear internal state after a detection."""


# --- Audio utilities ---------------------------------------------------------


def compute_rms_dbfs(samples: np.ndarray) -> float:
    """Return the RMS level of int16 ``samples`` in dBFS (0 = full scale), floored at ``MIN_DBFS``."""
    if samples.size == 0:
        return MIN_DBFS
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    return _amplitude_to_dbfs(rms)


def compute_peak_dbfs(samples: np.ndarray) -> float:
    """Return the peak level of int16 ``samples`` in dBFS, floored at ``MIN_DBFS``."""
    if samples.size == 0:
        return MIN_DBFS
    peak = float(np.max(np.abs(samples.astype(np.float64))))
    return _amplitude_to_dbfs(peak)


def _amplitude_to_dbfs(amplitude: float) -> float:
    if amplitude <= 0.0:
        return MIN_DBFS
    return max(MIN_DBFS, 20.0 * math.log10(amplitude / FULL_SCALE))


def save_wav(path: Path, audio: np.ndarray, sample_rate_hz: int) -> None:
    """Write 16-bit PCM audio to a WAV file.

    Args:
        path: Destination file.
        audio: int16 samples of shape (frames,) or (frames, channels).
        sample_rate_hz: Sample rate stored in the file header.

    Raises:
        TypeError: If ``audio`` is not int16; other dtypes would be silently truncated.
        ValueError: If ``audio`` is neither 1- nor 2-dimensional.
    """
    if audio.dtype != np.int16:
        raise TypeError(f"Audio must be int16, got {audio.dtype}.")
    if audio.ndim not in (1, 2):
        raise ValueError(f"Audio must have shape (frames,) or (frames, channels), got {audio.shape}.")
    frames = audio if audio.ndim == 2 else audio[:, np.newaxis]
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(frames.shape[1])
        wav_file.setsampwidth(SAMPLE_WIDTH_BYTES)
        wav_file.setframerate(sample_rate_hz)
        wav_file.writeframes(np.ascontiguousarray(frames, dtype="<i2").tobytes())


# --- Recording and streaming -------------------------------------------------


class AudioRecorder:
    """Blocking recordings of fixed length or until silence."""

    def __init__(self, backend: AudioBackend, tuning: ReSpeakerTuning) -> None:
        """Use the given backend for audio and tuning for on-chip VAD."""
        self._backend: AudioBackend = backend
        self._tuning: ReSpeakerTuning = tuning

    def record(self, duration_s: float) -> np.ndarray:
        """Record all channels for ``duration_s`` seconds.

        Raises:
            ValueError: If ``duration_s`` is not positive.
        """
        if duration_s <= 0:
            raise ValueError(f"duration_s must be positive, got {duration_s}.")
        frame_count = round(duration_s * self._backend.config.sample_rate_hz)
        return self._backend.read(frame_count)

    def record_until_silence(self, max_duration_s: float, silence_timeout_s: float) -> np.ndarray:
        """Record all channels until the on-chip VAD reports no voice for ``silence_timeout_s`` or ``max_duration_s`` is reached.

        Silence counts from the start of the recording, so it also ends after ``silence_timeout_s``
        if nobody starts speaking.

        Args:
            max_duration_s: Upper limit of the recording length.
            silence_timeout_s: Continuous silence that ends the recording.

        Returns:
            int16 array of shape (frames, channel_count), a whole number of blocks long.

        Raises:
            ValueError: If a duration is not positive.
            MicrophoneError: If reading audio or the VAD flag fails.
        """
        if max_duration_s <= 0 or silence_timeout_s <= 0:
            raise ValueError(f"Durations must be positive, got max_duration_s={max_duration_s}, silence_timeout_s={silence_timeout_s}.")
        block_size = self._backend.config.block_size_frames
        max_blocks = max(1, self._seconds_to_blocks(max_duration_s))
        silent_block_limit = max(1, self._seconds_to_blocks(silence_timeout_s))
        blocks: list[np.ndarray] = []
        silent_blocks = 0
        while len(blocks) < max_blocks and silent_blocks < silent_block_limit:
            blocks.append(self._backend.read(block_size))
            silent_blocks = 0 if self._tuning.is_voice_active() else silent_blocks + 1
        return np.concatenate(blocks)

    def _seconds_to_blocks(self, duration_s: float) -> int:
        config = self._backend.config
        return math.ceil(duration_s * config.sample_rate_hz / config.block_size_frames)


class AudioStream:
    """Background thread that distributes audio blocks to registered listeners.

    Listeners run on the reader thread, so they must be faster than real time or audio is dropped.
    ``start`` and ``stop`` are meant to be called from one controlling thread (or from a listener).
    """

    def __init__(self, backend: AudioBackend) -> None:
        """Use the given backend as audio source."""
        self._backend: AudioBackend = backend
        self._listeners: list[AudioCallback] = []
        self._listeners_lock: threading.Lock = threading.Lock()
        self._stop_event: threading.Event = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None
        self._error_lock: threading.Lock = threading.Lock()

    def add_listener(self, callback: AudioCallback) -> None:
        """Register a callback that receives each audio block."""
        with self._listeners_lock:
            self._listeners.append(callback)

    def remove_listener(self, callback: AudioCallback) -> None:
        """Unregister a callback; unknown callbacks are ignored."""
        with self._listeners_lock:
            if callback in self._listeners:
                self._listeners.remove(callback)

    def clear_listeners(self) -> None:
        """Unregister all callbacks."""
        with self._listeners_lock:
            self._listeners.clear()

    def start(self) -> None:
        """Start the background capture thread if it is not running and clear any previous ``error``."""
        if self.is_running:
            return
        with self._error_lock:
            self._error = None
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="AudioStream", daemon=True)
        self._thread.start()

    def stop(self, timeout_s: float = 2.0) -> bool:
        """Ask the capture thread to stop and wait up to ``timeout_s`` for it to finish.

        When called from the capture thread itself (i.e. from a listener), the stop is only
        requested; the thread exits after the current block.

        Args:
            timeout_s: Maximum time to wait; bounded so that a hanging device read cannot block forever.

        Returns:
            False if the thread is still running after ``timeout_s``, otherwise True.
        """
        thread = self._thread
        if thread is None:
            return True
        self._stop_event.set()
        if thread is threading.current_thread():
            return True
        thread.join(timeout_s)
        if thread.is_alive():
            _logger.warning("Audio stream thread did not stop within %.1f s; the device read may be hanging.", timeout_s)
            return False
        self._thread = None
        return True

    @property
    def is_running(self) -> bool:
        """Whether the capture thread is running."""
        return self._thread is not None and self._thread.is_alive()

    @property
    def error(self) -> Exception | None:
        """The error that ended the capture thread, or ``None`` if it has not failed since the last ``start``."""
        with self._error_lock:
            return self._error

    def _run(self) -> None:
        try:
            self._read_until_stopped()
        except Exception as error:
            _logger.exception("Audio stream stopped because reading failed.")
            with self._error_lock:
                self._error = error

    def _read_until_stopped(self) -> None:
        block_size = self._backend.config.block_size_frames
        while not self._stop_event.is_set():
            block = self._backend.read(block_size)
            for listener in self._snapshot_listeners():
                _notify_listener(listener, block)

    def _snapshot_listeners(self) -> list[AudioCallback]:
        with self._listeners_lock:
            return list(self._listeners)


def _notify_listener(listener: AudioCallback, block: np.ndarray) -> None:
    # A failing listener must not stop the stream for the other listeners.
    try:
        listener(block)
    except Exception:
        _logger.exception("Audio listener failed.")


def _require[T](component: T | None, description: str) -> T:
    if component is None:
        raise MicrophoneError(f"No {description} configured.")
    return component


def _validate_channels(channels: tuple[int, ...], channel_count: int) -> None:
    if not channels or any(not 0 <= channel < channel_count for channel in channels):
        raise ValueError(f"Channels must be a non-empty selection of 0-{channel_count - 1}, got {channels}.")


def _run_cleanup_step(step: Callable[[], None]) -> Exception | None:
    try:
        step()
    except Exception as error:
        return error
    return None


def _raise_first_error(errors: list[Exception | None]) -> None:
    failures = [error for error in errors if error is not None]
    for error in failures[1:]:
        _logger.error("Additional error while closing the microphone.", exc_info=error)
    if failures:
        raise failures[0]


def _cancel_quietly(engine: StreamingSpeechToText) -> None:
    try:
        engine.cancel_session()
    except Exception:
        _logger.warning("Could not cancel the speech-to-text session.", exc_info=True)


UtteranceCallback = Callable[[Utterance], None]


class _LiveSession:
    """Live transcription state; its ``accept_block`` runs on the reader thread and records the first engine failure."""

    def __init__(self, engine: StreamingSpeechToText, stream: AudioStream, on_utterance: UtteranceCallback | None) -> None:
        self.engine: StreamingSpeechToText = engine
        self._stream: AudioStream = stream
        self._on_utterance: UtteranceCallback | None = on_utterance
        self._error: Exception | None = None
        self._error_lock: threading.Lock = threading.Lock()

    @property
    def error(self) -> Exception | None:
        """The engine error that ended the session, or ``None``."""
        with self._error_lock:
            return self._error

    def accept_block(self, audio: np.ndarray) -> None:
        """Feed one mono block, deliver finished utterances and stop the stream if the engine fails."""
        if self.error is not None:
            return
        try:
            self.engine.accept_audio(audio)
            self._deliver_finished_utterances()
        except Exception as error:
            _logger.error("Live transcription failed; stopping the audio stream.", exc_info=error)
            with self._error_lock:
                self._error = error
            self._stream.stop()

    def _deliver_finished_utterances(self) -> None:
        callback = self._on_utterance
        if callback is None:
            return
        for utterance in self.engine.pop_final_utterances():
            # User code must not end the session.
            try:
                callback(utterance)
            except Exception:
                _logger.exception("Utterance callback failed.")


# --- Facade ------------------------------------------------------------------


class Microphone:
    """High-level microphone API composing capture, tuning, LEDs, STT and wake word.

    Lifecycle operations (``open``, ``close``, ``start_listening``, ``stop_listening``,
    ``start_live_transcription``, ``finish_live_transcription``) are serialized by a reentrant lock.
    Blocking recordings and live-transcription queries do not hold it, so ``close`` can always be
    called from another thread or from a callback.
    """

    def __init__(
        self,
        backend: AudioBackend | None = None,
        speech_to_text: SpeechToText | None = None,
        wake_word_detector: WakeWordDetector | None = None,
        usb_device: ReSpeakerUsbDevice | None = None,
        led_config: LedConfig | None = None,
    ) -> None:
        """Create components; defaults to ``SounddeviceBackend`` and ``ReSpeakerUsbDevice``. STT and wake word are optional."""
        self._backend: AudioBackend = backend if backend is not None else SounddeviceBackend(AudioConfig())
        self._usb_device: ReSpeakerUsbDevice = usb_device if usb_device is not None else ReSpeakerUsbDevice()
        self._tuning: ReSpeakerTuning = ReSpeakerTuning(self._usb_device)
        self._leds: LedRing = LedRing(self._usb_device, led_config)
        self._recorder: AudioRecorder = AudioRecorder(self._backend, self._tuning)
        self._stream: AudioStream = AudioStream(self._backend)
        self._speech_to_text: SpeechToText | None = speech_to_text
        self._wake_word_detector: WakeWordDetector | None = wake_word_detector
        self._lifecycle_lock: threading.RLock = threading.RLock()
        self._is_open: bool = False
        self._live_session: _LiveSession | None = None

    @property
    def is_open(self) -> bool:
        """Whether ``open`` succeeded and ``close`` has not been called since."""
        return self._is_open

    def open(self) -> None:
        """Open the USB device and audio backend; does nothing if already open.

        Raises:
            DeviceNotFoundError: If the USB device or audio input device is missing.
            MicrophoneError: If a device cannot be opened; nothing is left open in that case.
        """
        with self._lifecycle_lock:
            if self._is_open:
                return
            self._usb_device.open()
            try:
                self._backend.open()
            except Exception:
                self._usb_device.close()
                raise
            self._is_open = True

    def close(self) -> None:
        """Cancel live transcription, stop streaming, turn LEDs off and release all devices.

        Every step is attempted even if an earlier one fails; a failure to turn the LEDs off is only
        logged. Calling ``close`` again does nothing.

        It is safe to call from a listener or utterance callback: the capture thread is then only
        asked to stop and exits after the current block. If another thread holds the lifecycle lock
        while waiting for the capture thread, that wait is bounded by ``AudioStream.stop``, so such a
        call is delayed by at most that timeout instead of deadlocking.

        Raises:
            Exception: The first error from stopping the session, closing the backend or releasing the USB device.
        """
        with self._lifecycle_lock:
            if not self._is_open:
                return
            self._is_open = False
            errors = [_run_cleanup_step(self._cancel_live_transcription)]
            self._turn_leds_off()
            errors.append(_run_cleanup_step(self._backend.close))
            errors.append(_run_cleanup_step(self._usb_device.close))
        _raise_first_error(errors)

    def __enter__(self) -> Microphone:
        """Open on entering a ``with`` block."""
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        """Close on leaving a ``with`` block."""
        self.close()

    def record_to_file(self, path: Path, duration_s: float, channels: tuple[int, ...] = (PROCESSED_CHANNEL,)) -> None:
        """Record ``duration_s`` seconds of the selected channels to a WAV file.

        Args:
            path: Destination WAV file.
            duration_s: Recording length.
            channels: Channel indices to save, e.g. ``RAW_MIC_CHANNELS``.

        Raises:
            ValueError: If ``duration_s`` is not positive or ``channels`` is empty or out of range.
            MicrophoneError: If the microphone is not open, is streaming, or reading fails.
        """
        self._ensure_ready_to_record()
        _validate_channels(channels, self._backend.config.channel_count)
        audio = self._recorder.record(duration_s)
        save_wav(path, audio[:, list(channels)], self._backend.config.sample_rate_hz)

    def record_until_silence_to_file(self, path: Path, max_duration_s: float, silence_timeout_s: float = 1.0) -> None:
        """Record the processed channel until silence and save it to a WAV file.

        See ``AudioRecorder.record_until_silence`` for the stop conditions.

        Raises:
            ValueError: If a duration is not positive.
            MicrophoneError: If the microphone is not open, is streaming, or reading fails.
        """
        self._ensure_ready_to_record()
        audio = self._recorder.record_until_silence(max_duration_s, silence_timeout_s)
        save_wav(path, audio[:, PROCESSED_CHANNEL], self._backend.config.sample_rate_hz)

    def transcribe(self, max_duration_s: float, silence_timeout_s: float = 1.0) -> str:
        """Record until silence and return the recognized text.

        Args:
            max_duration_s: Upper limit of the recording length.
            silence_timeout_s: Continuous silence that ends the recording.

        Returns:
            The recognized text.

        Raises:
            ValueError: If a duration is not positive.
            MicrophoneError: If no speech-to-text engine is configured, the microphone is not open or is streaming.
        """
        speech_to_text = _require(self._speech_to_text, "speech-to-text engine")
        self._ensure_ready_to_record()
        audio = self._recorder.record_until_silence(max_duration_s, silence_timeout_s)
        return speech_to_text.transcribe(audio[:, PROCESSED_CHANNEL], self._backend.config.sample_rate_hz)

    def start_live_transcription(
        self,
        english_model_path: Path | None = None,
        german_model_path: Path | None = None,
        on_utterance: UtteranceCallback | None = None,
    ) -> None:
        """Start bilingual live transcription of the processed channel in the background.

        The Vosk engine is created lazily from the model paths unless a streaming engine was configured.

        Args:
            english_model_path: Vosk English model directory; required only if no engine is configured.
            german_model_path: Vosk German model directory; required only if no engine is configured.
            on_utterance: Called with each finished utterance on the audio thread; it must return quickly.
                Its exceptions are logged and ignored. Without it, poll ``pop_live_utterances``.

        Raises:
            MicrophoneError: If the microphone is not open, is already streaming, or the engine cannot be
                created or started. Nothing is left running in that case.
        """
        with self._lifecycle_lock:
            self._ensure_ready_to_record()
            engine = self._get_streaming_speech_to_text(english_model_path, german_model_path)
            self._start_live_session(_LiveSession(engine, self._stream, on_utterance))

    def get_live_transcription_partial(self) -> str:
        """Return the provisional text of the utterance currently being spoken.

        Raises:
            MicrophoneError: If no session is active or it failed, e.g. because the device was unplugged.
        """
        session = self._require_live_session()
        self._raise_if_live_failed(session, require_running=True)
        return session.engine.get_partial_text()

    def pop_live_utterances(self) -> list[Utterance]:
        """Remove and return the utterances finished since the last call, oldest first.

        Utterances already delivered to ``on_utterance`` are not returned again.

        Raises:
            MicrophoneError: If no session is active or it failed.
        """
        session = self._require_live_session()
        self._raise_if_live_failed(session, require_running=True)
        return session.engine.pop_final_utterances()

    def finish_live_transcription(self) -> str:
        """Stop live capture, end the session and return the text of all utterances not yet popped.

        The session is always torn down, also when this raises.

        Raises:
            MicrophoneError: If no session is active or it failed; the transcript would be incomplete.
        """
        with self._lifecycle_lock:
            session = self._require_live_session()
            try:
                text = self._finish_live_session(session)
            except BaseException:
                self._cancel_live_transcription()
                raise
            self._live_session = None
            return text

    def get_live_transcription_language(self) -> str | None:
        """Return the language of the latest live or WAV utterance, e.g. ``"en"`` or ``"de"``.

        Returns:
            The language code, or ``None`` if nothing was recognized or no streaming engine exists.
        """
        engine = self._speech_to_text
        return engine.get_detected_language() if isinstance(engine, StreamingSpeechToText) else None

    def transcribe_wav(
        self,
        path: Path,
        english_model_path: Path | None = None,
        german_model_path: Path | None = None,
    ) -> str:
        """Transcribe an existing PCM WAV file in German or English; the microphone need not be open.

        Args:
            path: WAV file to transcribe.
            english_model_path: Vosk English model directory; required only if no engine is configured.
            german_model_path: Vosk German model directory; required only if no engine is configured.

        Returns:
            The recognized text.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            MicrophoneError: If the engine is busy with a live session, cannot be created or rejects the file.
        """
        self._ensure_not_listening()
        with self._lifecycle_lock:
            engine = self._get_streaming_speech_to_text(english_model_path, german_model_path)
        return engine.transcribe_wav(path)

    def get_direction_deg(self) -> int:
        """Return the current direction of arrival in degrees."""
        return self._tuning.get_direction_deg()

    def get_volume_dbfs(self, duration_s: float = 0.5) -> float:
        """Measure the ambient RMS level of the raw mics over ``duration_s`` seconds.

        Raises:
            MicrophoneError: If the microphone is not open or is streaming.
        """
        self._ensure_ready_to_record()
        # Raw channels avoid the AGC applied to the processed channel.
        audio = self._recorder.record(duration_s)
        return compute_rms_dbfs(audio[:, list(RAW_MIC_CHANNELS)])

    def is_voice_active(self) -> bool:
        """Return whether the on-chip VAD currently detects voice."""
        return self._tuning.is_voice_active()

    def wait_for_wake_word(self, timeout_s: float | None = None) -> bool:
        """Block until the wake word is heard.

        Args:
            timeout_s: Maximum waiting time, or ``None`` to wait forever.

        Returns:
            True if the wake word was detected, False on timeout.

        Raises:
            MicrophoneError: If no wake-word detector is configured, the microphone is not open or is streaming.
        """
        detector = _require(self._wake_word_detector, "wake-word detector")
        self._ensure_ready_to_record()
        detector.reset()
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        block_size = self._backend.config.block_size_frames
        while deadline is None or time.monotonic() < deadline:
            if detector.process(self._backend.read(block_size)[:, PROCESSED_CHANNEL]):
                return True
        return False

    def start_listening(self, callback: AudioCallback) -> None:
        """Stream processed audio blocks to ``callback`` in a background thread.

        Raises:
            MicrophoneError: If the microphone is not open.
        """

        def forward_processed(block: np.ndarray) -> None:
            """Forward only the processed channel to the caller."""
            callback(block[:, PROCESSED_CHANNEL])

        with self._lifecycle_lock:
            self._require_open()
            self._stream.add_listener(forward_processed)
            self._stream.start()

    def stop_listening(self) -> None:
        """Stop the background stream and remove all listeners."""
        with self._lifecycle_lock:
            self._stream.stop()
            self._stream.clear_listeners()

    def configure_processing(self, agc: bool, echo_cancellation: bool, noise_suppression: bool) -> None:
        """Enable or disable the on-chip audio processing features."""
        self._tuning.set_agc_enabled(agc)
        self._tuning.set_echo_cancellation_enabled(echo_cancellation)
        self._tuning.set_noise_suppression_enabled(noise_suppression)

    @property
    def leds(self) -> LedRing:
        """Access the LED ring for feedback patterns."""
        return self._leds

    def _get_streaming_speech_to_text(
        self,
        english_model_path: Path | None,
        german_model_path: Path | None,
    ) -> StreamingSpeechToText:
        """Return or lazily create a streaming speech-to-text engine."""
        speech_to_text = self._speech_to_text
        if speech_to_text is None:
            if english_model_path is None or german_model_path is None:
                raise MicrophoneError("Both English and German Vosk model paths are required.")
            speech_to_text = self._create_vosk_speech_to_text(english_model_path, german_model_path)
            self._speech_to_text = speech_to_text
        if not isinstance(speech_to_text, StreamingSpeechToText):
            raise MicrophoneError("The configured speech-to-text engine does not support streaming or WAV transcription.")
        return speech_to_text

    @staticmethod
    def _create_vosk_speech_to_text(english_model_path: Path, german_model_path: Path) -> StreamingSpeechToText:
        """Import and construct the bilingual Vosk engine, mapping every failure to ``MicrophoneError``."""
        try:
            from control.hardware.vosk_speech_to_text import VoskSpeechToText
        except ImportError as error:
            raise MicrophoneError("Vosk speech recognition is not installed.") from error
        try:
            return VoskSpeechToText(english_model_path, german_model_path)
        except MicrophoneError:
            raise
        except Exception as error:
            raise MicrophoneError(f"Could not create the Vosk speech-to-text engine: {error}") from error

    def _start_live_session(self, session: _LiveSession) -> None:
        """Start the engine session and feed it from the audio stream, rolling back fully on failure."""
        session.engine.start_session(self._backend.config.sample_rate_hz)
        self._live_session = session
        try:
            self.start_listening(session.accept_block)
        except BaseException:
            self._cancel_live_transcription()
            raise

    def _require_live_session(self) -> _LiveSession:
        """Return the active live session or raise an error."""
        session = self._live_session
        if session is None:
            raise MicrophoneError("No live transcription session is active.")
        return session

    def _raise_if_live_failed(self, session: _LiveSession, require_running: bool) -> None:
        """Raise if the engine or the audio stream failed, or the stream ended although it should be running."""
        cause = session.error or self._stream.error
        if cause is not None:
            raise MicrophoneError(f"Live transcription failed: {cause}") from cause
        if require_running and not self._stream.is_running:
            raise MicrophoneError("Live transcription has no audio input; the audio stream stopped unexpectedly.")

    def _finish_live_session(self, session: _LiveSession) -> str:
        """Stop capture and finish the engine session, raising if audio was lost along the way."""
        self._raise_if_live_failed(session, require_running=True)
        self.stop_listening()
        # The last block may have failed while the stream was stopping.
        self._raise_if_live_failed(session, require_running=False)
        return session.engine.finish_session()

    def _cancel_live_transcription(self) -> None:
        """Stop streaming and discard the current live transcription session, if any; engine errors are only logged."""
        session, self._live_session = self._live_session, None
        try:
            self.stop_listening()
        finally:
            if session is not None:
                _cancel_quietly(session.engine)

    def _turn_leds_off(self) -> None:
        """Turn the LEDs off, only logging failures since they must not prevent releasing the devices."""
        try:
            self._leds.off()
        except Exception:
            _logger.warning("Could not turn LEDs off while closing.", exc_info=True)

    def _require_open(self) -> None:
        """Raise when ``open`` has not been called."""
        if not self._is_open:
            raise MicrophoneError("The microphone is not open; call open() first.")

    def _ensure_ready_to_record(self) -> None:
        """Raise unless the microphone is open and its audio is not used by a background stream."""
        self._require_open()
        self._ensure_not_listening()

    def _ensure_not_listening(self) -> None:
        """Raise when a live session or the audio stream is using the engine or the audio input."""
        # The backend has a single reader; parallel reads would split the audio between consumers.
        if self._live_session is not None:
            raise MicrophoneError("A live transcription session is already active.")
        if self._stream.is_running:
            raise MicrophoneError("Stop listening before recording; the audio stream is in use.")
