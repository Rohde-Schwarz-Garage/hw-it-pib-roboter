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
from contextlib import contextmanager
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
            stream.start()
        except sd.PortAudioError as error:
            raise MicrophoneError(f"Could not open audio input: {error}") from error
        self._stream = stream

    def close(self) -> None:
        """Stop and close the input stream."""
        if self._stream is None:
            return
        self._stream.stop()
        self._stream.close()
        self._stream = None

    def read(self, frame_count: int) -> np.ndarray:
        """Read a blocking chunk of all channels."""
        if self._stream is None:
            raise MicrophoneError("Audio backend is not open.")
        data, overflowed = self._stream.read(frame_count)
        if overflowed:
            _logger.debug("Audio input overflow; older samples were dropped.")
        return data

    def _find_device_index(self) -> int:
        for index, device in enumerate(sd.query_devices()):
            if self._config.device_name in device["name"] and device["max_input_channels"] >= self._config.channel_count:
                return index
        raise DeviceNotFoundError(f"No input device '{self._config.device_name}' with {self._config.channel_count} channels found.")


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
        """Find the device via pyusb.

        Raises:
            DeviceNotFoundError: If the array is not connected or no libusb backend is installed.
        """
        try:
            device = usb.core.find(idVendor=self._vendor_id, idProduct=self._product_id)
        except usb.core.NoBackendError as error:
            raise DeviceNotFoundError("No libusb backend available.") from error
        if device is None:
            raise DeviceNotFoundError(f"ReSpeaker USB device {self._vendor_id:04x}:{self._product_id:04x} not found.")
        self._device = device

    def close(self) -> None:
        """Release the USB handle."""
        with self._lock:
            if self._device is not None:
                usb.util.dispose_resources(self._device)
                self._device = None

    def control_read(self, request: int, value: int, index: int, length: int) -> bytes:
        """Perform a locked vendor control IN transfer."""
        with self._locked_device() as device:
            return bytes(device.ctrl_transfer(_REQUEST_TYPE_IN, request, value, index, length, self._timeout_ms))

    def control_write(self, request: int, value: int, index: int, data: bytes) -> None:
        """Perform a locked vendor control OUT transfer."""
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
        """Read a tuning parameter by name, e.g. ``"DOAANGLE"``."""
        parameter = _lookup_tuning_parameter(name)
        command = _TUNING_READ_FLAG | parameter.offset
        if parameter.is_int:
            command |= _TUNING_INT_FLAG
        response = self._device.control_read(_VENDOR_REQUEST, command, parameter.parameter_id, _TUNING_RESPONSE_LENGTH)
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


class StreamingSpeechToText(SpeechToText, ABC):
    """Speech-to-text engine that supports live recognition and WAV files."""

    @abstractmethod
    def start_session(self, sample_rate_hz: int) -> None:
        """Start a new live recognition session."""

    @abstractmethod
    def accept_audio(self, audio: np.ndarray) -> None:
        """Add a mono audio chunk to the active recognition session."""

    @abstractmethod
    def get_partial_text(self) -> str:
        """Return the currently recognized live text."""

    @abstractmethod
    def finish_session(self) -> str:
        """Finish the live recognition session and return its final text."""

    @abstractmethod
    def cancel_session(self) -> None:
        """Discard the active live recognition session."""

    @abstractmethod
    def get_detected_language(self) -> str | None:
        """Return the selected language code, if audio has been recognized."""

    @abstractmethod
    def transcribe_wav(self, path: Path) -> str:
        """Transcribe a WAV file and return its text."""


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
    """Write int16 ``audio`` of shape (frames,) or (frames, channels) to a WAV file."""
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
        """Record until the on-chip VAD reports no voice for ``silence_timeout_s`` or ``max_duration_s`` is reached."""
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
    """Background thread that distributes audio blocks to registered listeners."""

    def __init__(self, backend: AudioBackend) -> None:
        """Use the given backend as audio source."""
        self._backend: AudioBackend = backend
        self._listeners: list[AudioCallback] = []
        self._listeners_lock: threading.Lock = threading.Lock()
        self._stop_event: threading.Event = threading.Event()
        self._thread: threading.Thread | None = None

    def add_listener(self, callback: AudioCallback) -> None:
        """Register a callback that receives each audio block."""
        with self._listeners_lock:
            self._listeners.append(callback)

    def remove_listener(self, callback: AudioCallback) -> None:
        """Unregister a callback; unknown callbacks are ignored."""
        with self._listeners_lock:
            if callback in self._listeners:
                self._listeners.remove(callback)

    def start(self) -> None:
        """Start the background capture thread if it is not running."""
        if self.is_running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="AudioStream", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the background capture thread and wait for it to finish."""
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join()
        self._thread = None

    @property
    def is_running(self) -> bool:
        """Whether the capture thread is running."""
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        block_size = self._backend.config.block_size_frames
        try:
            while not self._stop_event.is_set():
                block = self._backend.read(block_size)
                for listener in self._snapshot_listeners():
                    _notify_listener(listener, block)
        except MicrophoneError:
            _logger.exception("Audio stream stopped because reading failed.")

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


# --- Facade ------------------------------------------------------------------


class Microphone:
    """High-level microphone API composing capture, tuning, LEDs, STT and wake word."""

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
        self._listeners: list[AudioCallback] = []
        self._live_transcription_engine: StreamingSpeechToText | None = None
        self._last_transcription_language: str | None = None

    def open(self) -> None:
        """Open the USB device and audio backend."""
        self._usb_device.open()
        try:
            self._backend.open()
        except Exception:
            self._usb_device.close()
            raise

    def close(self) -> None:
        """Stop streaming, turn LEDs off and release all devices."""
        self._cancel_live_transcription()
        try:
            self._leds.off()
        except MicrophoneError:
            _logger.warning("Could not turn LEDs off while closing.", exc_info=True)
        self._backend.close()
        self._usb_device.close()

    def __enter__(self) -> Microphone:
        """Open on entering a ``with`` block."""
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        """Close on leaving a ``with`` block."""
        self.close()

    def record_to_file(self, path: Path, duration_s: float, channels: tuple[int, ...] = (PROCESSED_CHANNEL,)) -> None:
        """Record ``duration_s`` seconds of the selected channels to a WAV file."""
        self._ensure_not_listening()
        audio = self._recorder.record(duration_s)
        save_wav(path, audio[:, list(channels)], self._backend.config.sample_rate_hz)

    def record_until_silence_to_file(self, path: Path, max_duration_s: float, silence_timeout_s: float = 1.0) -> None:
        """Record the processed channel until silence and save it to a WAV file."""
        self._ensure_not_listening()
        audio = self._recorder.record_until_silence(max_duration_s, silence_timeout_s)
        save_wav(path, audio[:, PROCESSED_CHANNEL], self._backend.config.sample_rate_hz)

    def transcribe(self, max_duration_s: float, silence_timeout_s: float = 1.0) -> str:
        """Record until silence and return the recognized text.

        Raises:
            MicrophoneError: If no speech-to-text engine is configured.
        """
        speech_to_text = _require(self._speech_to_text, "speech-to-text engine")
        self._ensure_not_listening()
        audio = self._recorder.record_until_silence(max_duration_s, silence_timeout_s)
        return speech_to_text.transcribe(audio[:, PROCESSED_CHANNEL], self._backend.config.sample_rate_hz)

    def start_live_transcription(
        self,
        english_model_path: Path | None = None,
        german_model_path: Path | None = None,
    ) -> None:
        """Start bilingual live transcription and lazily create the Vosk engine."""
        if self._live_transcription_engine is not None:
            raise MicrophoneError("A live transcription session is already active.")
        self._ensure_not_listening()
        engine = self._get_streaming_speech_to_text(english_model_path, german_model_path)
        engine.start_session(self._backend.config.sample_rate_hz)
        try:
            self.start_listening(engine.accept_audio)
        except Exception:
            engine.cancel_session()
            raise
        self._live_transcription_engine = engine

    def get_live_transcription_partial(self) -> str:
        """Return the provisional text of the active live transcription."""
        return self._require_live_transcription().get_partial_text()

    def finish_live_transcription(self) -> str:
        """Stop live capture and return the final bilingual transcript."""
        engine = self._require_live_transcription()
        self.stop_listening()
        try:
            text = engine.finish_session()
            self._last_transcription_language = engine.get_detected_language()
            return text
        finally:
            self._live_transcription_engine = None

    def get_live_transcription_language(self) -> str | None:
        """Return ``en`` or ``de`` for the latest live transcript."""
        if self._live_transcription_engine is None:
            return self._last_transcription_language
        return self._live_transcription_engine.get_detected_language()

    def transcribe_wav(
        self,
        path: Path,
        english_model_path: Path | None = None,
        german_model_path: Path | None = None,
    ) -> str:
        """Transcribe an existing mono PCM WAV file in German or English."""
        self._ensure_not_listening()
        engine = self._get_streaming_speech_to_text(english_model_path, german_model_path)
        text = engine.transcribe_wav(path)
        self._last_transcription_language = engine.get_detected_language()
        return text

    def get_direction_deg(self) -> int:
        """Return the current direction of arrival in degrees."""
        return self._tuning.get_direction_deg()

    def get_volume_dbfs(self, duration_s: float = 0.5) -> float:
        """Measure the ambient RMS level of the raw mics over ``duration_s`` seconds."""
        self._ensure_not_listening()
        # Raw channels avoid the AGC applied to the processed channel.
        audio = self._recorder.record(duration_s)
        return compute_rms_dbfs(audio[:, list(RAW_MIC_CHANNELS)])

    def is_voice_active(self) -> bool:
        """Return whether the on-chip VAD currently detects voice."""
        return self._tuning.is_voice_active()

    def wait_for_wake_word(self, timeout_s: float | None = None) -> bool:
        """Block until the wake word is heard; return False on timeout.

        Raises:
            MicrophoneError: If no wake-word detector is configured.
        """
        detector = _require(self._wake_word_detector, "wake-word detector")
        self._ensure_not_listening()
        detector.reset()
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        block_size = self._backend.config.block_size_frames
        while deadline is None or time.monotonic() < deadline:
            if detector.process(self._backend.read(block_size)[:, PROCESSED_CHANNEL]):
                return True
        return False

    def start_listening(self, callback: AudioCallback) -> None:
        """Stream processed audio blocks to ``callback`` in a background thread."""

        def forward_processed(block: np.ndarray) -> None:
            """Forward only the processed channel to the caller."""
            callback(block[:, PROCESSED_CHANNEL])

        self._listeners.append(forward_processed)
        self._stream.add_listener(forward_processed)
        self._stream.start()

    def stop_listening(self) -> None:
        """Stop the background stream and remove all listeners."""
        self._stream.stop()
        for listener in self._listeners:
            self._stream.remove_listener(listener)
        self._listeners.clear()

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
        """Import and construct the bilingual Vosk engine."""
        try:
            from .vosk_speech_to_text import VoskSpeechToText
        except ImportError as error:
            raise MicrophoneError("Vosk speech recognition is not installed.") from error
        return VoskSpeechToText(english_model_path, german_model_path)

    def _require_live_transcription(self) -> StreamingSpeechToText:
        """Return the active live engine or raise an error."""
        if self._live_transcription_engine is None:
            raise MicrophoneError("No live transcription session is active.")
        return self._live_transcription_engine

    def _cancel_live_transcription(self) -> None:
        """Stop and discard the current live transcription session."""
        engine = self._live_transcription_engine
        self.stop_listening()
        if engine is not None:
            engine.cancel_session()
            self._live_transcription_engine = None

    def _ensure_not_listening(self) -> None:
        """Raise when the audio stream is already in use."""
        # The backend has a single reader; parallel reads would split the audio between consumers.
        if self._stream.is_running:
            raise MicrophoneError("Stop listening before recording; the audio stream is in use.")
