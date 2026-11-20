"""Piper text-to-speech engine."""

from __future__ import annotations

import subprocess
import sys
import wave
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event

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


def _piper_model_matches(path: Path, language: str, model: str | None) -> bool:
    """Match a standard Piper voice name against the requested language and model."""
    locale = path.stem.split("-", 1)[0].lower()
    requested = language.replace("-", "_").lower()
    language_matches = locale == requested or ("_" not in requested and locale.startswith(requested + "_"))
    return language_matches and (model is None or model in {path.stem, path.name})


def _select_piper_model(directory: str | Path | None, language: str, model: str | None) -> Path:
    """Select exactly one complete Piper voice from a directory."""
    if directory is None:
        raise TtsError("Piper requires a model directory.")
    try:
        folder = Path(directory).resolve(strict=True)
        if not folder.is_dir():
            raise TtsError("Piper model path must be a directory.")
        return _find_piper_model(folder, language, model)
    except (OSError, RuntimeError, TypeError, ValueError) as err:
        raise TtsError("Cannot read Piper model directory.") from err


def _find_piper_model(folder: Path, language: str, model: str | None) -> Path:
    """Find one matching model and verify its companion configuration exists."""
    if model is not None and (not isinstance(model, str) or not model):
        raise TtsError("Invalid Piper model name.")
    matches = [path for path in folder.glob("*.onnx") if path.is_file() and _piper_model_matches(path, language, model)]
    if len(matches) != 1:
        raise TtsError("Piper requires exactly one matching model; specify a model name if needed.")
    if not Path(str(matches[0]) + ".json").is_file():
        raise TtsError("Piper model configuration file is missing.")
    return matches[0]


def _run_piper(model: Path, text: str, output_directory: str, cancel: Event | None) -> AudioClip:
    """Run Piper in a cancellable process and return validated WAV audio."""
    output_path = Path(output_directory) / "speech.wav"
    input_path = Path(output_directory) / "input.txt"
    input_path.write_text(text, encoding="utf-8")
    command = [sys.executable, "-m", "piper", "--model", str(model), "--output-file", str(output_path), "--input-file", str(input_path)]
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_for_process(process, cancel)
    finally:
        terminate_process(process)
    rate, frames = read_tts_output(str(output_path))
    return encode_pcm(1, rate, frames)


class PiperEngine:
    """Synthesize speech using a local Piper voice."""

    def __init__(self, language: str, model_dir: str | Path | None, model: str | None = None) -> None:
        """Select a Piper voice for the requested language."""
        self._language: str = normalize_tts_language(language)
        self._model: Path = _select_piper_model(model_dir, self._language, model)

    @property
    def source(self) -> str:
        """Return the Piper source name."""
        return "piper"

    @property
    def language(self) -> str:
        """Return the selected language."""
        return self._language

    def synthesize_cancellable(self, text: str, cancel: Event | None) -> AudioClip:
        """Synthesize a WAV file while honoring cancellation and the deadline."""
        raise_if_cancelled(cancel)
        try:
            with TemporaryDirectory() as output_directory:
                return _run_piper(self._model, text, output_directory, cancel)
        except (PlaybackCancelledError, TtsError):
            raise
        except (OSError, RuntimeError, TypeError, ValueError, wave.Error) as err:
            raise TtsError("Speech engine failed to synthesize audio.") from err
