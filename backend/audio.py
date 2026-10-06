"""Audio helpers: upload validation, FFmpeg conversion and audio clean-up."""

from __future__ import annotations

import re
import shutil
import subprocess
import uuid
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

SUPPORTED_EXTENSIONS: set[str] = {".mp3", ".wav", ".m4a", ".mp4", ".webm"}
MAX_UPLOAD_MB: int = 1000
TARGET_SAMPLE_RATE: int = 160000
FFMPEG_TIMEOUT_SECONDS: int = 50000  # 30 minutes


class AudioError(Exception):
    """Raised when an audio-related step fails (validation, FFmpeg, processing)."""


def validate_upload(filename: str, size_bytes: int) -> None:
    """Check extension and size of an uploaded file.

    Raises:
        AudioError: if the file type is unsupported, empty or too large.
    """
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(e.lstrip(".").upper() for e in SUPPORTED_EXTENSIONS))
        raise AudioError(f"Unsupported file type '{extension or 'unknown'}'. Please upload: {allowed}.")
    if size_bytes <= 0:
        raise AudioError("The uploaded file is empty.")
    if size_bytes > MAX_UPLOAD_MB * 1024 * 1024:
        raise AudioError(
            f"The file is too large ({size_bytes / (1024 * 1024):.0f} MB). "
            f"Maximum allowed size is {MAX_UPLOAD_MB} MB. Try compressing or trimming it."
        )


def ensure_ffmpeg_available() -> str:
    """Return the FFmpeg executable path or raise an error if it is missing."""
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise AudioError(
            "FFmpeg was not found. Install FFmpeg and make sure it is available in your PATH."
        )
    return ffmpeg_path


def save_uploaded_file(filename: str, data: bytes, destination_dir: Path) -> Path:
    """Save uploaded bytes into ``destination_dir`` using a safe, unique file name."""
    destination_dir.mkdir(parents=True, exist_ok=True)
    original = Path(filename)
    safe_stem = re.sub(r"[^A-Za-z0-9_-]+", "_", original.stem).strip("_") or "lecture"
    unique_name = f"{uuid.uuid4().hex[:8]}_{safe_stem}{original.suffix.lower()}"
    target = destination_dir / unique_name
    try:
        target.write_bytes(data)
    except OSError as exc:
        raise AudioError(f"Could not save the uploaded file: {exc}") from exc
    return target


def convert_to_wav(input_path: Path, output_path: Path) -> Path:
    """Convert any supported media file to 16 kHz mono 16-bit WAV using FFmpeg."""
    ffmpeg_path = ensure_ffmpeg_available()

    if input_path.resolve() == output_path.resolve():
        raise AudioError("Input and output file must be different.")

    command = [
        ffmpeg_path,
        "-y",
        "-loglevel", "error",
        "-i", str(input_path),
        "-vn",                      # ignore video stream
        "-ac", "1",                 # mono
        "-ar", str(TARGET_SAMPLE_RATE),
        "-c:a", "pcm_s16le",        # 16-bit PCM WAV
        str(output_path),
    ]

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=FFMPEG_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise AudioError("FFmpeg took too long to convert the file. Try a shorter recording.") from exc
    except OSError as exc:
        raise AudioError(f"Could not run FFmpeg: {exc}") from exc

    if completed.returncode != 0:
        details = " ".join(completed.stderr.strip().splitlines()[-3:]) or "unknown error"
        raise AudioError(
            f"FFmpeg could not convert this file. It may be corrupted or have no audio track. Details: {details}"
        )

    if not output_path.exists() or output_path.stat().st_size <= 44:
        raise AudioError("FFmpeg produced an empty audio file. The input may not contain any audio.")

    return output_path


def process_audio(wav_path: Path, trim_silence: bool = True) -> Path:
    """Normalise volume and trim leading/trailing silence (in place).

    Raises:
        AudioError: if the audio cannot be read or is completely silent.
    """
    try:
        samples, _ = librosa.load(str(wav_path), sr=TARGET_SAMPLE_RATE, mono=True)
    except MemoryError as exc:
        raise AudioError("Not enough memory to process this recording. Try a shorter file.") from exc
    except Exception as exc:
        raise AudioError(f"Could not read the converted audio: {exc}") from exc

    if samples.size == 0 or float(np.max(np.abs(samples))) < 1e-4:
        raise AudioError("The recording appears to be silent or empty.")

    samples = librosa.util.normalize(samples)

    if trim_silence:
        trimmed, _ = librosa.effects.trim(samples, top_db=40)
        if trimmed.size > 0:
            samples = trimmed

    try:
        sf.write(str(wav_path), samples, TARGET_SAMPLE_RATE, subtype="PCM_16")
    except Exception as exc:
        raise AudioError(f"Could not write the processed audio: {exc}") from exc
    return wav_path


def get_duration_seconds(wav_path: Path) -> float:
    """Return the duration of an audio file in seconds."""
    try:
        return float(sf.info(str(wav_path)).duration)
    except RuntimeError as exc:
        raise AudioError(f"Could not read audio information: {exc}") from exc


def format_duration(seconds: float) -> str:
    """Format seconds as H:MM:SS."""
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


def cleanup_files(paths: list[Path]) -> None:
    """Delete temporary files, ignoring errors."""
    for path in paths:
        try:
            Path(path).unlink(missing_ok=True)
        except OSError:
            pass