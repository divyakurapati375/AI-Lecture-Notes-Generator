"""Speech-to-text using the SpeechRecognition package (Google Web Speech API).

Important limitations (handled in this module, not hidden):
* The free Google Web Speech backend only accepts short audio clips, so the
  lecture is split into 30-second chunks that are sent one at a time.
* It needs an internet connection and may fail or be rate-limited. Each chunk
  is retried a few times; repeated failures stop the process with a clear error.
* Very long recordings are limited to ``MAX_AUDIO_MINUTES`` minutes.
* Unclear speech is skipped and counted, so you can see how much was missed.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf
import speech_recognition as sr

CHUNK_SECONDS: int = 30
MAX_AUDIO_MINUTES: int = 120
MAX_RETRIES: int = 3
MAX_CONSECUTIVE_FAILURES: int = 3
SILENCE_RMS_THRESHOLD: float = 30.0  # on the int16 scale (max 32767)
CHUNKS_PER_PARAGRAPH: int = 4

ProgressCallback = Callable[[float, str], None]


class TranscriptionError(Exception):
    """Raised when transcription cannot continue."""


class _RecognitionRequestError(Exception):
    """Internal: the speech service could not be reached for one chunk."""


@dataclass
class TranscriptionResult:
    """Transcript text plus statistics about the process."""

    text: str
    total_chunks: int
    transcribed_chunks: int
    silent_chunks: int
    unclear_chunks: int
    failed_chunks: int
    truncated: bool


def _recognize_with_retry(recognizer: sr.Recognizer, audio_data: sr.AudioData, language: str) -> str:
    """Recognise one chunk. Returns '' if the speech is unclear."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return str(recognizer.recognize_google(audio_data, language=language)).strip()
        except sr.UnknownValueError:
            return ""  # speech not understood
        except sr.RequestError:
            if attempt < MAX_RETRIES:
                time.sleep(1.5 * attempt)
    raise _RecognitionRequestError()


def transcribe_wav(
    wav_path: Path,
    language: str = "en-US",
    chunk_seconds: int = CHUNK_SECONDS,
    max_minutes: int = MAX_AUDIO_MINUTES,
    progress_callback: ProgressCallback | None = None,
) -> TranscriptionResult:
    """Transcribe a WAV file chunk by chunk.

    Args:
        wav_path: 16-bit PCM WAV file (mono recommended).
        language: Language code such as ``en-US``.
        chunk_seconds: Length of each chunk sent to the speech service.
        max_minutes: Maximum amount of audio to transcribe.
        progress_callback: Optional function ``(fraction, message)``.

    Raises:
        TranscriptionError: for unreadable files or repeated service failures.
    """
    path = Path(wav_path)
    if not path.exists():
        raise TranscriptionError("The audio file to transcribe was not found.")

    try:
        info = sf.info(str(path))
    except RuntimeError as exc:
        raise TranscriptionError(f"Could not read the audio file: {exc}") from exc

    sample_rate = int(info.samplerate)
    max_frames = int(max_minutes * 60 * sample_rate)
    truncated = info.frames > max_frames
    frames_to_read = min(int(info.frames), max_frames)
    chunk_frames = int(chunk_seconds * sample_rate)

    if frames_to_read <= 0:
        raise TranscriptionError("The audio file contains no audio data.")

    total_chunks = math.ceil(frames_to_read / chunk_frames)
    recognizer = sr.Recognizer()

    pieces: list[str] = []
    silent = unclear = failed = transcribed = 0
    consecutive_failures = 0

    try:
        with sf.SoundFile(str(path)) as wav_file:
            for index in range(total_chunks):
                wanted = min(chunk_frames, frames_to_read - index * chunk_frames)
                samples = wav_file.read(frames=wanted, dtype="int16")
                if samples.size == 0:
                    break
                if samples.ndim > 1:
                    samples = samples[:, 0]

                if progress_callback:
                    progress_callback(
                        index / total_chunks,
                        f"Transcribing part {index + 1} of {total_chunks}...",
                    )

                rms = float(np.sqrt(np.mean(np.square(samples.astype(np.float64)))))
                if rms < SILENCE_RMS_THRESHOLD:
                    silent += 1
                    continue

                audio_data = sr.AudioData(samples.tobytes(), sample_rate, 2)

                try:
                    text = _recognize_with_retry(recognizer, audio_data, language)
                except _RecognitionRequestError:
                    failed += 1
                    consecutive_failures += 1
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        raise TranscriptionError(
                            "The speech recognition service could not be reached several times in a row. "
                            "Check your internet connection and try again later."
                        ) from None
                    continue
                except OSError as exc:
                    raise TranscriptionError(
                        f"Audio encoding for speech recognition failed (FLAC tool problem): {exc}"
                    ) from exc

                consecutive_failures = 0
                if text:
                    pieces.append(text)
                    transcribed += 1
                else:
                    unclear += 1

                time.sleep(0.2)  # be polite to the free service
    except sf.LibsndfileError as exc:
        raise TranscriptionError(f"Error while reading the audio: {exc}") from exc

    paragraphs = [
        " ".join(pieces[i : i + CHUNKS_PER_PARAGRAPH])
        for i in range(0, len(pieces), CHUNKS_PER_PARAGRAPH)
    ]

    return TranscriptionResult(
        text="\n\n".join(paragraphs),
        total_chunks=total_chunks,
        transcribed_chunks=transcribed,
        silent_chunks=silent,
        unclear_chunks=unclear,
        failed_chunks=failed,
        truncated=truncated,
    )