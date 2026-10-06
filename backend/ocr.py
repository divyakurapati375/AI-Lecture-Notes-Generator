"""Text extraction from images using pytesseract and the installed Tesseract OCR."""

from __future__ import annotations

import io
import os

import pytesseract
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_IMAGE_MB: int = 10
MIN_WIDTH_FOR_NO_UPSCALE: int = 1000


class OcrError(Exception):
    """Raised when OCR fails or the image is invalid."""


def _configure_tesseract() -> None:
    """Use TESSERACT_CMD from the environment if Tesseract is not in PATH (optional)."""
    custom_path = os.getenv("TESSERACT_CMD", "").strip()
    if custom_path:
        pytesseract.pytesseract.tesseract_cmd = custom_path


def is_tesseract_available() -> bool:
    """Return True if Tesseract can be started."""
    _configure_tesseract()
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def _prepare_image(image: Image.Image) -> Image.Image:
    """Convert to grayscale, boost contrast and enlarge small images for better OCR."""
    prepared = ImageOps.exif_transpose(image)
    prepared = ImageOps.grayscale(prepared)
    prepared = ImageOps.autocontrast(prepared)
    if prepared.width < MIN_WIDTH_FOR_NO_UPSCALE:
        prepared = prepared.resize(
            (prepared.width * 2, prepared.height * 2), Image.Resampling.LANCZOS
        )
    return prepared


def extract_text_from_image(image_bytes: bytes, language: str = "eng") -> str:
    """Extract text from image bytes.

    Args:
        image_bytes: Raw bytes of PNG/JPG/BMP/TIFF/WEBP image.
        language: Tesseract language code (default English).

    Returns:
        The recognised text (may be an empty string if nothing was found).

    Raises:
        OcrError: for invalid images, large files or Tesseract problems.
    """
    if not image_bytes:
        raise OcrError("The image file is empty.")
    if len(image_bytes) > MAX_IMAGE_MB * 1024 * 1024:
        raise OcrError(f"The image is larger than {MAX_IMAGE_MB} MB. Please upload a smaller image.")

    _configure_tesseract()

    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.load()
            prepared = _prepare_image(image)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise OcrError("This file is not a valid image or is corrupted.") from exc

    try:
        text = pytesseract.image_to_string(prepared, lang=language)
    except pytesseract.TesseractNotFoundError as exc:
        raise OcrError(
            "Tesseract OCR was not found. Install it and add it to PATH "
            "(or set TESSERACT_CMD in the .env file)."
        ) from exc
    except pytesseract.TesseractError as exc:
        raise OcrError(f"Tesseract failed to read the image: {exc}") from exc
    except RuntimeError as exc:
        raise OcrError(f"OCR timed out or failed: {exc}") from exc

    return text.strip()