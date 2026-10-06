"""Study-note generation with Google Gemini (modern ``google-genai`` SDK)
plus TXT and DOCX export helpers."""

from __future__ import annotations

import io
import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

# Change the model here (or set GEMINI_MODEL in .env) if you want another one.
GEMINI_MODEL = "gemini-3.5-flash-lite"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAX_INPUT_CHARS: int = 300_000  # very long transcripts are shortened to this size

SYSTEM_INSTRUCTION = (
    "You are an expert teaching assistant who turns lecture transcripts into clear, accurate "
    "study notes for engineering (B.Tech) students. Use only information found in the provided "
    "material and do not invent facts. The transcript was created automatically and may contain "
    "recognition mistakes; fix obvious ones using context. Respond with one valid JSON object only."
)

JSON_SCHEMA_TEXT = """Return a JSON object with exactly these keys:
{
  "title": "short lecture title",
  "summary": "3-5 sentence summary",
  "detailed_notes": [
    {"heading": "section heading", "points": ["point", "point"]}
  ],
  "key_points": ["important point"],
  "definitions": [
    {"term": "term", "definition": "clear definition"}
  ],
  "examples": ["example mentioned or implied in the lecture"],
  "exam_questions": [
    {"question": "possible exam question", "answer": "short model answer"}
  ],
  "revision_points": ["short last-minute revision point"]
}
If the material has nothing for a section, use an empty list. Output JSON only, with no markdown fences."""


class NotesError(Exception):
    """Raised when notes cannot be generated or exported."""


# --------------------------------------------------------------------------- #
# API key and client
# --------------------------------------------------------------------------- #
def _read_api_key() -> str:
    """Read GEMINI_API_KEY from the .env file / environment (never hardcoded)."""
    load_dotenv(PROJECT_ROOT / ".env")
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if key == "YOUR_API_KEY_HERE":
        return ""
    return key


def has_api_key() -> bool:
    """Return True if a usable API key is configured."""
    return bool(_read_api_key())


def get_api_key() -> str:
    """Return the API key or raise a helpful error."""
    key = _read_api_key()
    if not key:
        raise NotesError(
            "Gemini API key is missing. Open the .env file and set GEMINI_API_KEY=your_real_key, "
            "then restart the app."
        )
    return key


# --------------------------------------------------------------------------- #
# Response parsing
# --------------------------------------------------------------------------- #
def _parse_json(raw_text: str) -> dict[str, Any]:
    """Parse the model's JSON, tolerating accidental code fences or extra text."""
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise NotesError("Gemini returned an unreadable response. Please try again.") from None
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            raise NotesError("Gemini returned an unreadable response. Please try again.") from None
    if not isinstance(data, dict):
        raise NotesError("Gemini returned an unexpected format. Please try again.")
    return data


def _as_str(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _as_str_list(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [_as_str(item) for item in value if _as_str(item)]


def _normalize(data: dict[str, Any]) -> dict[str, Any]:
    """Make sure every expected key exists with the expected type."""
    sections: list[dict[str, Any]] = []
    for item in data.get("detailed_notes", []) or []:
        if isinstance(item, dict):
            heading = _as_str(item.get("heading")) or "Notes"
            points = _as_str_list(item.get("points"))
        else:
            heading, points = "Notes", _as_str_list(item)
        if points:
            sections.append({"heading": heading, "points": points})

    definitions: list[dict[str, str]] = []
    for item in data.get("definitions", []) or []:
        if isinstance(item, dict) and _as_str(item.get("term")):
            definitions.append(
                {"term": _as_str(item.get("term")), "definition": _as_str(item.get("definition"))}
            )

    questions: list[dict[str, str]] = []
    for item in data.get("exam_questions", []) or []:
        if isinstance(item, dict) and _as_str(item.get("question")):
            questions.append(
                {"question": _as_str(item.get("question")), "answer": _as_str(item.get("answer"))}
            )
        elif isinstance(item, str) and item.strip():
            questions.append({"question": item.strip(), "answer": ""})

    return {
        "title": _as_str(data.get("title")) or "Lecture Notes",
        "summary": _as_str(data.get("summary")),
        "detailed_notes": sections,
        "key_points": _as_str_list(data.get("key_points")),
        "definitions": definitions,
        "examples": _as_str_list(data.get("examples")),
        "exam_questions": questions,
        "revision_points": _as_str_list(data.get("revision_points")),
    }


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def _build_prompt(transcript: str, extra_text: str) -> str:
    parts = [JSON_SCHEMA_TEXT, "\n=== LECTURE TRANSCRIPT ===\n", transcript[:MAX_INPUT_CHARS]]
    if extra_text.strip():
        parts.append("\n\n=== TEXT EXTRACTED FROM SLIDES/IMAGES (OCR, may contain errors) ===\n")
        parts.append(extra_text[:50_000])
    return "".join(parts)


def _friendly_api_error(exc: errors.APIError) -> str:
    code = getattr(exc, "code", None)
    detail = getattr(exc, "message", None) or str(exc)
    if code in (400, 401, 403) and "api key" in detail.lower():
        return "Your Gemini API key is invalid. Please check GEMINI_API_KEY in the .env file."
    if code in (401, 403):
        return "Gemini rejected the request (permission denied). Check your API key and its permissions."
    if code == 404:
        return f"The Gemini model '{GEMINI_MODEL}' was not found. Change GEMINI_MODEL in utils/notes.py."
    if code == 429:
        return "Gemini rate limit or quota reached. Please wait a minute and try again."
    if code is not None and code >= 500:
        return f"Gemini server error {code}: {detail}"
    return f"Gemini API error: {detail}"


def generate_notes(transcript: str, extra_text: str = "") -> dict[str, Any]:
    """Generate structured study notes from a transcript and optional OCR text."""

    if not transcript or not transcript.strip():
        raise NotesError(
            "The transcript is empty, so there is nothing to summarise."
        )

    api_key = get_api_key()

    model_name = (
        os.getenv("GEMINI_MODEL", "").strip()
        or GEMINI_MODEL
    )

    try:
        client = genai.Client(api_key=api_key)

        max_retries = 3
        response = None

        for attempt in range(max_retries):

            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=_build_prompt(
                        transcript,
                        extra_text,
                    ),
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        temperature=0.3,
                        response_mime_type="application/json",
                    ),
                )

                break

            except errors.APIError as exc:

                code = getattr(exc, "code", None)

                if code == 503 and attempt < max_retries - 1:
                    time.sleep(5)
                    continue

                raise

    except errors.APIError as exc:

        raise NotesError(
            _friendly_api_error(exc)
        ) from exc

    except Exception as exc:

        raise NotesError(
            f"Could not contact Gemini. "
            f"Check your internet connection. "
            f"Details: {exc}"
        ) from exc

    raw_text = getattr(response, "text", None)

    if not raw_text:
        raise NotesError(
            "Gemini returned no text. "
            "Please try again or edit the transcript."
        )

    return _normalize(
        _parse_json(raw_text)
    )
# --------------------------------------------------------------------------- #
# Export helpers
# --------------------------------------------------------------------------- #
def notes_to_text(notes: dict[str, Any]) -> str:
    """Convert the notes dictionary into plain text."""
    lines: list[str] = [notes["title"].upper(), "=" * max(len(notes["title"]), 10), ""]

    lines += ["SUMMARY", "-------", notes["summary"] or "N/A", ""]

    lines += ["DETAILED NOTES", "--------------"]
    for section in notes["detailed_notes"]:
        lines.append(section["heading"])
        lines += [f"  - {point}" for point in section["points"]]
        lines.append("")

    lines += ["KEY POINTS", "----------"]
    lines += [f"- {point}" for point in notes["key_points"]] or ["N/A"]
    lines.append("")

    lines += ["IMPORTANT DEFINITIONS", "---------------------"]
    lines += [f"- {d['term']}: {d['definition']}" for d in notes["definitions"]] or ["N/A"]
    lines.append("")

    lines += ["EXAMPLES", "--------"]
    lines += [f"- {example}" for example in notes["examples"]] or ["N/A"]
    lines.append("")

    lines += ["POSSIBLE EXAM QUESTIONS", "-----------------------"]
    if notes["exam_questions"]:
        for number, item in enumerate(notes["exam_questions"], start=1):
            lines.append(f"Q{number}. {item['question']}")
            if item["answer"]:
                lines.append(f"    Answer: {item['answer']}")
    else:
        lines.append("N/A")
    lines.append("")

    lines += ["QUICK REVISION POINTS", "---------------------"]
    lines += [f"- {point}" for point in notes["revision_points"]] or ["N/A"]
    lines.append("")

    return "\n".join(lines)


def notes_to_docx_bytes(notes: dict[str, Any]) -> bytes:
    """Convert the notes dictionary into a DOCX file (returned as bytes)."""
    try:
        from docx import Document  # python-docx
    except ImportError as exc:
        raise NotesError("python-docx is not installed.") from exc

    document = Document()
    document.add_heading(notes["title"], level=0)

    document.add_heading("Summary", level=1)
    document.add_paragraph(notes["summary"] or "N/A")

    document.add_heading("Detailed Notes", level=1)
    for section in notes["detailed_notes"]:
        document.add_heading(section["heading"], level=2)
        for point in section["points"]:
            document.add_paragraph(point, style="List Bullet")

    document.add_heading("Key Points", level=1)
    for point in notes["key_points"] or ["N/A"]:
        document.add_paragraph(point, style="List Bullet")

    document.add_heading("Important Definitions", level=1)
    if notes["definitions"]:
        for item in notes["definitions"]:
            paragraph = document.add_paragraph(style="List Bullet")
            paragraph.add_run(f"{item['term']}: ").bold = True
            paragraph.add_run(item["definition"])
    else:
        document.add_paragraph("N/A")

    document.add_heading("Examples", level=1)
    for example in notes["examples"] or ["N/A"]:
        document.add_paragraph(example, style="List Bullet")

    document.add_heading("Possible Exam Questions", level=1)
    if notes["exam_questions"]:
        for number, item in enumerate(notes["exam_questions"], start=1):
            paragraph = document.add_paragraph()
            paragraph.add_run(f"Q{number}. {item['question']}").bold = True
            if item["answer"]:
                document.add_paragraph(f"Answer: {item['answer']}")
    else:
        document.add_paragraph("N/A")

    document.add_heading("Quick Revision Points", level=1)
    for point in notes["revision_points"] or ["N/A"]:
        document.add_paragraph(point, style="List Bullet")

    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()