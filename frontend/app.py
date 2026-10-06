"""
AI Lecture Notes Generator - Streamlit user interface.

Features:
- Lecture audio/video transcription
- Slide/image OCR
- AI-generated study notes
- TXT/DOCX downloads

Run:
    python -m streamlit run frontend/app.py
"""

from __future__ import annotations

import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Project path
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from dotenv import load_dotenv

from backend import audio as audio_utils
from backend import notes as notes_utils
from backend import ocr as ocr_utils
from backend import transcription as transcription_utils


# ---------------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------------

BASE_DIR = PROJECT_ROOT
AUDIO_DIR = BASE_DIR / "audio"
OUTPUT_DIR = BASE_DIR / "output"

load_dotenv(BASE_DIR / ".env")

AUDIO_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Languages
# ---------------------------------------------------------------------------

LANGUAGES: dict[str, str] = {
    "English (US)": "en-US",
    "English (India)": "en-IN",
    "English (UK)": "en-GB",
    "Hindi": "hi-IN",
    "Spanish": "es-ES",
    "French": "fr-FR",
    "German": "de-DE",
}

IMAGE_TYPES = ["png", "jpg", "jpeg", "bmp", "tiff", "webp"]

DOCX_MIME = (
    "application/vnd.openxmlformats-officedocument."
    "wordprocessingml.document"
)


# ===========================================================================
# SESSION STATE
# ===========================================================================

def init_state() -> None:
    """Create session-state keys used by the app."""
    st.session_state.setdefault("transcript_text", "")
    st.session_state.setdefault("ocr_text", "")
    st.session_state.setdefault("study_notes", None)


# ===========================================================================
# CUSTOM CSS
# ===========================================================================

def inject_css() -> None:
    """Apply application styling."""
    st.markdown(
        """
        <style>
        .stApp {
            background: #f7f9fc;
        }

        .block-container {
            max-width: 1250px;
            padding-top: 1.5rem;
            padding-bottom: 3rem;
        }

        #MainMenu {
            visibility: hidden;
        }

        footer {
            visibility: hidden;
        }

        [data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid #e7ebf2;
        }

        [data-testid="stSidebar"] .block-container {
            padding: 1.4rem 1rem;
        }

        .brand {
            padding: 0.4rem 0.5rem 1.3rem 0.5rem;
        }

        .brand-row {
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .brand-icon {
            width: 44px;
            height: 44px;
            border-radius: 13px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(135deg, #4f7cff, #7c5cff);
            color: white;
            font-size: 22px;
            box-shadow: 0 8px 20px rgba(79,124,255,0.20);
        }

        .brand-title {
            font-size: 1.12rem;
            font-weight: 800;
            color: #17233c;
            line-height: 1.2;
        }

        .brand-subtitle {
            color: #718096;
            font-size: 0.78rem;
            margin-top: 2px;
        }

        [data-testid="stSidebar"] .stRadio > div {
            gap: 5px;
        }

        [data-testid="stSidebar"] .stRadio label {
            padding: 8px 10px;
            border-radius: 10px;
            color: #53627a;
            font-weight: 550;
        }

        .hero {
            position: relative;
            overflow: hidden;
            border-radius: 26px;
            padding: 2.7rem 3rem;
            min-height: 350px;
            background:
                radial-gradient(
                    circle at 88% 20%,
                    rgba(124,92,255,0.18),
                    transparent 27%
                ),
                radial-gradient(
                    circle at 72% 85%,
                    rgba(79,124,255,0.16),
                    transparent 30%
                ),
                linear-gradient(135deg, #edf5ff, #f7f3ff);
            border: 1px solid #e1e9f8;
            margin-bottom: 1.6rem;
            box-shadow: 0 12px 35px rgba(30,50,90,0.05);
        }

        .hero h1 {
            font-size: 2.7rem;
            line-height: 1.08;
            color: #10213f;
            margin: 0;
            max-width: 650px;
            font-weight: 850;
            letter-spacing: -1px;
        }

        .hero p {
            color: #506789;
            font-size: 1.08rem;
            line-height: 1.65;
            max-width: 580px;
            margin-top: 1rem;
        }

        .hero-flow {
            display: flex;
            gap: 12px;
            align-items: center;
            flex-wrap: wrap;
            margin: 1.5rem 0;
            color: #45618c;
            font-size: 0.9rem;
        }

        .flow-item {
            display: flex;
            align-items: center;
            gap: 7px;
        }

        .check {
            width: 21px;
            height: 21px;
            display: inline-flex;
            justify-content: center;
            align-items: center;
            background: #4f7cff;
            color: white;
            border-radius: 50%;
            font-size: 12px;
            font-weight: 700;
        }

        .hero-visual {
            height: 220px;
            border-radius: 24px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(
                135deg,
                rgba(255,255,255,0.55),
                rgba(224,232,255,0.45)
            );
            border: 1px solid rgba(255,255,255,0.8);
            font-size: 78px;
            box-shadow: inset 0 0 35px rgba(255,255,255,0.6);
        }

        .feature-card {
            padding: 1.5rem;
            border-radius: 20px;
            min-height: 175px;
            border: 1px solid #e7ebf3;
            background: white;
            box-shadow: 0 5px 20px rgba(20,40,80,0.04);
        }

        .feature-icon {
            width: 47px;
            height: 47px;
            border-radius: 15px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 22px;
            margin-bottom: 1rem;
        }

        .blue-icon {
            background: #e8f1ff;
        }

        .purple-icon {
            background: #f1eaff;
        }

        .green-icon {
            background: #e4f8f0;
        }

        .feature-title {
            font-size: 1.08rem;
            font-weight: 800;
            color: #17233c;
        }

        .feature-text {
            color: #64748b;
            line-height: 1.5;
            margin-top: 0.4rem;
            font-size: 0.92rem;
        }

        .section-title {
            color: #17233c;
            font-size: 1.55rem;
            font-weight: 800;
            margin-top: 2rem;
            margin-bottom: 0.2rem;
        }

        .section-subtitle {
            color: #718096;
            margin-bottom: 1.3rem;
        }

        .step-card {
            text-align: center;
            padding: 1rem 0.5rem;
        }

        .step-number {
            width: 50px;
            height: 50px;
            margin: auto;
            border-radius: 50%;
            background: #eef3ff;
            color: #4169e1;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            border: 1px solid #dfe8ff;
        }

        .step-title {
            margin-top: 0.7rem;
            font-weight: 750;
            color: #17233c;
        }

        .step-text {
            color: #718096;
            font-size: 0.88rem;
            line-height: 1.45;
        }

        .bottom-banner {
            margin-top: 2rem;
            padding: 1.5rem 2rem;
            border-radius: 20px;
            background: linear-gradient(135deg, #f0f3ff, #f8f5ff);
            border: 1px solid #e4e7f5;
        }

        .bottom-banner-title {
            font-weight: 800;
            color: #17233c;
            font-size: 1.05rem;
        }

        .bottom-banner-text {
            color: #64748b;
            margin-top: 0.3rem;
        }

        .page-heading {
            font-size: 2rem;
            font-weight: 850;
            color: #17233c;
            margin-bottom: 0.2rem;
        }

        .page-description {
            color: #718096;
            margin-bottom: 1.5rem;
        }

        .stButton > button {
            border-radius: 12px;
            font-weight: 650;
            min-height: 44px;
            border: 1px solid #e1e7f0;
        }

        [data-testid="stFileUploader"] {
            border-radius: 16px;
        }

        @media (max-width: 800px) {
            .hero {
                padding: 2rem;
            }

            .hero h1 {
                font-size: 2.1rem;
            }

            .hero p {
                font-size: 1rem;
            }

            .hero-visual {
                margin-top: 1rem;
                height: 160px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ===========================================================================
# SIDEBAR
# ===========================================================================

def render_sidebar() -> tuple[str, str]:
    """Render sidebar and return language and selected page."""
    with st.sidebar:
        st.html(
            """
            <div class="brand">
                <div class="brand-row">
                    <div class="brand-icon">📖</div>
                    <div>
                        <div class="brand-title">AI Lecture Notes</div>
                        <div class="brand-subtitle">Generator</div>
                    </div>
                </div>
            </div>
            """
        )

        page = st.radio(
            "Navigation",
            [
                "🏠 Home",
                "🎙️ Upload & Transcribe",
                "🖼️ Slides OCR",
                "📚 Study Notes",
            ],
            label_visibility="collapsed",
        )

        st.divider()

        st.markdown("### ⚙️ Settings")

        language_label = st.selectbox(
            "Lecture language",
            list(LANGUAGES.keys()),
        )

        st.markdown("### 🛡️ System Check")

        if shutil.which("ffmpeg"):
            st.success("✓ FFmpeg found")
        else:
            st.error("✕ FFmpeg not found")

        if ocr_utils.is_tesseract_available():
            st.success("✓ Tesseract OCR found")
        else:
            st.warning("⚠ Tesseract OCR unavailable")

        if notes_utils.has_api_key():
            st.success("✓ Gemini API key found")
        else:
            st.error("✕ Gemini API key missing")

        st.divider()

        st.html(
            """
            <div style="
                color:#64748b;
                font-size:0.82rem;
                line-height:1.5;
            ">
                💡 <b>Better Notes</b><br>
                ✨ Brighter Future
            </div>
            """
        )

    return LANGUAGES[language_label], page


# ===========================================================================
# HERO
# ===========================================================================

def render_header() -> None:
    """Render home hero."""
    st.html(
        """
        <div class="hero">
            <div>
                <h1>
                    ✨ AI Lecture Notes<br>
                    Generator
                </h1>

                <p>
                    Turn your lectures into study-ready notes in minutes.
                    Upload audio or video, get an accurate transcript,
                    and generate structured notes with AI.
                </p>

                <div class="hero-flow">
                    <div class="flow-item">
                        <span class="check">✓</span>
                        Upload audio/video
                    </div>

                    <span>→</span>

                    <div class="flow-item">
                        <span class="check">✓</span>
                        Get transcript
                    </div>

                    <span>→</span>

                    <div class="flow-item">
                        <span class="check">✓</span>
                        Generate notes
                    </div>
                </div>
            </div>
        </div>
        """
    )


# ===========================================================================
# HOME PAGE
# ===========================================================================

def render_home() -> None:
    """Render home page."""
    st.html(
        """
        <div class="section-title">Start learning smarter</div>
        <div class="section-subtitle">
            Choose what you want to do with your lecture.
        </div>
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.html(
            """
            <div class="feature-card">
                <div class="feature-icon blue-icon">🎙️</div>
                <div class="feature-title">Transcribe</div>
                <div class="feature-text">
                    Convert lecture audio and video into editable text.
                </div>
            </div>
            """
        )

        if st.button(
            "Open Transcriber →",
            key="home_transcribe",
            use_container_width=True,
        ):
            st.session_state["home_navigation"] = "🎙️ Upload & Transcribe"
            st.rerun()

    with col2:
        st.html(
            """
            <div class="feature-card">
                <div class="feature-icon purple-icon">🖼️</div>
                <div class="feature-title">Extract from Slides</div>
                <div class="feature-text">
                    Read important text from lecture screenshots and slides.
                </div>
            </div>
            """
        )

        if st.button(
            "Open Slide OCR →",
            key="home_ocr",
            use_container_width=True,
        ):
            st.session_state["home_navigation"] = "🖼️ Slides OCR"
            st.rerun()

    with col3:
        st.html(
            """
            <div class="feature-card">
                <div class="feature-icon green-icon">📚</div>
                <div class="feature-title">AI Study Notes</div>
                <div class="feature-text">
                    Generate summaries, key points, definitions and exam questions.
                </div>
            </div>
            """
        )

        if st.button(
            "Open Study Notes →",
            key="home_notes",
            use_container_width=True,
        ):
            st.session_state["home_navigation"] = "📚 Study Notes"
            st.rerun()

    st.html(
        """
        <div class="section-title">How it works</div>
        <div class="section-subtitle">
            Get your study material in 4 simple steps.
        </div>
        """
    )

    s1, s2, s3, s4 = st.columns(4)

    steps = [
        ("01", "Upload", "Add your lecture audio or video."),
        ("02", "Transcribe", "Convert your lecture into text."),
        ("03", "Enhance", "Extract text from your slides."),
        ("04", "Generate", "Get structured study notes."),
    ]

    for col, (number, title, description) in zip(
        [s1, s2, s3, s4],
        steps,
    ):
        with col:
            st.html(
                f"""
                <div class="step-card">
                    <div class="step-number">{number}</div>
                    <div class="step-title">{title}</div>
                    <div class="step-text">{description}</div>
                </div>
                """
            )

    st.html(
        """
        <div class="bottom-banner">
            <div class="bottom-banner-title">
                🎓 Built for B.Tech students
            </div>
            <div class="bottom-banner-text">
                Save time on note-taking and focus more on understanding and revision.
            </div>
        </div>
        """
    )


# ===========================================================================
# SAFE FILE NAME
# ===========================================================================

def safe_filename_stem(
    text: str,
    fallback: str = "lecture_notes",
) -> str:
    """Convert free text into a safe filename."""
    cleaned = re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        text.strip(),
    ).strip("_")
    return cleaned[:60] or fallback


# ===========================================================================
# LECTURE PROCESSING
# ===========================================================================

def process_lecture(
    uploaded_file,
    language: str,
) -> None:
    """Run the complete lecture-processing pipeline."""
    temp_files: list[Path] = []
    error_message: str | None = None
    warnings: list[str] = []

    with st.status(
        "Processing your lecture...",
        expanded=True,
    ) as status:
        try:
            st.write("Step 1/5: Checking the file...")

            audio_utils.validate_upload(
                uploaded_file.name,
                uploaded_file.size,
            )
            audio_utils.ensure_ffmpeg_available()

            st.write("Step 2/5: Saving the uploaded file...")

            source_path = audio_utils.save_uploaded_file(
                uploaded_file.name,
                uploaded_file.getvalue(),
                AUDIO_DIR,
            )
            temp_files.append(source_path)

            st.write(
                "Step 3/5: Converting to WAV "
                "(16 kHz, mono) with FFmpeg..."
            )

            wav_path = audio_utils.convert_to_wav(
                source_path,
                AUDIO_DIR / f"{source_path.stem}_16k.wav",
            )
            temp_files.append(wav_path)

            st.write(
                "Step 4/5: Cleaning and normalising the audio..."
            )

            audio_utils.process_audio(wav_path)

            duration = audio_utils.get_duration_seconds(wav_path)

            st.write(
                f"Audio length: "
                f"{audio_utils.format_duration(duration)}"
            )

            st.write(
                "Step 5/5: Transcribing speech "
                "(this can take a while)..."
            )

            progress_bar = st.progress(
                0.0,
                text="Starting transcription...",
            )

            def on_progress(
                fraction: float,
                message: str,
            ) -> None:
                progress_bar.progress(
                    min(max(fraction, 0.0), 1.0),
                    text=message,
                )

            result = transcription_utils.transcribe_wav(
                wav_path,
                language=language,
                progress_callback=on_progress,
            )

            progress_bar.progress(
                1.0,
                text="Transcription finished",
            )

            if not result.text.strip():
                raise transcription_utils.TranscriptionError(
                    "No speech could be recognised. "
                    "The audio may be silent, too noisy, "
                    "or in a different language than the one selected."
                )

            st.session_state["transcript_text"] = result.text
            st.session_state["study_notes"] = None

            if result.unclear_chunks:
                warnings.append(
                    f"{result.unclear_chunks} of "
                    f"{result.total_chunks} audio parts "
                    f"were unclear and were skipped."
                )

            if result.failed_chunks:
                warnings.append(
                    f"{result.failed_chunks} audio parts "
                    f"failed because of network/service errors."
                )

            if result.truncated:
                warnings.append(
                    f"Only the first "
                    f"{transcription_utils.MAX_AUDIO_MINUTES} "
                    f"minutes were transcribed."
                )

            status.update(
                label="Transcription complete",
                state="complete",
                expanded=False,
            )

        except (
            audio_utils.AudioError,
            transcription_utils.TranscriptionError,
        ) as exc:
            error_message = str(exc)
            status.update(
                label="Processing failed",
                state="error",
                expanded=True,
            )

        except Exception as exc:
            error_message = f"Unexpected error: {exc}"
            status.update(
                label="Processing failed",
                state="error",
                expanded=True,
            )

        finally:
            audio_utils.cleanup_files(temp_files)

    if error_message:
        st.error(error_message)
    else:
        st.success(
            "Transcript is ready. "
            "Review it below, then open Study Notes."
        )

        for warning in warnings:
            st.warning(warning)


# ===========================================================================
# LECTURE PAGE
# ===========================================================================

def render_lecture_tab(language: str) -> None:
    """Upload lecture and show transcript."""
    st.html(
        """
        <div class="page-heading">🎙️ Upload &amp; Transcribe</div>
        <div class="page-description">
            Upload your lecture audio or video and convert it into editable text.
        </div>
        """
    )

    extensions = [
        ext.lstrip(".")
        for ext in sorted(audio_utils.SUPPORTED_EXTENSIONS)
    ]

    uploaded = st.file_uploader(
        "Lecture audio or video",
        type=extensions,
        help=(
            f"Supported: {', '.join(e.upper() for e in extensions)}. "
            f"Maximum size: {audio_utils.MAX_UPLOAD_MB} MB."
        ),
    )

    if uploaded is not None:
        st.info(
            f"Selected: {uploaded.name} "
            f"({uploaded.size / (1024 * 1024):.1f} MB)"
        )

    if st.button(
        "🎙️ Transcribe Lecture",
        type="primary",
        disabled=uploaded is None,
        use_container_width=True,
    ):
        process_lecture(uploaded, language)

    st.markdown("### 📄 Transcript")

    st.text_area(
        "Transcript",
        key="transcript_text",
        height=350,
        placeholder=(
            "Your transcript will appear here after transcription."
        ),
        label_visibility="collapsed",
    )

    words = len(st.session_state["transcript_text"].split())
    st.caption(f"📝 {words} words")


# ===========================================================================
# OCR PAGE
# ===========================================================================

def render_ocr_tab() -> None:
    """Extract text from slide screenshots."""
    st.html(
        """
        <div class="page-heading">🖼️ Slides OCR</div>
        <div class="page-description">
            Upload lecture slides, screenshots or whiteboard images and extract their text.
        </div>
        """
    )

    images = st.file_uploader(
        "Lecture images",
        type=IMAGE_TYPES,
        accept_multiple_files=True,
    )

    if st.button(
        "🔍 Extract Text",
        disabled=not images,
        type="primary",
        use_container_width=True,
    ):
        parts: list[str] = []

        for image_file in images:
            try:
                with st.spinner(f"Reading {image_file.name}..."):
                    text = ocr_utils.extract_text_from_image(
                        image_file.getvalue()
                    )

                if text:
                    parts.append(
                        f"--- {image_file.name} ---\n{text}"
                    )
                else:
                    st.warning(
                        f"No text found in {image_file.name}."
                    )

            except ocr_utils.OcrError as exc:
                st.error(f"{image_file.name}: {exc}")

        if parts:
            st.session_state["ocr_text"] = "\n\n".join(parts)
            st.success("Text extracted successfully.")

    st.markdown("### 📝 Extracted Text")

    st.text_area(
        "Extracted text",
        key="ocr_text",
        height=350,
        placeholder=(
            "Text extracted from your images will appear here."
        ),
        label_visibility="collapsed",
    )


# ===========================================================================
# SAVE NOTES
# ===========================================================================

def save_copy_to_output(
    text: str,
    title: str,
) -> None:
    """Save a TXT copy of generated notes."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    file_path = (
        OUTPUT_DIR
        / f"{safe_filename_stem(title)}_{stamp}.txt"
    )

    try:
        file_path.write_text(
            text,
            encoding="utf-8",
        )
    except OSError:
        pass


# ===========================================================================
# NOTES CONTENT
# ===========================================================================

def render_notes_content(study_notes: dict) -> None:
    """Display generated study notes."""
    st.markdown(f"# 📚 {study_notes['title']}")

    st.markdown("## 📝 Summary")
    st.write(
        study_notes["summary"]
        or "No summary available."
    )

    st.markdown("## 📖 Detailed Notes")

    if study_notes["detailed_notes"]:
        for section in study_notes["detailed_notes"]:
            st.markdown(f"### {section['heading']}")

            for point in section["points"]:
                st.markdown(f"- {point}")
    else:
        st.write("No detailed notes available.")

    st.markdown("## 🔑 Key Points")

    for point in (
        study_notes["key_points"]
        or ["None found."]
    ):
        st.markdown(f"- {point}")

    st.markdown("## 📘 Important Definitions")

    if study_notes["definitions"]:
        for item in study_notes["definitions"]:
            st.markdown(
                f"- **{item['term']}**: "
                f"{item['definition']}"
            )
    else:
        st.write("None found.")

    st.markdown("## 💡 Examples")

    for example in (
        study_notes["examples"]
        or ["None found."]
    ):
        st.markdown(f"- {example}")

    st.markdown("## 🎯 Possible Exam Questions")

    if study_notes["exam_questions"]:
        for number, item in enumerate(
            study_notes["exam_questions"],
            start=1,
        ):
            st.markdown(
                f"### Q{number}. "
                f"{item['question']}"
            )

            if item["answer"]:
                st.markdown(
                    f"**Answer:** {item['answer']}"
                )
    else:
        st.write("None found.")

    st.markdown("## ⚡ Quick Revision Points")

    for point in (
        study_notes["revision_points"]
        or ["None found."]
    ):
        st.markdown(f"- {point}")


# ===========================================================================
# DOWNLOADS
# ===========================================================================

def render_downloads(study_notes: dict) -> None:
    """Show TXT and DOCX download buttons."""
    st.markdown("## ⬇️ Download Notes")

    stem = safe_filename_stem(study_notes["title"])

    col_txt, col_docx = st.columns(2)

    with col_txt:
        st.download_button(
            "📄 Download as TXT",
            data=(
                notes_utils
                .notes_to_text(study_notes)
                .encode("utf-8")
            ),
            file_name=f"{stem}.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with col_docx:
        try:
            st.download_button(
                "📝 Download as DOCX",
                data=notes_utils.notes_to_docx_bytes(study_notes),
                file_name=f"{stem}.docx",
                mime=DOCX_MIME,
                use_container_width=True,
            )
        except notes_utils.NotesError as exc:
            st.warning(
                f"DOCX export is not available: {exc}"
            )


# ===========================================================================
# STUDY NOTES PAGE
# ===========================================================================

def render_notes_tab() -> None:
    """Generate and display AI study notes."""
    st.html(
        """
        <div class="page-heading">📚 AI Study Notes</div>
        <div class="page-description">
            Turn your transcript and slide text into structured study material.
        </div>
        """
    )

    st.markdown("### 📄 Transcript")

    transcript = st.text_area(
        "Transcript",
        value=st.session_state["transcript_text"],
        height=300,
        placeholder="Paste your transcript here...",
        key="study_notes_transcript",
        label_visibility="collapsed",
    )

    st.session_state["transcript_text"] = transcript

    ocr_text = st.session_state["ocr_text"].strip()

    if transcript:
        st.success(
            f"✓ Transcript ready — {len(transcript.split())} words"
        )
    else:
        st.info(
            "No transcript yet. Go to Upload & Transcribe first."
        )

    include_ocr = st.checkbox(
        "Also use text extracted from images",
        value=True,
        disabled=not ocr_text,
        help=(
            "Enabled when you have extracted text in the Slides OCR section."
        ),
    )

    if st.button(
        "✨ Generate Study Notes",
        type="primary",
        disabled=not transcript,
        use_container_width=True,
    ):
        try:
            with st.spinner(
                "Gemini is creating your study notes..."
            ):
                result = notes_utils.generate_notes(
                    transcript,
                    (
                        ocr_text
                        if include_ocr and ocr_text
                        else ""
                    ),
                )

            st.session_state["study_notes"] = result

            save_copy_to_output(
                notes_utils.notes_to_text(result),
                result["title"],
            )

            st.success(
                "🎉 Study notes generated successfully!"
            )

        except notes_utils.NotesError as exc:
            st.error(str(exc))

    study_notes = st.session_state["study_notes"]

    if study_notes:
        st.divider()
        render_notes_content(study_notes)
        st.divider()
        render_downloads(study_notes)


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    """Application entry point."""
    st.set_page_config(
        page_title="AI Lecture Notes Generator",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_state()
    inject_css()

    language, page = render_sidebar()

    if page == "🏠 Home":
        render_header()
        render_home()

    elif page == "🎙️ Upload & Transcribe":
        render_lecture_tab(language)

    elif page == "🖼️ Slides OCR":
        render_ocr_tab()

    elif page == "📚 Study Notes":
        render_notes_tab()


if __name__ == "__main__":
    main()
