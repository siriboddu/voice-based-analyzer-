# ==========================================================
# Import Libraries
# ==========================================================

import streamlit as st

from src.speech_to_text import SpeechToText
from src.semantic_analysis import SemanticAnalyzer
from src.audio_features import AudioFeatures
from src.scoring import ScoringEngine
from src.audio_visualizer import AudioVisualizer
from src.pdf_report import PDFReport

from src.ui import (
    show_header,
    show_sidebar,
    show_transcript,
    show_semantic,
    show_audio,
    show_overall,
    show_download,
    show_footer
)

from src.utils import (
    validate_audio,
    delete_temp_file
)

# ==========================================================
# Streamlit Page Configuration
# ==========================================================

st.set_page_config(
    page_title="Voice-Based Concept Understanding Analyser",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# Sidebar
# ==========================================================

show_sidebar()

# ==========================================================
# Header
# ==========================================================

show_header()

# ==========================================================
# Upload Section
# ==========================================================

left, right = st.columns([2, 1])

with left:
    st.subheader("Upload Student Audio")
    audio = st.file_uploader(
        "Choose an audio file",
        type=["wav", "mp3", "m4a"],
        help="Supported formats: WAV, MP3, M4A. Max size: 20MB."
    )

with right:
    st.subheader("Select Topic")
    topic = st.selectbox(
        "Choose the topic being explained",
        options=[
            "Data Structures",
            "Object Oriented Programming",
            "Database Management Systems",
            "Operating Systems",
            "Computer Networks"
        ]
    )

# ==========================================================
# Model Initialization
# ==========================================================

@st.cache_resource
def initialize_engines():
    return (
        SpeechToText(),
        SemanticAnalyzer(),
        AudioFeatures(),
        ScoringEngine(),
        AudioVisualizer()
    )

stt, analyzer, audio_features, scoring, visualizer = initialize_engines()

# ==========================================================
# Main Evaluation Pipeline
# ==========================================================

if audio is not None:
    st.divider()
    st.header("⚡ Processing & AI Analysis Pipeline")

    try:
        # Validate uploaded audio file limits and format metrics
        validate_audio(audio)

        # 1. Speech to Text Conversion
        with st.spinner("🎙 Converting Student Audio Speech to Text..."):
            transcript, audio_path = stt.transcribe_audio(audio)
            show_transcript(transcript)

        # 2. Semantic Alignment Processing
        with st.spinner("🧠 Evaluating Conceptual Semantic Alignment..."):
            semantic_results = analyzer.analyze(topic, transcript)
            show_semantic(semantic_results)

        # 3. Audio Performance Metric Tracking
        with st.spinner("📊 Analysing Voice Waveform and Speech Features..."):
            features = audio_features.analyze(audio_path, transcript)
            show_audio(features)

        # Extract values for calculation parameters
        semantic_score = semantic_results["score"]
        feedback = semantic_results["feedback"]
        confidence = semantic_results["confidence"]
        strengths = semantic_results["strengths"]
        improvements = semantic_results["improvements"]
        recommendation = semantic_results["recommendation"]

        # 4. Aggregated Scoring Processing
        overall_score, grade, recommendation = scoring.calculate(
            semantic_score=semantic_score,
            wpm=features["wpm"],
            pause_ratio=features["pause_ratio"],
            filler_count=features["filler_count"],
            energy=features["energy"]
        )

        show_overall(overall_score, grade, recommendation)

        # 5. PDF Generation Execution 
        with st.spinner("📄 Compiling Professional AI Evaluation PDF Report..."):
            report_generator = PDFReport()
            pdf_path = report_generator.generate(
                transcript=transcript,
                semantic_score=semantic_score,
                feedback=feedback,
                confidence=confidence,
                strengths=strengths,
                improvements=improvements,
                features=features,
                overall_score=overall_score,
                grade=grade,
                recommendation=recommendation,
                audio_path=audio_path
            )
            
        # Display the file download link button layout
        show_download(pdf_path)
    
    # ==========================================================
    # ERROR HANDLING
    # ==========================================================
    except Exception as e:
        st.error("❌ An error occurred during analysis.")
        st.exception(e)

    # ==========================================================
    # CLEANUP
    # ==========================================================
    finally:
        try:
            if "audio_path" in locals():
                delete_temp_file(audio_path)
        except Exception:
            pass

# ==========================================================
# FOOTER
# ==========================================================
show_footer()