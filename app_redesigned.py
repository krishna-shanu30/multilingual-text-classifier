from pathlib import Path
import sys
import json
from datetime import datetime, timezone

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from language_intelligence.engine import load_artifacts, predict_text
from language_intelligence.interpretation import interpret_text

st.set_page_config(
    page_title="Language Intelligence Studio",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

LANGUAGES = {
    "Arabic": "ar",
    "Bulgarian": "bg",
    "German": "de",
    "Greek": "el",
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "Hindi": "hi",
    "Italian": "it",
    "Japanese": "ja",
    "Dutch": "nl",
    "Polish": "pl",
    "Portuguese": "pt",
    "Russian": "ru",
    "Swahili": "sw",
    "Thai": "th",
    "Turkish": "tr",
    "Urdu": "ur",
    "Vietnamese": "vi",
    "Chinese (Simplified)": "zh",
}

@st.cache_resource
def get_model():
    return load_artifacts(ROOT, use_candidate=True)

try:
    model, vectorizer, encoder = get_model()
except Exception as exc:
    st.error(f"Could not load the language detection model: {exc}")
    st.stop()

if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "Light"

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

with st.sidebar:
    st.markdown("## 🌐 Language Studio")
    st.caption("Multilingual text detection and interpretation")

    st.selectbox(
        "Appearance",
        ["Light", "Dark"],
        key="theme_mode",
    )

    st.divider()
    st.markdown("**Supported languages**")
    st.caption("20 languages")
    st.markdown("English · Portuguese · Spanish · French · German · Hindi · Arabic · Chinese · Japanese · and more")

dark = st.session_state.theme_mode == "Dark"

if dark:
    bg = "#101827"
    panel = "#182235"
    fg = "#F1F5F9"
    muted = "#A8B4C7"
    border = "#344258"
    accent = "#8B9CFF"
    input_bg = "#111B2C"
else:
    bg = "#F5F7FB"
    panel = "#FFFFFF"
    fg = "#172033"
    muted = "#64748B"
    border = "#DFE5EF"
    accent = "#4058D6"
    input_bg = "#FFFFFF"

st.markdown(
    f"""
    <style>
    .stApp {{
        background: {bg};
        color: {fg};
    }}
    [data-testid="stSidebar"] {{
        background: {panel};
        border-right: 1px solid {border};
    }}
    [data-testid="stHeader"] {{
        background: transparent;
    }}
    h1, h2, h3, p, label, .stMarkdown {{
        color: {fg};
    }}
    .hero {{
        background: {panel};
        border: 1px solid {border};
        border-radius: 20px;
        padding: 28px;
        margin-bottom: 20px;
    }}
    .hero-label {{
        color: {accent};
        font-weight: 700;
        letter-spacing: 1px;
        font-size: 12px;
        text-transform: uppercase;
    }}
    .muted {{
        color: {muted};
    }}
    .result-card {{
        background: {panel};
        border: 1px solid {border};
        border-radius: 16px;
        padding: 20px;
        min-height: 110px;
    }}
    .result-value {{
        font-size: 24px;
        font-weight: 700;
        color: {accent};
        overflow-wrap: anywhere;
    }}
    div[data-testid="stTextArea"] textarea {{
        background: {input_bg};
        color: {fg};
        border: 1px solid {border};
        border-radius: 12px;
    }}
    div[data-testid="stSelectbox"] > div > div {{
        border-radius: 10px;
    }}
    div.stButton > button {{
        border-radius: 10px;
        min-height: 44px;
        font-weight: 600;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div class="hero-label">AI-powered multilingual workspace</div>
      <h1 style="margin:8px 0 4px 0;">Understand any text.</h1>
      <p class="muted" style="margin:0;">
        Detect the source language, choose your output language,
        and generate an interpretation in one place.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([1.25, 0.75], gap="large")

with left:
    st.subheader("Your text")
    st.caption("Paste a sentence, paragraph, message, or article.")

    example = st.selectbox(
        "Quick example",
        [
            "Choose an example",
            "English",
            "Portuguese",
            "Spanish",
            "French",
            "Hindi",
            "Swahili",
            "Japanese",
        ],
    )

    examples = {
        "English": "Every morning, I learn something new because curiosity helps me understand the world.",
        "Portuguese": "Todas as manhãs, gosto de aprender algo novo porque a curiosidade me ajuda a compreender o mundo.",
        "Spanish": "Cada mañana me gusta aprender algo nuevo porque la curiosidad me ayuda a comprender el mundo.",
        "French": "Chaque matin, j'aime apprendre quelque chose de nouveau, car la curiosité m'aide à comprendre le monde.",
        "Hindi": "हर सुबह मुझे कुछ नया सीखना पसंद है क्योंकि जिज्ञासा मुझे दुनिया को बेहतर समझने में मदद करती है।",
        "Swahili": "Kila asubuhi ninapenda kujifunza jambo jipya kwa sababu udadisi hunisaidia kuelewa ulimwengu.",
        "Japanese": "毎朝、新しいことを学ぶのが好きです。好奇心は世界をより深く理解する助けになります。",
    }

    if example in examples:
        if st.button("Use this example", use_container_width=True):
            st.session_state.source_text = examples[example]

    source_text = st.text_area(
        "Input text",
        key="source_text",
        height=230,
        placeholder="Type or paste your text here...",
        help="For more reliable language identification, enter a complete sentence or paragraph.",
    )

    st.caption(f"{len(source_text)} characters")

with right:
    st.subheader("Output settings")
    st.caption("Choose the language for your result.")

    output_language = st.selectbox(
        "Result language",
        list(LANGUAGES.keys()),
        index=list(LANGUAGES.keys()).index("English"),
    )

    st.info(
        "The app detects the source language automatically. "
        "Your selected language is used for the interpretation request."
    )

    st.markdown("**What you'll get**")
    st.markdown(
        "- Detected language\n"
        "- Prediction status\n"
        "- Interpretation in your chosen language\n"
        "- Downloadable report"
    )

    analyze = st.button(
        "Analyze and interpret text",
        type="primary",
        use_container_width=True,
    )

if analyze:
    if not source_text.strip():
        st.warning("Please enter some text first.")
    else:
        try:
            with st.spinner("Detecting language..."):
                result = predict_text(
                    source_text,
                    model,
                    vectorizer,
                    encoder,
                    top_k=5,
                )

            with st.spinner(f"Preparing result in {output_language}..."):
                interpretation = interpret_text(
                    text=result["text"],
                    detected_language=result["language"],
                    output_language=output_language,
                )

            report = {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "input_text": result["text"],
                "detected_language": result["language"],
                "detected_language_code": result["language_code"],
                "model_score": result["confidence"],
                "score_type": result.get("score_type", "Model probability"),
                "status": result["status"],
                "output_language": output_language,
                "interpretation": interpretation,
                "candidates": result["candidates"],
            }

            st.session_state.analysis_result = report

        except Exception as exc:
            st.error(f"Analysis failed: {exc}")

report = st.session_state.analysis_result

if report:
    st.divider()
    st.markdown("## Your results")

    a, b, c = st.columns(3)

    with a:
        st.markdown(
            f'<div class="result-card"><div class="muted">Detected language</div>'
            f'<div class="result-value">{report["detected_language"]}</div>'
            f'<div class="muted">{report["detected_language_code"]}</div></div>',
            unsafe_allow_html=True,
        )

    with b:
        st.markdown(
            f'<div class="result-card"><div class="muted">Result language</div>'
            f'<div class="result-value">{report["output_language"]}</div>'
            f'<div class="muted">Selected by you</div></div>',
            unsafe_allow_html=True,
        )

    with c:
        st.markdown(
            f'<div class="result-card"><div class="muted">Text length</div>'
            f'<div class="result-value">{len(report["input_text"]):,}</div>'
            f'<div class="muted">Characters</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("### Detection status")
    if (
        "Short input" in report["status"]
        or "Ambiguous" in report["status"]
        or "cautiously" in report["status"]
    ):
        st.warning(report["status"])
    else:
        st.success(report["status"])

    if report["score_type"] == "SVM decision score (not probability)":
        st.caption(
            "The classifier uses an SVM decision score, not a probability. "
            "Do not interpret this number as a confidence percentage."
        )

    st.markdown("### Interpretation / translated result")
    interpretation = report["interpretation"]

    if isinstance(interpretation, str):
        st.text_area(
            "Result text",
            value=interpretation,
            height=240,
            key="display_interpretation",
        )
        downloadable_text = interpretation
    else:
        st.json(interpretation)
        downloadable_text = json.dumps(
            interpretation,
            ensure_ascii=False,
            indent=2,
        )

    with st.expander("View original input"):
        st.write(report["input_text"])

    with st.expander("View alternative language candidates"):
        for item in report["candidates"]:
            if report["score_type"] == "SVM decision score (not probability)":
                score = f'{item["confidence"]:.3f} (decision score)'
            else:
                score = f'{item["confidence"]:.1%}'
            st.write(f'{item["language"]} ({item["language_code"]}) — {score}')

    st.markdown("### Download your result")
    d1, d2 = st.columns(2)

    with d1:
        st.download_button(
            "Download result as TXT",
            data=(
                f'Detected language: {report["detected_language"]}\n'
                f'Output language: {report["output_language"]}\n'
                f'Status: {report["status"]}\n\n'
                f'Original text:\n{report["input_text"]}\n\n'
                f'Result:\n{downloadable_text}'
            ),
            file_name="language-analysis-result.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with d2:
        st.download_button(
            "Download full report as JSON",
            data=json.dumps(report, ensure_ascii=False, indent=2, default=str),
            file_name="language-analysis-report.json",
            mime="application/json",
            use_container_width=True,
        )

    if st.button("Clear result", use_container_width=True):
        st.session_state.analysis_result = None
        st.rerun()

st.divider()
st.caption(
    "Language Intelligence Studio · Linear SVM · Local-first interface. "
    "Predictions can be uncertain for short, mixed-language, or ambiguous text."
)
