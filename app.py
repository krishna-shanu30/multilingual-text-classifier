from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from language_intelligence.agents import run_local_llm_explanation, run_quality_agent, run_report_agent, run_script_agent
from language_intelligence.engine import explain_prediction, load_artifacts, predict_text
from language_intelligence.interpretation import interpret_text

st.set_page_config(page_title="Language Intelligence Studio", page_icon="🌐", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --ink:#e9eef8; --muted:#9caac0; --line:rgba(148,163,184,.18); --panel:rgba(17,26,44,.78); --accent:#8b9dff; }
html, body, [class*="css"] { font-family: 'Manrope', sans-serif; }
.stApp { background: radial-gradient(ellipse at 15% 0%, #1b2a4a 0, #0b1020 44%, #080d18 100%); color:var(--ink); }
.block-container { padding-top: 1.8rem; max-width: 1440px; }
[data-testid="stSidebar"] { background:rgba(7,12,24,.88); border-right:1px solid var(--line); }
.hero { padding: 1.55rem 1.7rem; border:1px solid var(--line); border-radius:22px; background:linear-gradient(125deg,rgba(38,54,91,.88),rgba(16,25,44,.72)); margin-bottom:1.25rem; }
.eyebrow { color:#a5b4fc; letter-spacing:.16em; text-transform:uppercase; font-size:.72rem; font-weight:800; }
.hero h1 { margin:.35rem 0 .45rem; font-size:clamp(2rem,4vw,3.1rem); line-height:1.12; letter-spacing:-.055em; }
.hero p { color:#b9c5d9; margin:0; max-width:780px; }
.metric-card { background:var(--panel); border:1px solid var(--line); border-radius:16px; padding:1rem 1.1rem; min-height:105px; }
.metric-label { color:var(--muted); font-size:.78rem; text-transform:uppercase; letter-spacing:.08em; }
.metric-value { color:#f8fafc; font-size:1.65rem; font-weight:800; margin-top:.3rem; }
.panel { background:var(--panel); border:1px solid var(--line); border-radius:18px; padding:1.2rem; }
.small-muted { color:var(--muted); font-size:.88rem; }
div.stButton > button { border-radius:11px; border:1px solid rgba(165,180,252,.4); font-weight:700; }
[data-testid="stMetric"] { background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:14px; }
hr { border-color:var(--line); }
code { font-family:'DM Mono',monospace; }
</style>
<div class="hero">
  <div class="eyebrow">NLP · Model Intelligence · Local-first AI</div>
  <h1>Language Intelligence Studio</h1>
  <p>Explore multilingual text, inspect model evidence, profile writing systems, and run transparent analysis on a lightweight 20-language classifier.</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading baseline model artifacts…")
def get_artifacts():
    return load_artifacts(ROOT, use_candidate=True)

try:
    model, vectorizer, encoder = get_artifacts()
except Exception as exc:
    st.error(f"Could not load the original model artifacts: {exc}")
    st.stop()

with st.sidebar:
    st.markdown("## 🌐 L.I.S.")
    st.caption("Language Intelligence Studio · v2.0")
    page = st.radio("WORKSPACE", ["Analyze text", "Batch lab", "Model insights", "AI agent report", "About & setup"], label_visibility="collapsed")
    st.divider()
    st.markdown("**Active model**")
    if vectorizer is None and encoder is None:
        pipeline_features = model.named_steps["features"]
        feature_count = sum(
            len(branch.get_feature_names_out())
            for _, branch in pipeline_features.transformer_list
        )
        st.caption(
            f"{len(model.classes_)} language labels · "
            f"{feature_count:,} combined character and word features · Linear SVM"
        )
    else:
        st.caption("Character n-gram TF-IDF + Logistic Regression")
        st.caption(
            f"{len(encoder.classes_)} language labels · "
            f"{len(vectorizer.vocabulary_):,} learned features"
        )

if "history" not in st.session_state:
    st.session_state.history = []

if page == "Analyze text":
    left, right = st.columns([1.08, .92], gap="large")
    with left:
        st.markdown("### Analyze a sample")
        st.markdown('<p class="small-muted">Paste a sentence or paragraph. Longer, natural samples generally provide more evidence.</p>', unsafe_allow_html=True)
        sample = st.text_area("Text input", placeholder="Type or paste multilingual text here…", height=220, label_visibility="collapsed", key="single_text")
        output_language = st.selectbox(
            "Translate and explain in",
            [
                "English", "Hindi", "Gujarati", "French", "Spanish",
                "German", "Japanese", "Arabic", "Chinese", "Portuguese",
                "Russian",
            ],
            key="output_language",
        )
        examples = {
            "English": "Machine learning helps computers discover patterns in data.",
            "Hindi": "मशीन लर्निंग डेटा में पैटर्न खोजने में मदद करती है।",
            "French": "Le traitement automatique des langues transforme notre façon d'analyser les textes.",
            "Japanese": "人工知能はデータから有用なパターンを見つけることができます。",
            "Arabic": "يساعد التعلم الآلي الحواسيب على اكتشاف الأنماط في البيانات.",
        }
        chosen = st.selectbox("Try a sample", ["Choose an example…", *examples.keys()])
        if chosen != "Choose an example…":
            st.caption("Copy this sample into the input above to test the model.")
            st.code(examples[chosen], language=None)
        run = st.button("✦ Analyze language", type="primary", use_container_width=True)
    with right:
        st.markdown("### What this workspace returns")
        st.markdown('<div class="panel"><b>01 · Prediction</b><p class="small-muted">Top language and model score, plus alternative candidates.</p><hr><b>02 · Text profile</b><p class="small-muted">Character and token counts, detected Unicode writing systems.</p><hr><b>03 · Evidence</b><p class="small-muted">Active character n-grams contributing to the predicted class.</p><hr><b>04 · Caution signals</b><p class="small-muted">Short-input and ambiguous-score warnings, without overstating certainty.</p></div>', unsafe_allow_html=True)
    if run:
        try:
            result = predict_text(
                sample, model, vectorizer, encoder, top_k=5
            )
            result["timestamp"] = datetime.now(timezone.utc).isoformat()
            st.session_state.last_result = result
            st.session_state.history.insert(0, result)
            st.session_state.history = st.session_state.history[:100]

            with st.spinner("Qwen is translating and interpreting the text..."):
                interpretation = interpret_text(
                    text=result["text"],
                    detected_language=result["language"],
                    output_language=output_language,
                )

            st.session_state.last_interpretation = interpretation
            st.session_state.interpretation_language = output_language

        except ValueError as exc:
            st.warning(str(exc))
        except RuntimeError as exc:
            st.error(str(exc))
    if st.session_state.get("last_result"):
        result = st.session_state.last_result
        st.divider()
        st.markdown("### Analysis result")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Predicted language", result["language"])
        if result.get("score_type"):
            m2.metric("SVM decision score", f"{result['confidence']:.4f}")
            st.caption("Decision scores are not probabilities and are not percentages.")
        else:
            m2.metric("Model score", f"{result['confidence']:.1%}")
        m3.metric("Characters", f"{result['character_count']:,}")
        m4.metric("Tokens (whitespace)", f"{result['word_count']:,}")
        if result["character_count"] < 12 or result["confidence"] < .55:
            st.warning(result["status"])
        else:
            st.success(result["status"])
        c1, c2 = st.columns([1, 1], gap="large")
        with c1:
            st.markdown("#### Top candidate languages")
            candidates = pd.DataFrame(result["candidates"])
            if result.get("score_type"):
                candidates = candidates.rename(columns={"confidence": "Decision score"})
            else:
                candidates["confidence"] = candidates["confidence"].map(lambda x: f"{x:.2%}")
                candidates = candidates.rename(columns={"confidence": "Model score"})
            st.dataframe(candidates.rename(columns={"language": "Language", "language_code": "Code"}), hide_index=True, use_container_width=True)
        with c2:
            st.markdown("#### Writing-system profile")
            if result["script_counts"]:
                st.bar_chart(pd.DataFrame({"Characters": result["script_counts"]}))
            else:
                st.info("No letter characters were detected in the input.")
            st.caption(result["script_summary"])
        st.markdown("#### Model evidence · active character n-grams")
        evidence = explain_prediction(result["text"], result, model, vectorizer, encoder)
        if evidence:
            ev = pd.DataFrame(evidence).rename(columns={"ngram": "Character n-gram", "contribution": "Linear contribution"})
            st.dataframe(ev, hide_index=True, use_container_width=True)
            st.caption("Positive linear contributions for the predicted class, weighted by the input TF-IDF values. These are model features, not a linguistic proof.")
        else:
            st.caption("No interpretable active feature evidence was available for this input.")
        st.download_button("Download result JSON", json.dumps(result, ensure_ascii=False, indent=2), file_name="language-analysis.json", mime="application/json")

        interpretation = st.session_state.get("last_interpretation")
        if interpretation:
            st.divider()
            st.markdown("### Translation & language intelligence")
            st.caption(
                "Generated by your local Ollama model. Linguistic details "
                "may require verification."
            )

            st.markdown("#### Translation")
            st.write(interpretation["translation"])

            st.markdown("#### Meaning")
            st.write(interpretation["meaning"])

            st.markdown("#### Context")
            st.write(interpretation["context"])

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("#### Transliteration")
                st.write(interpretation["transliteration"] or "Not applicable")
            with col_b:
                st.markdown("#### Writing system")
                st.write(interpretation["script"] or result["script_summary"])

            st.markdown("#### Word-by-word explanation")
            words = interpretation["word_by_word"]
            if words:
                st.dataframe(
                    pd.DataFrame(words),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No word-level breakdown was returned.")

            st.markdown("#### Idioms & expressions")
            st.write(interpretation["idioms"] or "No specific idioms identified.")

            st.markdown("#### Cultural notes")
            st.write(
                interpretation["cultural_notes"]
                or "No additional cultural notes returned."
            )

            st.download_button(
                "Download full interpretation JSON",
                json.dumps(interpretation, ensure_ascii=False, indent=2),
                file_name="language-interpretation.json",
                mime="application/json",
                key="download_interpretation",
            )


elif page == "Batch lab":
    st.markdown("### Batch classification lab")
    st.markdown('<p class="small-muted">Upload a CSV containing a <code>text</code> column. Processing uses the existing local classifier.</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Choose CSV", type=["csv"])
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            if "text" not in df.columns:
                st.error("CSV must contain a column named `text`.")
            else:
                st.caption(f"Loaded {len(df):,} rows")
                limit = st.slider("Rows to process in this run", 1, max(1, min(len(df), 5000)), min(len(df), 1000))
                if st.button("Run batch analysis", type="primary"):
                    rows = []
                    progress = st.progress(0)
                    for i, value in enumerate(df["text"].fillna("").astype(str).iloc[:limit], start=1):
                        if value.strip():
                            r = predict_text(value, model, vectorizer, encoder, top_k=3)
                            score_type = r.get("score_type", "Model probability")
                            is_svm = "SVM" in score_type
                            candidate_text = []
                            for candidate in r["candidates"]:
                                if is_svm:
                                    score_label = "{:.3f}".format(candidate["confidence"])
                                else:
                                    score_label = "{:.1%}".format(candidate["confidence"])
                                candidate_text.append(
                                    "{} ({})".format(candidate["language"], score_label)
                                )
                            rows.append({
                                "text": value,
                                "language": r["language"],
                                "language_code": r["language_code"],
                                "model_score": r["confidence"],
                                "score_type": score_type,
                                "status": r["status"],
                                "top_3": " | ".join(candidate_text),
                            })
                        else:
                            rows.append({
                                "text": value,
                                "language": "",
                                "language_code": "",
                                "model_score": None,
                                "score_type": "",
                                "status": "Empty input",
                                "top_3": "",
                            })
                        progress.progress(i / min(limit, len(df)))
                    out = pd.DataFrame(rows)
                    st.session_state.batch_results = out
                    st.success(f"Processed {len(out):,} rows.")
        except Exception as exc:
            st.error(f"Could not read the CSV: {exc}")
    if st.session_state.get("batch_results") is not None:
        out = st.session_state.batch_results
        a, b, c = st.columns(3)
        a.metric("Rows processed", f"{len(out):,}")
        b.metric("Unique predicted languages", int(out.loc[out.language != "", "language"].nunique()))
        c.metric(
            "Rows needing review",
            int(out["status"].str.contains(
                "Short input|Ambiguous|cautiously|Low ",
                case=False, regex=True, na=False,
            ).sum()),
        )
        st.dataframe(out, use_container_width=True, hide_index=True)
        st.download_button("Download batch results CSV", out.to_csv(index=False).encode("utf-8-sig"), "language-batch-results.csv", "text/csv", type="primary")

elif page == "Model insights":
    st.markdown("### Model & dataset insights")
    st.markdown('<p class="small-muted">These are dataset-level facts, not claims about held-out model performance.</p>', unsafe_allow_html=True)
    data_path = ROOT / "data" / "train.csv"
    if data_path.exists():
        @st.cache_data(show_spinner="Reading dataset summary…")
        def dataset_summary(path: str, modified: float):
            data = pd.read_csv(path)
            labels = data["labels"].value_counts().sort_values(ascending=False) if "labels" in data else pd.Series(dtype=int)
            lengths = data["text"].fillna("").astype(str).str.len() if "text" in data else pd.Series(dtype=int)
            return len(data), labels, lengths.describe(), data.columns.tolist()
        total, label_counts, length_stats, columns = dataset_summary(str(data_path), data_path.stat().st_mtime)
        a, b, c, d = st.columns(4)
        a.metric("Dataset rows", f"{total:,}")
        b.metric("Model labels", len(encoder.classes_))
        c.metric("Vocabulary features", f"{len(vectorizer.vocabulary_):,}")
        d.metric("Median text length", f"{length_stats.get('50%', 0):.0f}")
        st.markdown("#### Label distribution")
        if not label_counts.empty:
            st.bar_chart(label_counts.rename(index=lambda code: str(code)))
            st.dataframe(label_counts.rename_axis("Label code").rename("Rows").reset_index(), hide_index=True, use_container_width=True)
        st.caption("Columns found: " + ", ".join(columns))
    else:
        st.info("Dataset file not found. Model predictions still work with the saved artifacts.")
    st.markdown("#### Artifact inventory")
    artifact_rows = []
    for name in ["language_model.pkl", "tfidf_vectorizer.pkl", "label_encoder.pkl"]:
        p = ROOT / name
        artifact_rows.append({"Artifact": name, "Present": p.exists(), "Size (KB)": round(p.stat().st_size / 1024, 1) if p.exists() else None})
    st.dataframe(pd.DataFrame(artifact_rows), hide_index=True, use_container_width=True)
    st.warning("The serialized model was trained with a specific scikit-learn version. If you see an InconsistentVersionWarning, use the documented environment or retrain from the supplied dataset before relying on reproducibility.")

elif page == "AI agent report":
    st.markdown("### Local-first agent analysis")
    st.markdown('<p class="small-muted">The deterministic analysis agents run locally. The optional LLM explanation uses an Ollama model installed on your machine; no cloud API is required.</p>', unsafe_allow_html=True)
    default_text = st.session_state.get("last_result", {}).get("text", "")
    text = st.text_area("Text to analyze", value=default_text, height=160, key="agent_text")
    if st.button("Run analysis agents", type="primary"):
        try:
            prediction = predict_text(text, model, vectorizer, encoder)
            quality = run_quality_agent(text, prediction)
            script = run_script_agent(prediction)
            report = run_report_agent(text, prediction, quality, script)
            st.session_state.agent_report = report
            st.session_state.agent_text_used = text
        except ValueError as exc:
            st.warning(str(exc))
    if st.session_state.get("agent_report"):
        report = st.session_state.agent_report
        st.markdown("#### Agent output")
        st.write(report["summary"])
        a, b = st.columns(2)
        with a:
            st.markdown("**Input & Confidence Agent**")
            st.json({"findings": report["quality_findings"]})
        with b:
            st.markdown("**Script Profiling Agent**")
            st.json(report["input_profile"]["scripts"])
        st.markdown("**Report Agent · structured report**")
        st.json(report)
        with st.expander("Optional local LLM explanation (Ollama)"):
            model_name = st.text_input("Ollama model", value="qwen2.5:3b", help="Pull this model in your terminal with: ollama pull qwen2.5:3b")
            if st.button("Generate local LLM explanation"):
                with st.spinner("Calling your local Ollama model…"):
                    try:
                        explanation = run_local_llm_explanation(st.session_state.agent_text_used, report, model_name=model_name)
                        st.session_state.local_explanation = explanation
                    except RuntimeError as exc:
                        st.error(str(exc))
            if st.session_state.get("local_explanation"):
                st.write(st.session_state.local_explanation)
        st.download_button("Download agent report JSON", json.dumps(report, ensure_ascii=False, indent=2), "agent-report.json", "application/json")

elif page == "About & setup":
    st.markdown("### Project architecture")
    st.markdown("""
    **Active inference:** character-and-word TF-IDF features → Linear SVM → language label + decision scores (not probabilities).

    **Analysis modules:** Unicode script profiling, input-quality checks, active feature evidence and structured reports.

    **Optional local LLM:** Ollama integration is isolated from core prediction. The app remains usable when Ollama is not installed or running.

    **Engineering principles:** preserve the trained baseline, make uncertainty visible, separate modules, keep data local by default, and avoid claiming unmeasured accuracy.
    """)
    st.markdown("#### Quick start")
    st.code("python -m venv .venv\n# Windows PowerShell\n.venv\\Scripts\\Activate.ps1\npip install -r requirements.txt\nstreamlit run app.py", language="powershell")
    st.markdown("#### Optional local LLM setup")
    st.code("# Install Ollama separately, then:\nollama pull qwen2.5:3b\n# Start Ollama and use the AI agent report page", language="bash")
    st.info("The active Linear SVM is the primary predictor. Its decision scores are not probabilities. Agent outputs are supporting analysis; optional LLM explanations must not be treated as ground truth.")

if st.session_state.history:
    with st.sidebar.expander(f"Recent predictions ({len(st.session_state.history)})"):
        history_df = pd.DataFrame([
            {
                "Time (UTC)": r.get("timestamp", ""),
                "Language": r["language"],
                "Score": (
                    "{:.3f}".format(r["confidence"])
                    if r.get("score_type")
                    else "{:.1%}".format(r["confidence"])
                ),
                "Score type": r.get("score_type", "Model probability"),
                "Characters": r["character_count"],
                "Status": r["status"],
            }
            for r in st.session_state.history
        ])
        st.dataframe(history_df, hide_index=True, use_container_width=True)
        if st.button("Clear session history"):
            st.session_state.history = []
            st.rerun()

st.markdown("---")
st.markdown('<p class="small-muted">Language Intelligence Studio · Portfolio edition · Active model: Linear SVM · Local-first by design</p>', unsafe_allow_html=True)
