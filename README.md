# Language Intelligence Studio

A local-first multilingual text analysis application built on the original 20-language character n-gram TF-IDF + Logistic Regression classifier. This portfolio upgrade preserves the existing dataset and serialized model artifacts while adding a modular inference engine, model evidence inspection, Unicode script profiling, batch CSV processing, structured analysis agents, optional local LLM explanations, tests, and Docker support.

## Features

- **Language detection:** 20 labels using the original saved model and vectorizer.
- **Candidate ranking:** inspect the top five language candidates and their model scores.
- **Text profiling:** character count, whitespace-separated token count, and heuristic Unicode script counts.
- **Model evidence:** active character n-grams with positive linear contribution for the predicted class.
- **Batch lab:** classify a CSV with a `text` column and export results.
- **Local-first analysis agents:** input/confidence checks, script profiling, and structured report assembly.
- **Optional local LLM:** Ollama-powered explanation; not required for core predictions and no hosted API key is needed.
- **Model insights:** inspect dataset label distribution and artifact inventory.
- **Engineering:** pytest checks, GitHub Actions CI, Dockerfile and Compose configuration.

## Architecture

```text
Streamlit UI
  ├── Analyze text ──> Inference engine ──> TF-IDF ──> Logistic Regression
  │                        ├── candidate ranking
  │                        ├── Unicode script profiler
  │                        └── active n-gram evidence
  ├── Batch lab ─────> same inference engine ──> CSV export
  └── Agent report
       ├── Input & Confidence Agent
       ├── Script Profiling Agent
       ├── Report Agent
       └── Optional Ollama explanation (local endpoint)
```

## Quick start (Windows PowerShell)

Python 3.11 is recommended for the documented environment. Use a virtual environment to avoid changing other Python projects.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL printed by Streamlit, usually `http://localhost:8501`.

### Linux / macOS

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Optional local LLM with Ollama

The classifier and deterministic analysis agents work without an LLM. To enable local natural-language explanations:

1. Install Ollama from its official website.
2. Pull a small instruct model compatible with your machine, for example:

   ```bash
   ollama pull qwen2.5:3b
   ```

3. Ensure Ollama is running and open **AI agent report → Optional local LLM explanation** in the app.
4. If needed, configure `OLLAMA_BASE_URL` and `OLLAMA_MODEL` environment variables. `.env.example` documents the defaults; the application does not automatically load `.env` files.

A 3B model can still be slow or memory-intensive on some CPU-only systems. You can skip the LLM entirely; it is an optional reporting layer and does not affect the baseline classifier.

## Batch CSV format

Upload a UTF-8 CSV with a column named `text`:

```csv
text
"This is an example sentence in English."
"Bonjour, comment allez-vous aujourd'hui?"
```

The app limits each run to 5,000 rows to keep interactive processing responsive. Blank rows are retained and marked as empty inputs.

## Tests and CI

```bash
pytest -q
```

GitHub Actions runs the test suite on pushes and pull requests. The tests check artifact loading, a known English prediction, candidate ordering, empty input validation, and multi-script profiling.

## Docker

```bash
docker compose up --build
```

Visit `http://localhost:8501`. Ollama is not included in the container; if you want to use a host Ollama service from Docker, configure a reachable `OLLAMA_BASE_URL`.

## Original baseline and reproducibility

The following original files are preserved:

- `language_model.pkl`
- `tfidf_vectorizer.pkl`
- `label_encoder.pkl`
- `data/train.csv`
- `notebooks/Multilingual Text Classifier Project-1.py`
- original screenshot and devcontainer configuration

The serialized artifacts were produced with scikit-learn 1.6.0, so the requirements pin that version to reduce pickle compatibility warnings. If you retrain the model, save the new artifacts and record evaluation results separately. Do not interpret Logistic Regression `predict_proba` outputs as calibrated probabilities without calibration testing.

## Important limitations

- The classifier assigns one language label to the entire input; it does not perform token-level language detection or reliable code-switching segmentation.
- Unicode script profiling is heuristic and does not prove a language identity.
- Top n-gram contributions provide a view into the linear model, not a complete causal explanation.
- Confidence thresholds are simple caution signals, not statistically calibrated uncertainty estimates.
- Accuracy, F1 and latency should be measured on a held-out test set before reporting them in a CV or project presentation.

## Technology stack

Python · Streamlit · Scikit-learn · TF-IDF · Logistic Regression · NumPy · Pandas · Joblib · Ollama (optional) · pytest · Docker · GitHub Actions

## Project goal

Demonstrate a practical, CPU-friendly NLP application with a clear baseline, transparent model inspection, modular analysis workflows, optional local LLM integration, and reproducible software-engineering practices.
