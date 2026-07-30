# Software Bug Analyzer: NLP + RAG + LLM System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![FAISS](https://img.shields.io/badge/FAISS-1.8.0-008080?style=for-the-badge)](https://faiss.ai)
[![spaCy](https://img.shields.io/badge/spaCy-3.7+-09A3D5?style=for-the-badge&logo=spacy&logoColor=white)](https://spacy.io)

An end-to-end AI-powered **Software Bug Analyzer** system that combines 6-stage Natural Language Processing (NLP), FAISS Vector RAG retrieval over historical GitHub issues, and abstract LLM reasoning (Groq / OpenAI / Gemini / Ollama) to automatically categorize, triage, assign teams, and recommend QA test suites for incoming software bug reports.

---

## 📐 System Architecture & Workflow

```
                             [ Historical GitHub Bug Dataset ]
                                             │
                                             ▼
                             [ Embeddings: SentenceTransformer ]
                                             │
                                             ▼
                        ┌─────────────────────────────────────────┐
                        │      FAISS Vector Store (RAG)           │
                        │   (Internal context, hidden from user)  │
                        └────────────────────┬────────────────────┘
                                             │
   Developer Bug Report ─────────────────────┼─────────────────────┐
   (Title + Description)                     │                     │
             │                               │                     │
             ▼                               ▼                     │
   [ 1. Raw Sanitization ]           [ Top 5 Vector Search ]       │
             │                               │                     │
             ▼                               │                     │
   [ 2. spaCy Tokenization ]                 │                     │
             │                               │                     │
             ▼                               │                     │
   [ 3. Software Entities ]                  │                     │
   (Action, Component, Error, Condition)     │                     │
             │                               │                     │
             ▼                               │                     │
   [ 4. Keyphrase Extraction ] ──────────────┘                     │
   (Hybrid: KeyBERT + YAKE + spaCy)                                │
             │                                                     │
             └───────────────────────┬─────────────────────────────┘
                                     │
                                     ▼
                      ┌──────────────────────────────┐
                      │    Abstracted LLM Client     │
                      │  (Groq / OpenAI / Gemini)    │
                      └──────────────┬───────────────┘
                                     │
                                     ▼
                     ┌────────────────────────────────┐
                     │     Streamlit Dashboard        │
                     │  (Category, Severity, Module,  │
                     │   Causes, Team, QA Tests)      │
                     └────────────────────────────────┘
```

---

## 🧠 Why Natural Language Processing (NLP) is Essential Before LLM

A common misconception is that Modern LLMs can directly ingest raw bug reports without prior NLP processing. However, introducing a **6-stage NLP pipeline** prior to RAG and LLM inference is critical for production software engineering:

1. **Noise Reduction & Token Efficiency**: Stack traces, log timestamps (`INFO:`, `File "...", line 42`), raw URLs, and markdown code fences inflate LLM context windows with irrelevancies. Preprocessing strips noise to retain high-density technical text.
2. **Deterministic Entity Extraction**: Regex & spaCy custom patterns explicitly extract domain software entities—**Actions** (*uploading*, *crashes*), **Components** (*images*, *auth_controller.py*), **Error Types** (*ERR_OOM_500*, *HTTP 504*), and **Conditions** (*>20MB*, *after 30s*).
3. **Optimized RAG Retrieval Queries**: Passing a full raw 500-word bug report to vector search generates diluted embedding vectors. Extracting keyphrases via **KeyBERT + YAKE + spaCy** produces targeted, high-precision search queries for FAISS vector search.
4. **Structured Input Taxonomies**: NLP token distributions enforce consistency, allowing rule-based fallback heuristics when API keys or network connections fail.

---

## 🛠️ Project Structure

```
software-bug-analyzer/
├── config/
│   ├── config.yaml               # System parameters & taxonomy lists
│   └── prompts.yaml                # Prompt templates with strict JSON constraints
├── data/
│   ├── raw/                        # Raw GitHub issues payload
│   └── processed/                  # Cleaned & sanitized dataset
├── faiss_index/                    # Persisted FAISS vector index & metadata
├── src/
│   ├── dataset/                    # Collector & cleaner modules
│   ├── nlp/                        # Preprocessor, Tokenizer, Entity & Keyphrase Extractors
│   ├── embeddings/                 # SentenceTransformer encoder wrapper
│   ├── retrieval/                  # FAISS vector store & RAG retriever
│   ├── llm/                        # Prompt manager, multi-provider LLM client, Predictor
│   └── pipeline.py                 # Master pipeline orchestrator
├── backend/
│   ├── ai_service.py               # Streamlit resource cached service
│   ├── db.py                       # SQLite database manager (bugs.db)
│   └── analytics.py                # Plotly & Pandas chart transformers
├── pages/
│   ├── 1_Input.py                  # Bug submission form & live diagnosis visualizer
│   ├── 2_Output.py                 # Full analysis detail viewer
│   ├── 3_Dashboard.py              # Analytics metrics, charts, & CSV export
│   └── 4_History.py                # Searchable, filterable bug history table
├── app.py                          # Streamlit application main entry point
├── main.py                         # Standalone Person 1 CLI demo
├── person2_integration_example.py  # Person 2 integration reference code
├── scripts/                        # Dataset collection, cleaning, and FAISS build scripts
├── tests/test_pipeline.py          # PyTest / Unittest test suite
└── requirements.txt
```

---

## 🚀 Quick Start & Setup

### 1. Prerequisites & Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 2. Install Dependencies & spaCy Language Model
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Environment Configuration (Optional)
Copy `.env.example` to `.env` and fill in your API keys (the system runs out of the box using built-in sample data and rule-based heuristics if keys are omitted):
```env
GROQ_API_KEY=gsk_...
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AIzaSy...
```

> If no LLM API credentials are provided, the pipeline automatically falls back to a deterministic rule-based prediction engine so the project is still usable for demo and offline evaluation.

---

## 🧪 Verification & Running Demos

### Run Automated Unit Tests (100% Offline Compatible)
```bash
python tests/test_pipeline.py
```

### Run Standalone Person 1 CLI Demo
```bash
python main.py
```

### Run Person 2 Streamlit Web Application
```bash
streamlit run app.py
```

---

## 🤝 Person 1 vs. Person 2 Split & Integration Contract

- **Person 1 (AI Pipeline)**: Owns `src/` modules (`pipeline.py`). Exposes `BugAnalysisPipeline.analyze(title, description)`.
- **Person 2 (Application Engineer)**: Owns `backend/` and `pages/` Streamlit UI. Consumes Person 1 via `from src.pipeline import BugAnalysisPipeline`.
- **Privacy Contract**: Retracted historical RAG bugs are fed strictly into the internal LLM context prompt (`_internal`) and are **NEVER shown to the user** on the Streamlit frontend.
