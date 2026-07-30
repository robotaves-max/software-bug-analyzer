# Software Bug Analyzer Mega Prompt Specification

This document preserves the initial specification for the complete Software Bug Analyzer project.

=== PROJECT OVERVIEW ===
Title: Software Bug Analyzer - NLP + RAG + LLM

Workflow:
GitHub Bug Dataset (Historical, Title, Description, Labels, Resolution)
  -> Knowledge Base / RAG (FAISS)
  -> Developer submits new Bug Report (Title + Description)
  -> Text Preprocessing (remove markdown, HTML, URLs, code blocks, stack traces, logs)
  -> Tokenization (spaCy)
  -> Entity Extraction (Action, Component, Error Type, Condition, file paths, versions, OS)
  -> Keyphrase Extraction (Hybrid: KeyBERT + YAKE + spaCy noun chunks)
  -> Sentence Embedding (SentenceTransformer all-MiniLM-L6-v2)
  -> FAISS Retrieval (Top 5 relevant historical bugs - INTERNAL ONLY)
  -> LLM Analysis (Groq llama-3.1-70b OR GPT-4o-mini OR Gemini via abstract client)
  -> Developer Dashboard (Streamlit)
