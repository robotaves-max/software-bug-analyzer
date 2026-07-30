# GitHub Push Guide

This guide explains how to initialize a Git repository, commit files, and push the **Software Bug Analyzer** project to GitHub.

---

## 1. Initialize Git Repository
In the root directory of `software-bug-analyzer`:

```bash
git init
git branch -M main
```

---

## 2. Check Exclusions & Verify `.gitignore`
Verify that large data models, virtual environments, and `.env` secrets are excluded:

```bash
git status
```

*Ensure `.env`, `venv/`, `*.db`, `data/raw/*.json`, and `faiss_index/*.faiss` are ignored.*

---

## 3. Add and Commit Files

```bash
git add .
git commit -m "feat: Initial commit for Software Bug Analyzer (NLP + RAG + LLM + Streamlit)"
```

---

## 4. Connect to GitHub Remote Repository

Create a new empty repository on GitHub (e.g., `software-bug-analyzer`), then link it:

```bash
git remote add origin https://github.com/YOUR_USERNAME/software-bug-analyzer.git
git push -u origin main
```
