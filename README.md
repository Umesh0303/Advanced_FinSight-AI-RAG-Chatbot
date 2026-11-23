# FinSight AI – Role-Based RAG Chatbot

## Project objective

FinSight AI is planned as a standalone local Python application for role-based question answering over finance documents. This repository contains the Phase 1 project scaffold; the RAG chatbot is not implemented yet.

## Technology stack

- Python 3.13
- LangChain
- Groq for the language model
- Google Gemini for embeddings
- ChromaDB with one persistent database and four logical collections
- PDF and DOCX documents

## User roles

- Admin: authorized for all four collections
- Business Analyst
- Risk Analyst
- Customer

Non-admin roles will only access explicitly authorized content.

## Basic setup

From the project root in PowerShell:

```powershell
py -3.13 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, allow it for the current process and retry:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

Set `GROQ_API_KEY`, `GOOGLE_API_KEY`, and `GROQ_MODEL` in `.env` before future integrations are implemented. Ingestion also requires a private `INGESTION_MANIFEST_HMAC_KEY` of at least 32 characters to authenticate saved authorization decisions. Generate one locally with `python -c "import secrets; print(secrets.token_hex(32))"` and keep it stable across restarts. Keep `.env` private; `.env.example` documents the required variable names.

## Future development stages

1. Implement document loading and automatic role classification for PDF and DOCX uploads.
2. Add text splitting and Google Gemini embeddings.
3. Configure the persistent ChromaDB database and four logical collections.
4. Implement role-based retrieval and access control.
5. Integrate Groq and assemble the RAG chain.
6. Add and run tests for ingestion, retrieval, the RAG chain, and role isolation.