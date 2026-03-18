# AI Agent Dev Assessment (Email + Query Agent)

This project implements two capabilities in one CLI agent using Python + LangChain tools:

1. **Agent 1 (Database Query Agent)**
   - Answers questions like: _"How many active logistics companies are registered in JAFZA?"_
   - Uses `count_companies` tool against mock REST API (with CSV fallback).

2. **Agent 2 (Email Inquiry Agent)**
   - Handles investor email inquiries about setup cost.
   - Extracts free zone, industry keyword, and optional headcount.
   - Uses `get_setup_estimates` tool to return a professional email-style response with:
     - average license fee estimate
     - average capital requirement
     - setup cost range

## Project Structure

- `main.py` – CLI app
- `agent/query_agent.py` – resilient agent executor
- `tools/api_tools.py` – tools and data logic
- `api/mock_api.py` – mock REST endpoint `/companies`
- `data/FreeZone_Companies_Data.csv` – dataset
- `smoke_test.py` – quick local validation

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Run (recommended quick mode, no cloud quota)

Use rule mode to avoid any LLM API limit issues:

```bash
export LLM_PROVIDER=rule
python3 main.py
```

## Run with Google Gemini

```bash
export LLM_PROVIDER=google
export GOOGLE_API_KEY="YOUR_REAL_KEY"
python3 main.py
```

## Run with Ollama (free local model)

```bash
brew install ollama
ollama serve
```

In another terminal:

```bash
ollama pull llama3.1:8b
export LLM_PROVIDER=ollama
export OLLAMA_MODEL=llama3.1:8b
python3 main.py
```

## Optional: Mock API server

Tools attempt the API first, then fallback to CSV automatically.

```bash
uvicorn api.mock_api:app --reload --port 8000
```

## Smoke Test

```bash
python3 smoke_test.py
```

This validates:
- Agent 1 style count query
- Agent 2 style setup estimate email
