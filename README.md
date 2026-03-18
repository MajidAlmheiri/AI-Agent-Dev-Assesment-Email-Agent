# AI Agent Dev Assessment — Email Inquiry Agent (Agent 2)

## Overview

An LLM-driven agent that acts as a **Professional Business Consultant for UAE Free Zones**. When an investor sends a natural-language inquiry (e.g., _"I want to open a trading company in JAFZA — what is the estimated cost?"_), the agent:

1. **Parses intent** — extracts the company type, target free zone, industry keyword, and optional employee headcount.
2. **Queries the dataset** — calls LangChain tools against a real company dataset to calculate average license fees, capital requirements, and setup cost ranges.
3. **Responds in professional email format** — returns a structured cost estimate with a subject line, greeting, itemized breakdown, and sign-off.

All intent parsing, tool routing, and email composition are handled by the LLM — there is no hardcoded regex or template logic.

## Tech Stack

| Component | Detail |
|-----------|--------|
| Language | Python 3.13 |
| LLM | [Ollama](https://ollama.com/) running `qwen2.5:7b` locally (free, no API key) |
| Framework | LangChain (`ChatOllama` + tool-calling loop) |
| Data API | FastAPI mock server (`/companies` endpoint) with CSV fallback |
| Dataset | `data/FreeZone_Companies_Data.csv` — registered free zone companies |

## Project Structure

```
├── main.py                  # CLI entry point (interactive loop)
├── agent/
│   └── query_agent.py       # LLM agent with memory + tool-calling loop
├── tools/
│   └── api_tools.py         # LangChain tools: count_companies, get_setup_estimates
├── api/
│   └── mock_api.py          # FastAPI mock REST API
├── data/
│   └── FreeZone_Companies_Data.csv
├── smoke_test.py            # Automated validation script
├── requirements.txt
└── .env                     # Ollama model configuration
```

## Setup

### 1. Install Python dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Install and start Ollama
(Note: The commands below are for macOS. If using Windows/Linux, download Ollama directly from ollama.com)
```bash
brew install ollama
brew services start ollama
ollama pull qwen2.5:7b
```


### 3. Configure environment

The `.env` file is already set up. It contains:

```
OLLAMA_MODEL=qwen2.5:7b
```

## How to Run

You need **two terminals**: one for the API server and one for the agent.

**Terminal 1 — Start the mock API server:**

```bash
uvicorn api.mock_api:app --reload
```

This starts the company data API at `http://localhost:8000`. Keep this running.

**Terminal 2 — Start the agent:**

```bash
python3 main.py
```

You will see:

```
Agent 2 (Email Inquiry w/ Memory) is ready  —  LLM-driven mode (Ollama local)
===============================================
   SHARJAH FREE ZONE RESEARCH ASSISTANT (v1.0)
===============================================
[User]:
```

## Example Queries

### Cost estimate inquiry

```
[User]: I want to open a trading company in JAFZA with 15 employees — what is the estimated cost?
```

The agent will call `get_setup_estimates` with `freezone="Jebel Ali Free Zone"`, `industry="trading"`, `target_employees=15` and reply with a professional email containing the cost breakdown.

### Cost estimate without headcount

```
[User]: What are the setup costs for a food business in Hamriyah Free Zone?
```

The agent passes `target_employees=0`, and the tool uses the industry average employee count.

### Company count query

```
[User]: How many active logistics companies are in SAIF Zone?
```

The agent calls `count_companies` and returns the count in a professional response.

### Follow-up questions

The agent maintains conversational memory, so you can ask follow-ups:

```
[User]: What about with 30 employees instead?
```

It will reuse the free zone and industry from the previous turn.

## How It Works

1. **System prompt** instructs the LLM to act as a Free Zone consultant, always reply in email format, translate abbreviations (JAFZA → Jebel Ali Free Zone), and extract broad industry keywords.
2. **User message** is appended to conversational memory.
3. **LLM decides** whether to call a tool (`get_setup_estimates` or `count_companies`) based on intent.
4. **Tool results** are appended back to memory as `ToolMessage`s.
5. **LLM composes** the final professional email using the real data from the tools.
6. **Response** is returned to the user via the CLI.

## Smoke Test

```bash
python3 smoke_test.py
```

Validates that both the count query and cost estimate flows execute without errors.
