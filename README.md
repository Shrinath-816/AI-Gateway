# The AI Gateway — Week 1 Project

A streaming, non-blocking FastAPI backend in front of an LLM provider
(via [OpenRouter](https://openrouter.ai)), with accurate per-request
token usage tracking, cost calculation, and model tiering.

This is Week 1 of a 10-week self-study AI Engineering program. See
`week-1-notes.md` (provided separately) for the concept notes this
project implements.

## What this project demonstrates

| Concept | Where it lives |
|---|---|
| Async, non-blocking LLM calls | `app/llm_client.py` (uses `AsyncOpenAI`) |
| SSE streaming to clients | `app/main.py` → `POST /chat` |
| Accurate token usage capture | `app/llm_client.py`, `app/usage_tracker.py` |
| Cost calculation | `app/pricing.py` |
| Model tiering | `app/model_router.py` |

## Project structure

```
ai-gateway/
├── app/
│   ├── __init__.py
│   ├── main.py            # FastAPI app, routes
│   ├── config.py          # Settings (env vars)
│   ├── schemas.py         # Pydantic request/response models
│   ├── llm_client.py      # Async OpenRouter client wrapper
│   ├── model_router.py    # Complexity -> model tier routing
│   ├── pricing.py         # Pricing table + cost calculator
│   └── usage_tracker.py   # Per-request telemetry
├── tests/
│   ├── test_model_router.py
│   ├── test_cost_calculator.py
│   └── test_usage_tracker.py
├── requirements.txt
├── Dockerfile
├── .env.example
└── README.md
```

## Setup

1. **Get an OpenRouter API key**: https://openrouter.ai/keys

2. **Check current model availability and pricing** at
   https://openrouter.ai/models — the model slugs in
   `app/model_router.py` and the prices in `app/pricing.py` are
   placeholders and may not reflect what's currently available or
   what it currently costs. Update both files before relying on this
   for real cost tracking.

3. **Create your environment file:**
   ```bash
   cp .env.example .env
   # then edit .env and paste your real OPENROUTER_API_KEY
   ```

4. **Install dependencies** (a virtual environment is recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

5. **Run the server:**
   ```bash
   uvicorn app.main:app --reload
   ```
   The API will be available at http://localhost:8000, with
   interactive docs at http://localhost:8000/docs.

## Usage

### Streaming chat (SSE)
```bash
curl -N -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain attention in one paragraph.", "complexity": "standard"}'
```
`-N` disables curl's output buffering so you can see tokens arrive
incrementally, the way a browser `EventSource` would.

### Non-streaming chat
```bash
curl -X POST http://localhost:8000/chat/sync \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Summarize RAG in two sentences.", "complexity": "simple"}'
```

### Usage summary
```bash
curl http://localhost:8000/usage
```

### Health check
```bash
curl http://localhost:8000/health
```

## Running tests

```bash
pytest
```

Tests cover the model router, cost calculator, and usage tracker —
all pure logic, no network calls, so they run without an API key.

## Running with Docker

```bash
docker build -t ai-gateway .
docker run -p 8000:8000 --env-file .env ai-gateway
```

## Definition of Done (Week 1 milestone)

- [x] `/chat` streams tokens incrementally (verifiable via `curl -N`)
- [x] Every request logs model, prompt/completion tokens, cost, and latency
- [x] Changing `complexity` in the request body changes which model is called
- [x] A request timeout is enforced so a hung provider call can't hang the server indefinitely
- [x] Client disconnects mid-stream stop further token consumption

## Known limitations (intentional — addressed in later weeks)

- No retry/fallback logic if the provider errors out (Week 8: AI Reliability & Safety)
- No conversation history / multi-turn memory (Week 4: Agents & State)
- Usage tracker is in-memory only and resets on restart (fine for now; Week 5 introduces proper observability)
- Pricing table is manually maintained, not fetched live from OpenRouter
