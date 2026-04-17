# ChatTask Backend

Minimal FastAPI backend for the ChatTask iOS app. Keeps the OpenAI API key server-side and exposes two endpoints:

- `POST /transcribe` — transcribe an audio file via OpenAI
- `POST /parse` — parse a natural language task string into structured JSON

---

## Setup

### 1. Clone / navigate to the project

```bash
cd ChatTask-backend
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

```bash
cp .env.example .env
```

Open `.env` and replace `your_key_here` with your actual OpenAI API key:

```
OPENAI_API_KEY=sk-...
```

---

## Run locally

```bash
uvicorn app.main:app --reload
```

The server starts at `http://127.0.0.1:8000`.

- Health check: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- Interactive API docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## Endpoints

### `GET /`

Health check.

**Response:**
```json
{ "status": "ok" }
```

---

### `POST /transcribe`

Transcribe an audio file.

**Request:** `multipart/form-data` with a field named `file`.

**Example curl:**
```bash
curl -X POST http://127.0.0.1:8000/transcribe \
  -F "file=@/path/to/audio.m4a"
```

**Response:**
```json
{
  "text": "pick up Ari at 6:15"
}
```

---

### `POST /parse`

Parse a natural language task string into structured data.

**Request:** JSON body with `text`, `now` (ISO 8601), and `timezone` (IANA name).

**Example curl:**
```bash
curl -X POST http://127.0.0.1:8000/parse \
  -H "Content-Type: application/json" \
  -d '{
    "text": "pick up Ari at 6:15",
    "now": "2026-04-16T17:00:00-04:00",
    "timezone": "America/New_York"
  }'
```

**Response:**
```json
{
  "action_type": "reminder",
  "title": "Pick up Ari",
  "notes": null,
  "scheduled_at": "2026-04-16T18:15:00-04:00",
  "confidence": 0.95,
  "language_code": "en"
}
```

---

## Running the automated tests

Tests live in `tests/` and use `pytest`.  No live OpenAI calls are made —
all upstream service calls are mocked.

```bash
# Activate your venv first, then:
pytest -v
```

To run a single file:

```bash
pytest tests/test_health.py -v
pytest tests/test_parse.py -v
pytest tests/test_transcribe.py -v
```

---

## Manual smoke tests

Start the server first:

```bash
uvicorn app.main:app --reload
```

**1. Health check**

```bash
curl http://127.0.0.1:8000/
# Expected: {"status":"ok"}
```

**2. Parse**

```bash
curl -X POST http://127.0.0.1:8000/parse \
  -H "Content-Type: application/json" \
  -d '{
    "text": "pick up Ari at 6:15",
    "now": "2026-04-16T17:00:00-04:00",
    "timezone": "America/New_York"
  }'
```

**3. Transcribe**

```bash
curl -X POST http://127.0.0.1:8000/transcribe \
  -F "file=@/path/to/audio.m4a"
```

> Requires a valid `OPENAI_API_KEY` in `.env` for `/parse` and `/transcribe`.

---

## Project structure

```
ChatTask-backend/
├── app/
│   ├── main.py               # FastAPI app entry point
│   ├── config.py             # Loads env vars
│   ├── routes/
│   │   ├── transcribe.py     # POST /transcribe
│   │   └── parse.py          # POST /parse
│   ├── services/
│   │   └── openai_service.py # OpenAI API calls
│   └── models/
│       ├── common.py         # Shared response models
│       └── parse_models.py   # ParseRequest / ParseResponse
├── tests/
│   ├── conftest.py           # Shared fixtures (TestClient)
│   ├── test_health.py        # GET / tests
│   ├── test_parse.py         # POST /parse tests
│   └── test_transcribe.py    # POST /transcribe tests
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```
