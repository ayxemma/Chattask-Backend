# SGLang Parser Integration

Experimental architecture for routing ChatTask task parsing through an external SGLang server (OpenAI-compatible API). **Disabled by default** — production on Render continues to use OpenAI unless explicitly enabled.

## Architecture overview

```
POST /parse
    └── task_parser_service.parse_task_text()
            ├── USE_SGLANG_PARSER=false → OpenAIClient (default)
            └── USE_SGLANG_PARSER=true  → SGLangClient
                    └── on failure → fallback OpenAIClient
```

### Module layout

| Path | Role |
|------|------|
| `app/services/task_parser_service.py` | Provider selection, latency logging, SGLang→OpenAI fallback |
| `app/llm_clients/base.py` | `BaseLLMClient`, `ParseTaskContext`, shared prompt builders |
| `app/llm_clients/openai_client.py` | Production OpenAI parser (`AsyncOpenAI`) |
| `app/llm_clients/sglang_client.py` | Experimental SGLang parser (same OpenAI SDK, custom `base_url`) |
| `app/llm_clients/parse_normalization.py` | Maps raw LLM JSON → iOS `ParseResponse` contract |
| `app/models/llm_parse_output.py` | Pydantic validation for structured LLM JSON |

Transcription (`POST /transcribe`), task-target resolution, and command interpretation remain on the existing OpenAI httpx path in `openai_service.py`.

## Environment setup

Add to `.env` (see `.env.example`):

```bash
# Required for all deployments
OPENAI_API_KEY=sk-...

# Experimental SGLang parser — leave disabled in production
USE_SGLANG_PARSER=false
SGLANG_BASE_URL=
SGLANG_API_KEY=
SGLANG_MODEL=
```

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_SGLANG_PARSER` | `false` | When `true`, `/parse` tries SGLang first |
| `SGLANG_BASE_URL` | empty | OpenAI-compatible base URL, e.g. `https://your-app.modal.run/v1` |
| `SGLANG_API_KEY` | empty | Bearer token if the SGLang gateway requires auth |
| `SGLANG_MODEL` | empty | Model name exposed by the SGLang server |

When `USE_SGLANG_PARSER=false` (Render production default), no SGLang configuration is read and behavior is unchanged.

## Connecting Modal-hosted SGLang later

This PR does **not** deploy or run SGLang. When you stand up a GPU endpoint (e.g. on Modal):

1. Deploy SGLang with the OpenAI-compatible `/v1/chat/completions` route.
2. Note the public HTTPS URL (e.g. `https://your-workspace--sglang-serve.modal.run/v1`).
3. Set Render env vars on a **staging** service first:

   ```bash
   USE_SGLANG_PARSER=true
   SGLANG_BASE_URL=https://your-workspace--sglang-serve.modal.run/v1
   SGLANG_API_KEY=<optional>
   SGLANG_MODEL=<model-id-on-server>
   ```

4. Keep `OPENAI_API_KEY` set — fallback uses OpenAI if SGLang is unreachable or returns invalid JSON.

The SGLang client is implemented with the official OpenAI Python SDK:

```python
from openai import AsyncOpenAI

client = AsyncOpenAI(
    api_key=SGLANG_API_KEY or "not-needed",
    base_url=SGLANG_BASE_URL,
)
```

## Example curl requests

### Parse via backend (default OpenAI path)

```bash
curl -s -X POST http://localhost:3000/parse \
  -H "Content-Type: application/json" \
  -d '{
    "text": "remind me to call John at 3pm tomorrow",
    "now": "2026-05-19T10:00:00-04:00",
    "timezone": "America/New_York"
  }' | jq .
```

### Direct SGLang server (after deployment)

```bash
curl -s -X POST "$SGLANG_BASE_URL/chat/completions" \
  -H "Authorization: Bearer $SGLANG_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"$SGLANG_MODEL"'",
    "temperature": 0,
    "response_format": {"type": "json_object"},
    "messages": [
      {"role": "system", "content": "Return JSON with intent, title, datetime, needs_time, recurrence, confidence."},
      {"role": "user", "content": "remind me to call John at 3pm tomorrow"}
    ]
  }' | jq .
```

## Structured JSON schema

LLM output is validated with Pydantic (`StructuredTaskParseOutput`). The full production schema matches the iOS `ParseResponse` contract. A simplified experimental schema is also supported:

```json
{
  "intent": "create_task",
  "title": "Call John",
  "datetime": "2026-05-20T15:00:00-04:00",
  "needs_time": true,
  "recurrence": null,
  "confidence": 0.92
}
```

Field mapping:

| Simplified | Production (`ParseResponse`) |
|------------|------------------------------|
| `intent: create_task` | `action_type: reminder` |
| `datetime` | `scheduled_at` |
| `needs_time` | `has_specific_time` |

## Fallback behavior

When `USE_SGLANG_PARSER=true`:

1. Request goes to `SGLangClient.parse_task()`.
2. On **any** failure (network, HTTP error, invalid JSON, validation error):
   - A warning is logged with the exception.
   - The same request is retried via `OpenAIClient`.
3. The HTTP response succeeds if OpenAI fallback succeeds.
4. Latency logs include `fallback_from=sglang` when fallback was used.

Log line format:

```
taskParse provider=openai latency_ms=842.3 success=True fallback_from=sglang
```

## Production safety

- **`USE_SGLANG_PARSER` defaults to `false`** — no code path change on Render unless the env var is set.
- **OpenAI remains required** — startup still validates `OPENAI_API_KEY`; fallback depends on it.
- **SGLang is opt-in per environment** — enable only on staging or a dedicated canary service.
- **No local SGLang server** — this PR adds client wiring only; zero GPU infra in the backend repo.
- **API contract unchanged** — `/parse` still returns the same `ParseResponse` JSON to iOS.

## Local testing without deployment

1. Install deps: `pip install -r requirements-dev.txt`
2. Run tests (mocks, no live LLM): `pytest`
3. Run the API with OpenAI only (default):

   ```bash
   export OPENAI_API_KEY=sk-...
   uvicorn app.main:app --reload --port 3000
   ```

4. Test SGLang **selection logic** without a real server:

   ```bash
   pytest tests/test_task_parser_service.py -v
   ```

5. To exercise a real SGLang endpoint locally, point env vars at your running server — do not commit secrets.
