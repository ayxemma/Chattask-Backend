import logging

from fastapi import APIRouter, HTTPException, Request

from app.config import OPENAI_API_KEY, OPENAI_INTERPRET_MODEL
from app.models.parse_models import CommandInterpretRequest, CommandInterpretResponse
from app.services.openai_service import interpret_command
from app.util.request_timing import TimingSpan, correlation_from_headers

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/interpret-command", response_model=CommandInterpretResponse)
async def interpret_command_route(request: Request, body: CommandInterpretRequest):
    correlation = correlation_from_headers(dict(request.headers))
    if body.command_session_id and not correlation.command_session_id:
        correlation.command_session_id = body.command_session_id
    span = TimingSpan(label="interpret-command", correlation=correlation)
    logger.info(
        "interpret-command requestReceived request_id=%s command_session_id=%s candidateTaskCount=%s likelyColdStart=%s model=%s",
        body.request_id or correlation.request_id,
        correlation.command_session_id,
        len(body.candidate_tasks),
        correlation.likely_cold_start,
        OPENAI_INTERPRET_MODEL,
    )

    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY is not configured")
        raise HTTPException(status_code=500, detail="Server is missing API key configuration.")
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="'text' field must not be empty.")

    try:
        result = await interpret_command(
            text=body.text,
            now=body.now,
            timezone=body.timezone,
            locale=body.locale,
            active_task=body.active_task.model_dump(exclude_none=True) if body.active_task else None,
            candidate_tasks=[task.model_dump(exclude_none=True) for task in body.candidate_tasks],
            request_id=body.request_id or correlation.request_id,
            command_session_id=correlation.command_session_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Command interpretation failed: %s", e)
        raise HTTPException(status_code=502, detail="Upstream command interpretation service error.")

    span.log(
        candidateTaskCount=len(body.candidate_tasks),
        totalInterpretMs=f"{span.elapsed_ms():.1f}",
        model=OPENAI_INTERPRET_MODEL,
    )
    return result
