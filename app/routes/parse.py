import logging

from fastapi import APIRouter, HTTPException, Request

from app.config import OPENAI_API_KEY, OPENAI_PARSE_MODEL
from app.models.parse_models import ParseRequest, ParseResponse
from app.services.openai_service import parse_task_text
from app.util.request_timing import TimingSpan, correlation_from_headers

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/parse", response_model=ParseResponse)
async def parse(request: Request, body: ParseRequest):
    """
    Accept a task text plus current time context and return structured task data via OpenAI.
    """
    correlation = correlation_from_headers(dict(request.headers))
    if body.command_session_id and not correlation.command_session_id:
        correlation.command_session_id = body.command_session_id
    span = TimingSpan(label="parse", correlation=correlation)

    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY is not configured")
        raise HTTPException(status_code=500, detail="Server is missing API key configuration.")

    if not body.text.strip():
        raise HTTPException(status_code=400, detail="'text' field must not be empty.")

    logger.info(
        "parse requestReceived request_id=%s command_session_id=%s likelyColdStart=%s model=%s",
        body.request_id or correlation.request_id,
        correlation.command_session_id,
        correlation.likely_cold_start,
        OPENAI_PARSE_MODEL,
    )

    try:
        result = await parse_task_text(
            text=body.text,
            now=body.now,
            timezone=body.timezone,
            locale=body.locale,
            parse_instructions=body.parse_instructions,
            source=body.source,
            last_active_task_id=body.last_active_task_id,
            active_task_title=body.active_task_title,
            active_task_scheduled_at=body.active_task_scheduled_at,
            active_task_notes=body.active_task_notes,
            active_task_recurrence=(
                body.active_task_recurrence.model_dump(exclude_none=True)
                if body.active_task_recurrence
                else None
            ),
            request_id=body.request_id or correlation.request_id,
            command_session_id=correlation.command_session_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Parse failed: %s", e)
        raise HTTPException(status_code=502, detail="Upstream parse service error.")

    span.log(totalInterpretMs=f"{span.elapsed_ms():.1f}", model=OPENAI_PARSE_MODEL)
    return result
