import logging

from fastapi import APIRouter, HTTPException

from app.config import OPENAI_API_KEY
from app.models.parse_models import ParseRequest, ParseResponse
from app.services.task_parser_service import parse_task_text

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/parse", response_model=ParseResponse)
async def parse(request: ParseRequest):
    """
    Accept a task text plus current time context and return structured task data via OpenAI.
    """
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY is not configured")
        raise HTTPException(status_code=500, detail="Server is missing API key configuration.")

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="'text' field must not be empty.")

    if request.request_id:
        logger.info("parse request_id=%s", request.request_id)

    try:
        result = await parse_task_text(
            text=request.text,
            now=request.now,
            timezone=request.timezone,
            locale=request.locale,
            parse_instructions=request.parse_instructions,
            source=request.source,
            last_active_task_id=request.last_active_task_id,
            active_task_title=request.active_task_title,
            active_task_scheduled_at=request.active_task_scheduled_at,
            active_task_notes=request.active_task_notes,
            active_task_recurrence=(
                request.active_task_recurrence.model_dump(exclude_none=True)
                if request.active_task_recurrence
                else None
            ),
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Parse failed: %s", e)
        raise HTTPException(status_code=502, detail="Upstream parse service error.")

    return result
