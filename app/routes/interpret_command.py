import logging

from fastapi import APIRouter, HTTPException

from app.config import OPENAI_API_KEY
from app.models.parse_models import CommandInterpretRequest, CommandInterpretResponse
from app.services.openai_service import interpret_command

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/interpret-command", response_model=CommandInterpretResponse)
async def interpret_command_route(request: CommandInterpretRequest):
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY is not configured")
        raise HTTPException(status_code=500, detail="Server is missing API key configuration.")
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="'text' field must not be empty.")

    try:
        return await interpret_command(
            text=request.text,
            now=request.now,
            timezone=request.timezone,
            locale=request.locale,
            active_task=request.active_task.model_dump(exclude_none=True) if request.active_task else None,
            candidate_tasks=[task.model_dump(exclude_none=True) for task in request.candidate_tasks],
            request_id=request.request_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Command interpretation failed: %s", e)
        raise HTTPException(status_code=502, detail="Upstream command interpretation service error.")
