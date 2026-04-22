import logging

from fastapi import APIRouter, HTTPException

from app.config import OPENAI_API_KEY
from app.models.parse_models import ParseRequest, ParseResponse
from app.services.openai_service import parse_task_text

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

    try:
        result = await parse_task_text(request.text, request.now, request.timezone)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Parse failed: %s", e)
        raise HTTPException(status_code=502, detail="Upstream parse service error.")

    return result
