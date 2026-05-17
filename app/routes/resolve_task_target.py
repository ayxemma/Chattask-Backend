import logging

from fastapi import APIRouter, HTTPException

from app.config import OPENAI_API_KEY
from app.models.parse_models import TaskTargetResolveRequest, TaskTargetResolveResponse
from app.services.openai_service import resolve_task_target

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/resolve-task-target", response_model=TaskTargetResolveResponse)
async def resolve_task_target_route(request: TaskTargetResolveRequest):
    """Resolve which provided candidate task an edit command refers to."""
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY is not configured")
        raise HTTPException(status_code=500, detail="Server is missing API key configuration.")

    if not request.user_text.strip():
        raise HTTPException(status_code=400, detail="'user_text' field must not be empty.")

    if not request.candidates:
        return TaskTargetResolveResponse(resolution="no_match", confidence=0.0, reason="no candidates")

    try:
        return await resolve_task_target(
            user_text=request.user_text,
            action_type=request.action_type,
            target_title=request.target_title,
            target_time=request.target_time,
            candidates=[candidate.model_dump(exclude_none=True) for candidate in request.candidates],
            active_task_id=request.active_task_id,
            timezone=request.timezone,
            locale=request.locale,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Task target resolution failed: %s", e)
        raise HTTPException(status_code=502, detail="Upstream task target resolution service error.")
