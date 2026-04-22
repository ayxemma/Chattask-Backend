import logging

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config import OPENAI_API_KEY
from app.models.common import TranscribeResponse
from app.services.openai_service import transcribe_audio

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(file: UploadFile = File(...)):
    """
    Accept an audio file upload and return a transcript via OpenAI.
    """
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY is not configured")
        raise HTTPException(status_code=500, detail="Server is missing API key configuration.")

    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No audio file provided.")

    content_type = file.content_type or "application/octet-stream"

    # Allow any audio/* type plus common fallbacks from mobile clients
    if not (content_type.startswith("audio/") or content_type == "application/octet-stream"):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type '{content_type}'. Please upload an audio file.",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        text = await transcribe_audio(file_bytes, file.filename, content_type)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("Transcription failed: %s", e)
        raise HTTPException(status_code=502, detail="Upstream transcription service error.")

    return TranscribeResponse(text=text)
