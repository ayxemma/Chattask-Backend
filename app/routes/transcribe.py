import logging
import time

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app.config import OPENAI_API_KEY, OPENAI_TRANSCRIBE_MODEL
from app.models.common import TranscribeResponse
from app.services.openai_service import transcribe_audio
from app.util.request_timing import TimingSpan, correlation_from_headers

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(request: Request, file: UploadFile = File(...)):
    """
    Accept an audio file upload and return a transcript via OpenAI.
    """
    correlation = correlation_from_headers(dict(request.headers))
    span = TimingSpan(label="transcribe", correlation=correlation)
    logger.info(
        "transcribe requestReceived request_id=%s command_session_id=%s likelyColdStart=%s model=%s",
        correlation.request_id,
        correlation.command_session_id,
        correlation.likely_cold_start,
        OPENAI_TRANSCRIBE_MODEL,
    )

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

    read_t0 = time.perf_counter()
    file_bytes = await file.read()
    audio_read_ms = (time.perf_counter() - read_t0) * 1000
    audio_size = len(file_bytes)
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        openai_t0 = time.perf_counter()
        logger.info(
            "transcribe openAITranscriptionStart request_id=%s audioSizeBytes=%s",
            correlation.request_id,
            audio_size,
        )
        text, openai_retry = await transcribe_audio(file_bytes, file.filename, content_type)
        openai_ms = (time.perf_counter() - openai_t0) * 1000
        span.openai_ms = openai_ms
        span.retry_count = openai_retry
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("Transcription failed: %s", e)
        raise HTTPException(status_code=502, detail="Upstream transcription service error.")

    serialize_t0 = time.perf_counter()
    response = TranscribeResponse(text=text)
    serialize_ms = (time.perf_counter() - serialize_t0) * 1000

    span.log(
        audioReadMs=f"{audio_read_ms:.1f}",
        audioSizeBytes=audio_size,
        openAITranscriptionMs=f"{openai_ms:.1f}",
        responseSerializationMs=f"{serialize_ms:.1f}",
        totalTranscribeMs=f"{span.elapsed_ms():.1f}",
        transcriptLength=len(text),
        model=OPENAI_TRANSCRIBE_MODEL,
    )
    return response
