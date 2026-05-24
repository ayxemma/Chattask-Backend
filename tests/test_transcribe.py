"""Tests for POST /transcribe."""
import io
from unittest.mock import AsyncMock, patch

import pytest
from starlette.testclient import TestClient

FAKE_AUDIO = b"\xff\xfb\x90\x00" + b"\x00" * 128  # minimal non-empty bytes
TRANSCRIPT = "pick up Ari at 6:15"


def _audio_upload(content: bytes = FAKE_AUDIO, filename: str = "test.m4a", mime: str = "audio/x-m4a"):
    """Helper that returns a files dict suitable for TestClient."""
    return {"file": (filename, io.BytesIO(content), mime)}


# ---------------------------------------------------------------------------
# Happy-path
# ---------------------------------------------------------------------------

def test_transcribe_returns_text(client: TestClient):
    """Mocks transcribe_audio and verifies the response shape."""
    with patch(
        "app.routes.transcribe.transcribe_audio",
        new_callable=AsyncMock,
        return_value=(TRANSCRIPT, 0),
    ):
        response = client.post("/transcribe", files=_audio_upload())

    assert response.status_code == 200
    assert response.json() == {"text": TRANSCRIPT}


def test_transcribe_calls_service_with_bytes(client: TestClient):
    """Verifies bytes, filename, and content-type are forwarded to the service."""
    mock_fn = AsyncMock(return_value=(TRANSCRIPT, 0))
    with patch("app.routes.transcribe.transcribe_audio", new=mock_fn):
        client.post("/transcribe", files=_audio_upload(filename="voice.m4a", mime="audio/x-m4a"))

    mock_fn.assert_awaited_once()
    args = mock_fn.call_args[0]
    assert isinstance(args[0], bytes) and len(args[0]) > 0  # file_bytes
    assert args[1] == "voice.m4a"                            # filename
    assert "audio" in args[2]                                # content_type


def test_transcribe_accepts_wav(client: TestClient):
    with patch(
        "app.routes.transcribe.transcribe_audio",
        new_callable=AsyncMock,
        return_value=("hello", 0),
    ):
        response = client.post("/transcribe", files=_audio_upload(mime="audio/wav", filename="clip.wav"))
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

def test_transcribe_returns_422_when_no_file_provided(client: TestClient):
    """FastAPI returns 422 when the required 'file' field is absent."""
    response = client.post("/transcribe")
    assert response.status_code == 422


def test_transcribe_returns_400_for_empty_file(client: TestClient):
    with patch("app.routes.transcribe.transcribe_audio", new_callable=AsyncMock):
        response = client.post("/transcribe", files=_audio_upload(content=b""))
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_transcribe_returns_415_for_wrong_content_type(client: TestClient):
    with patch("app.routes.transcribe.transcribe_audio", new_callable=AsyncMock):
        response = client.post(
            "/transcribe",
            files=_audio_upload(mime="text/plain", filename="note.txt"),
        )
    assert response.status_code == 415


# ---------------------------------------------------------------------------
# Service / upstream errors
# ---------------------------------------------------------------------------

def test_transcribe_returns_500_on_value_error(client: TestClient):
    """ValueError from the service (e.g. bad API key at call time) → 500."""
    with patch(
        "app.routes.transcribe.transcribe_audio",
        new_callable=AsyncMock,
        side_effect=ValueError("OPENAI_API_KEY is not set"),
    ):
        response = client.post("/transcribe", files=_audio_upload())
    assert response.status_code == 500


def test_transcribe_returns_502_on_upstream_error(client: TestClient):
    """Generic network/HTTP exception from the service → 502."""
    with patch(
        "app.routes.transcribe.transcribe_audio",
        new_callable=AsyncMock,
        side_effect=Exception("connection reset"),
    ):
        response = client.post("/transcribe", files=_audio_upload())
    assert response.status_code == 502


# ---------------------------------------------------------------------------
# Missing API key
# ---------------------------------------------------------------------------

def test_transcribe_returns_500_when_api_key_missing(client: TestClient):
    with patch("app.routes.transcribe.OPENAI_API_KEY", None):
        response = client.post("/transcribe", files=_audio_upload())
    assert response.status_code == 500
    assert "API key" in response.json()["detail"]
