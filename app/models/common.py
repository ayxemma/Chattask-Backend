from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str


class TranscribeResponse(BaseModel):
    text: str


class ErrorResponse(BaseModel):
    detail: str
