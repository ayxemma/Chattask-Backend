import logging

from fastapi import FastAPI

from app.routes import transcribe, parse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI(
    title="ChatTask Backend",
    description="Server-side API for ChatTask: transcription and task parsing via OpenAI.",
    version="0.1.0",
)

app.include_router(transcribe.router)
app.include_router(parse.router)


@app.get("/", tags=["health"])
async def health():
    return {"status": "ok"}
