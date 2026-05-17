import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from app.config import cors_allow_origins, validate_config
from app.routes import interpret_command, parse, resolve_task_target, transcribe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    validate_config()
    logger.info("Configuration validated; OPENAI_API_KEY is set.")
    yield


app = FastAPI(
    title="ChatTask Backend",
    description="Server-side API for ChatTask: transcription and task parsing via OpenAI.",
    version="0.1.0",
    lifespan=lifespan,
)

_origins = cors_allow_origins()
if _origins:
    logger.info("CORS enabled for %d origin(s).", len(_origins))
else:
    logger.info(
        "CORS_ORIGIN not set; browser cross-origin requests are not allowed. "
        "Set CORS_ORIGIN to a comma-separated list of allowed origins for production web clients."
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(transcribe.router)
app.include_router(parse.router)
app.include_router(resolve_task_target.router)
app.include_router(interpret_command.router)


@app.get("/health", tags=["health"])
async def health():
    return {"status": "ok"}


@app.get("/", tags=["health"])
async def root_health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    from app.config import listen_port

    port = listen_port()
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )
