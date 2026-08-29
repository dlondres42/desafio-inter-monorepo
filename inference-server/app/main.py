"""The inference service.

Loads exactly one model bundle at startup and serves it. Its entire import from
``dolores-lib`` is ``dolores.inference`` — no training code, no MLflow, nothing
that would drag the training stack into a serving container.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from dolores.inference import load_model
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.api.routers import health, predict
from app.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the bundle once, before the first request.

    Deliberately without a try/except. A missing or incompatible artifact should
    kill the process so Kubernetes reports CrashLoopBackOff and the rollout halts
    with the previous version still serving — rather than starting a server that
    returns 500 for every request. The library validates the manifest and the
    runtime versions before unpickling anything, so a bad bundle fails here and
    not at the first prediction.
    """
    app.state.model = load_model(get_settings().model_path)
    yield


app = FastAPI(
    title="dolores inference server",
    description="Serves one dolores-lib model bundle, named by MODEL_PATH.",
    lifespan=lifespan,
)
app.include_router(health.router)
app.include_router(predict.router)


@app.get("/")
async def read_root() -> JSONResponse:
    return JSONResponse({"Hello": "World"})
