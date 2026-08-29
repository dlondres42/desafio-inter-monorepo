from fastapi import FastAPI
from fastapi.responses import JSONResponse

from .api.routers import health

app = FastAPI()
app.include_router(health.router)


@app.get("/")
async def read_root() -> JSONResponse:
    return JSONResponse({"Hello": "World"})
