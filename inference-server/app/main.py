from fastapi import FastAPI, Response

from .api.routers import health

app = FastAPI()
app.include_router(health.router)


@app.get("/")
async def read_root() -> Response:
    response = Response(content={"Hello": "World"})
    return response
