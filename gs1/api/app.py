"""Serve local GS1 inference using the TypeSafe wire format."""

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse

from gs1.api.local import BackendError, LocalBackend
from gs1.api.settings import Settings
from gs1.model.schema import DecisionRequest, DecisionResponse


def create_app(settings=None):
    settings = settings or Settings()
    backend = LocalBackend(settings)

    @asynccontextmanager
    async def lifespan(app):
        try:
            await backend.start()
            yield
        finally:
            await backend.close()

    app = FastAPI(title="GS1", version="0.1.0", lifespan=lifespan)
    app.state.backend = backend

    @app.exception_handler(BackendError)
    async def backend_error(request, error):
        headers = {"Retry-After": "1"} if error.status == 529 else None
        return JSONResponse(
            status_code=error.status, content={"error": {"message": str(error)}}, headers=headers
        )

    @app.exception_handler(ValueError)
    async def value_error(request, error):
        return JSONResponse(status_code=422, content={"detail": [{"msg": str(error)}]})

    @app.post("/v1/systemone", response_model=DecisionResponse)
    async def system_one(body: DecisionRequest, response: Response):
        if body.model not in {settings.local_model, backend.metadata.get("resolved_model")}:
            raise BackendError(f"Unknown model: {body.model}", 422)
        started = time.perf_counter()
        result = await backend.evaluate(body)
        response.headers["X-Request-ID"] = uuid.uuid4().hex
        response.headers["Server-Timing"] = f"total;dur={(time.perf_counter() - started) * 1000:.3f}"
        return result

    @app.get("/v1/models")
    async def models():
        return {
            "models": [
                dict(
                    name=settings.local_model,
                    description="GS1 native decision model",
                    release_date=getattr(backend.loaded.model.config, "release_date", None) or "",
                )
            ]
            if backend.available
            else []
        }

    return app
