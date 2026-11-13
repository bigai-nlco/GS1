"""Serve local GS1 inference using the TypeSafe wire format."""

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse

from gs1.api.local import BackendError
from gs1.api.memory import MemoryBudgetBackend
from gs1.api.settings import Settings
from gs1.model.schema import DecisionRequest, DecisionResponse


def create_app(settings=None, *, backend=None):
    settings = settings or Settings()
    backend = backend if backend is not None else MemoryBudgetBackend(settings)

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
        content = {"message": str(error)}
        if error.details is not None:
            content["details"] = error.details
        return JSONResponse(status_code=error.status, content={"error": content}, headers=headers)

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
        stats = getattr(result, "_gs1_stats", {})
        response.headers["X-GS1-Graph-Hits"] = str(stats.get("graph", {}).get("hits", 0))
        response.headers["X-GS1-Graph-Misses"] = str(stats.get("graph", {}).get("misses", 0))
        response.headers["X-GS1-Profile-Fallback"] = str(stats.get("runtime_profile_fallback", False)).lower()
        return result

    @app.post("/v1/admission")
    async def admission(body: DecisionRequest):
        if body.model not in {settings.local_model, backend.metadata.get("resolved_model")}:
            raise BackendError(f"Unknown model: {body.model}", 422)
        return await backend.admission(body)

    @app.get("/v1/runtime")
    async def runtime():
        profile = getattr(backend, "profile", None)
        return {
            "available": backend.available,
            "reason": backend.reason,
            "engine": settings.engine,
            "precision": settings.precision,
            "model": settings.local_model,
            "resolved_model": backend.metadata.get("resolved_model"),
            "cuda_graphs_enabled": settings.cuda_graphs,
            "capture_policy": settings.graph_capture_policy,
            "arithmetic": profile.data["arithmetic"] if profile else "native",
            "qualified_geometries": len(profile.data["signatures"]) if profile else 0,
            "memory_budget": dict(getattr(backend, "memory_budget", {})),
            "last_admission": dict(getattr(backend, "last_admission", {})),
            "slots": [
                {
                    "last_request": dict(engine.last_stats),
                    "memory_budget": dict(getattr(backend, "slot_budgets", {}).get(index, {})),
                    "graph_statistics": dict(engine.graph_cache.stats) if engine.graph_cache else {},
                    "graph_cache_mib": engine.graph_cache.bytes / 1024**2 if engine.graph_cache else 0,
                    "resident_graphs": len(engine.graph_cache.entries) if engine.graph_cache else 0,
                }
                for index, engine in enumerate(backend.engines)
            ],
        }

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
