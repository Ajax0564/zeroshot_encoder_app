from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

from gliner_app.models.model import (
    GLiNERModel,
    ClassificationRequest,
    EntityRequest,
)


class ClassificationResult(BaseModel):
    text: str
    classification: Any


class ClassificationResponse(BaseModel):
    results: list[ClassificationResult]


class EntityResult(BaseModel):
    text: str
    entities: Any


class EntityResponse(BaseModel):
    results: list[EntityResult]


tags_metadata = [
    {
        "name": "Inference",
        "description": "Zero-shot text classification and entity extraction endpoints.",
    },
    {
        "name": "System & Cache",
        "description": "Health checks and Redis cache management.",
    },
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application startup and shutdown lifecycle.

    The GLiNER model is loaded once when the application starts.
    Redis/cache resources are cleaned up when the application shuts down.
    """
    print("Loading GLiNER2.5 model...", flush=True)

    app.state.model = GLiNERModel(
        max_concurrent_requests=4
    )

    print("GLiNER2.5 API is running...", flush=True)

    try:
        # Application runs while inside this context.
        yield

    finally:
        print("Shutting down GLiNER2.5 API...", flush=True)

        model = getattr(app.state, "model", None)

        if model is not None:
            cache = getattr(model, "cache", None)

            if cache is not None and hasattr(cache, "close"):
                await cache.close()

        print("GLiNER2.5 API shutdown complete.", flush=True)


app = FastAPI(
    title="GLiNER 2.5 Inference Service",
    description="""
    ### 🚀 GLiNER 2.5 API Service

    This API provides:

    * **Zero-Shot Classification** - Classify unstructured text using custom label schemas.
    * **Named Entity Recognition (NER)** - Extract entities dynamically with confidence scoring.
    * **Redis Caching Integration** - Automatic caching for repeated payloads.
    """,
    version="0.1.0",
    openapi_tags=tags_metadata,
    lifespan=lifespan,
    swagger_ui_parameters={
        "syntaxHighlight.theme": "obsidian",
        "defaultModelsExpandDepth": -1,
        "docExpansion": "list",
    },
)


@app.get("/", tags=["System & Cache"])
async def root():
    return {
        "status": "ok",
        "message": "GLiNER2.5 API is running",
    }


@app.delete("/cache", tags=["System & Cache"])
async def clear_cache():
    await app.state.model.cache.clear()

    return {
        "status": "ok",
        "message": "GLiNER cache cleared",
    }


@app.post(
    "/classify",
    response_model=ClassificationResponse,
    summary="Classify text",
    description="Classify one or more texts against the provided labels.",
    tags=["Inference"],
)
async def classify(
    request: ClassificationRequest,
) -> ClassificationResponse:
    return await app.state.model.classify(request)


@app.post(
    "/entities",
    response_model=EntityResponse,
    summary="Extract entities",
    description="Extract entities from one or more texts.",
    tags=["Inference"],
)
async def entities(
    request: EntityRequest,
) -> EntityResponse:
    return await app.state.model.entities(request)
