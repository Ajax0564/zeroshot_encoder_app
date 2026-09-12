import asyncio
import torch
from gliner2 import AutoExtractor
from typing import Annotated
from pydantic import BaseModel, Field, field_validator
from ..config import MODEL_NAME, LOCAL_DIRECTORY,CACHE_TTL,REDIS_URL
from ..cache.redis_cache import RedisCache


ClassificationLabels = list[str] | dict[str, str]

class ClassificationRequest(BaseModel):
    texts: str | list[str]
    labels: ClassificationLabels
    task_name: str
    is_multilabel: bool = False
    include_confidence: bool = True

    model_config = {
        "json_schema_extra": {
            "example": {
                "texts": [
                    "The phone has an excellent camera and great battery life.",
                    "The phone is expensive but the camera is amazing.",
                    "The delivery was late and the packaging was damaged."
                ],
                "labels": [
                    "camera",
                    "battery",
                    "price",
                    "delivery",
                    "packaging"
                ],
                "task_name": "product_features",
                "is_multilabel": True,
                "include_confidence": True
            }
        }
    }

    @field_validator("labels")
    @classmethod
    def validate_labels(cls, value):
        if not value:
            raise ValueError("labels cannot be empty")

        if isinstance(value, list):
            if not all(isinstance(label, str) and label.strip() for label in value):
                raise ValueError("labels must contain non-empty strings")

        elif isinstance(value, dict):
            if not all(
                isinstance(label, str)
                and label.strip()
                and isinstance(description, str)
                and description.strip()
                for label, description in value.items()
            ):
                raise ValueError(
                    "label descriptions must be dict[str, str] with non-empty values"
                )

        return value

class EntityRequest(BaseModel):
    texts: str | list[str]
    labels: ClassificationLabels
    task_name: str
    include_confidence: bool = True
    include_spans: bool=True
    threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5

    model_config = {
        "json_schema_extra": {
            "example": {
                "texts": [
                    "Apple released the new iPhone in California.",
                    "Microsoft is headquartered in Redmond, Washington."
                ],
                "labels": [
                    "company",
                    "product",
                    "location"
                ],
                "task_name": "named_entities",
                "include_confidence": True,
                "threshold": 0.5
            }
        }
    }

    @field_validator("labels")
    @classmethod
    def validate_labels(cls, value):
        if not value:
            raise ValueError("labels cannot be empty")

        if isinstance(value, list):
            if not all(isinstance(label, str) and label.strip() for label in value):
                raise ValueError("labels must contain non-empty strings")

        elif isinstance(value, dict):
            if not all(
                isinstance(label, str)
                and label.strip()
                and isinstance(description, str)
                and description.strip()
                for label, description in value.items()
            ):
                raise ValueError(
                    "label descriptions must be dict[str, str] with non-empty values"
                )

        return value


def get_device() -> str:
    """Use CUDA when a compatible GPU is available, otherwise use the CPU."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_extractor():
    LOCAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    device = get_device()

    print(f"Model directory: {LOCAL_DIRECTORY}")
    print(f"Model directory contents: {list(LOCAL_DIRECTORY.iterdir())}")
    print(f"Using device: {device}", flush=True)

    if any(LOCAL_DIRECTORY.iterdir()):
        print(f"Loading local model: {LOCAL_DIRECTORY}", flush=True)

        extractor = AutoExtractor.from_pretrained(
            str(LOCAL_DIRECTORY),
            map_location=device,
        )

        print("Local model loaded!", flush=True)

        return extractor

    print(f"Downloading: {MODEL_NAME}", flush=True)

    extractor = AutoExtractor.from_pretrained(
        MODEL_NAME,
        map_location=device,
    )

    print("Download/load complete!", flush=True)

    print(f"Saving model to: {LOCAL_DIRECTORY}", flush=True)

    extractor.save_pretrained(
        str(LOCAL_DIRECTORY)
    )

    print("Model saved!", flush=True)

    return extractor


class GLiNERModel:
    def __init__(self,max_concurrent_requests: int = 4):
        self.extractor = load_extractor()
        self.inference_semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.cache = RedisCache(
            redis_url=REDIS_URL,
            ttl=CACHE_TTL,
        )

    async def classify(self, request: ClassificationRequest) -> dict:
        cache_key = self.cache.make_key("classification",request.model_dump())
        cached_result = await self.cache.get(
        cache_key
    )

        if cached_result is not None:

            print(
                f"Classification CACHE HIT: {cache_key}",
                flush=True
            )

            return cached_result

        print(
            f"Classification CACHE MISS: {cache_key}",
            flush=True
        )


        schema = self.extractor.create_schema().classification(
                request.task_name, request.labels,multi_label=request.is_multilabel, include_confidence=request.include_confidence
            )

        # Normalize texts to list format for extract
        texts_input = [request.texts] if isinstance(request.texts, str) else request.texts

        # Run blocking inference in a background thread
        async with self.inference_semaphore:
            results = await asyncio.to_thread(
                self.extractor.batch_extract,texts_input, schema,include_confidence=request.include_confidence)
            
        response = {
        "results": [
            {
                "text": text,
                "classification": result,
            }
            for text, result in zip(texts_input, results)
        ] }
        await self.cache.set(
        cache_key,
        response )

        return response

    async def entities(self, request: EntityRequest) -> dict:
        cache_key = self.cache.make_key("entities",request.model_dump())
        cached_result = await self.cache.get(
        cache_key)

        if cached_result is not None:

            print(
                f"Entity CACHE HIT: {cache_key}",
                flush=True
            )

            return cached_result

        print(
            f"Entity CACHE MISS: {cache_key}",
            flush=True
        )

        texts_input = [request.texts] if isinstance(request.texts, str) else request.texts

        async with self.inference_semaphore:
            results = await asyncio.to_thread(
                self.extractor.batch_extract_entities,
                texts_input,request.labels,
                include_confidence=request.include_confidence,
                threshold=request.threshold,
                include_spans=request.include_spans
            )

        response = {
        "results": [
            {
                "text": text,
                "entities": result,
            }
            for text, result in zip(texts_input, results)
        ] }
        await self.cache.set(cache_key,response)

        return response