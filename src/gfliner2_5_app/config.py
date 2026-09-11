from pathlib import Path
import os

MODEL_NAME: str = "fastino/gliner2.5-base-v1"
LOCAL_DIRECTORY: Path = Path(__file__).parent / "model" / MODEL_NAME.replace("/", "_")

REDIS_URL = os.getenv(
    "REDIS_URL",
    "redis://localhost:6379/0"
)

CACHE_TTL = int(
    os.getenv(
        "CACHE_TTL",
        "3600"
    )
)