import json
import hashlib
import redis.asyncio as redis

# 3600 seconds = 1 hour
class RedisCache:

    def __init__(
        self,
        redis_url: str,
        ttl: int = 3600
    ):
        self.redis = redis.from_url(
            redis_url,
            encoding="utf-8",
            decode_responses=True,
        )

        self.ttl = ttl

    def make_key(
        self,
        prefix: str,
        data: dict
    ) -> str:

        serialized = json.dumps(
            data,
            sort_keys=True,
            ensure_ascii=False,
        )

        hashed = hashlib.sha256(
            serialized.encode("utf-8")
        ).hexdigest()

        return f"gliner:{prefix}:{hashed}"

    async def get(self, key: str):

        try:
            value = await self.redis.get(key)

            if value is None:
                return None

            return json.loads(value)

        except Exception as e:
            print(f"Redis GET failed: {e}")
            return None

    async def set(self, key: str, value):

        try:
            await self.redis.set(
                key,
                json.dumps(value, ensure_ascii=False),
                ex=self.ttl,
            )

        except Exception as e:
            print(f"Redis SET failed: {e}")
            return None

    async def delete(self, key: str):

        await self.redis.delete(key)

    async def clear(self):

        keys = []

        async for key in self.redis.scan_iter(
            match="gliner:*"
        ):
            keys.append(key)

        if keys:
            await self.redis.delete(*keys)

    async def close(self):

        await self.redis.aclose()
