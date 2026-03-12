from motor.motor_asyncio import AsyncIOMotorClient
from elasticsearch import AsyncElasticsearch

from orchestrator.core.config import settings
from orchestrator.utils.logging import get_logger

logger = get_logger(__name__)

# Module-level clients — initialised on startup
_mongo_client: AsyncIOMotorClient | None = None
_es_client: AsyncElasticsearch | None = None


async def connect_db() -> None:
    global _mongo_client, _es_client

    _mongo_client = AsyncIOMotorClient(settings.MONGO_URI)
    await _mongo_client.admin.command("ping")
    logger.info("mongodb_connected", uri=settings.MONGO_URI)

    _es_client = AsyncElasticsearch(settings.ES_HOST)
    await _es_client.info()
    logger.info("elasticsearch_connected", host=settings.ES_HOST)


async def disconnect_db() -> None:
    if _mongo_client:
        _mongo_client.close()
    if _es_client:
        await _es_client.close()
    logger.info("databases_disconnected")


def get_mongo_db():
    if not _mongo_client:
        raise RuntimeError("MongoDB not initialised — call connect_db() first")
    return _mongo_client[settings.MONGO_DB]


def get_es_client() -> AsyncElasticsearch:
    if not _es_client:
        raise RuntimeError("Elasticsearch not initialised — call connect_db() first")
    return _es_client
