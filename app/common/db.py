import asyncio
import logging
from pathlib import Path

from beanie import init_beanie, Link
from motor.core import AgnosticDatabase
from motor.motor_asyncio import AsyncIOMotorClient

logger = logging.getLogger(__name__)


async def _get_mongo_client(settings):
    """
    separate so it could be decorated, the reason were not also initiating in this function
    """
    logger.info('attempting to connect to mongodb')
    if settings.SSL_CA_CERTS:
        client = AsyncIOMotorClient(
            settings.MONGO_FULL_URL,
            tlsCertificateKeyFile=str(Path(__file__).parent / 'X509-cert-6661849979066861196.pem'),
            tls=True,
            uuidRepresentation="standard"
        )
    else:
        client: AsyncIOMotorClient = AsyncIOMotorClient(settings.MONGO_FULL_URL, uuidRepresentation="standard")

    await client.admin.command({'ping': 1})
    return client


async def init_db(
        settings, documents, allow_index_dropping: bool = False
) -> tuple[list['Document'], 'AgnosticDatabase']:
    client = await _get_mongo_client(settings=settings)
    db: AgnosticDatabase = client[settings.DATABASE_NAME]
    db.get_io_loop = asyncio.get_event_loop
    await init_beanie(database=db, document_models=documents, allow_index_dropping=allow_index_dropping)
    logger.info('successfully connected to mongo')
    return documents, db
