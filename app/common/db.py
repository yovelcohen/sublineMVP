import asyncio
import logging
from pathlib import Path

from beanie import init_beanie
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
            "mongodb+srv://cluster1.bb0fppm.mongodb.net/?authSource=%24external&authMechanism=MONGODB-X509&retryWrites=true&w=majority",
            tls=True,
            tlsCertificateKeyFile=str(Path(__file__).parent / 'X509-cert-3256472255189972074.pem'),
            uuidRepresentation="standard"
        )
    else:
        client: AsyncIOMotorClient = AsyncIOMotorClient(settings.MONGO_FULL_URL, uuidRepresentation="standard")

    await client.admin.command({'ping': 1})
    return client


async def init_db(settings, documents) -> tuple[list['Document'], 'AgnosticDatabase']:
    client = await _get_mongo_client(settings=settings)
    db: AgnosticDatabase = client[settings.DATABASE_NAME]
    db.get_io_loop = asyncio.get_event_loop
    await init_beanie(database=db, document_models=documents)
    logger.info('successfully connected to mongo')
    return documents, db

# async def geta():
#     await init_db(settings, [User, Client])
#     cl = Client(name='demo', slug='demo')
#     await cl.save()
#     acc = UserAccess()
#     acc.set(UserAccess.Levels.ADMIN)
#     r = await User.create_user(first_name='yovel', last_name='cohen',
#                                email='yovell04@gmail.com', username='yovelcohen', password='huckfvi1', client=cl,
#                                access=acc)
#     return r
