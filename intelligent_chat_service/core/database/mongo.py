from typing import Optional
import motor
import motor.motor_asyncio
import config


class MongoDBManager:
    """Utility class to connect to Mongodb"""

    def __init__(
        self,
        db_name: Optional[str] = config.MONGO_DATABASE_NAME,
        collection_name: str = None,
    ):
        self.mongodb_uri = f"mongodb://{config.MONGO_USER}:{config.MONGO_PASSWORD}@{config.MONGO_HOST}:{config.MONGO_PORT}/?authMechanism={config.MONGO_AUTH_MECH}&authSource=admin"
        self.db_name = db_name
        self.collection_name = collection_name
        self._mongo_client = None
        self._mongo_db = None
        self._collection = None

    @property
    def mongo_client(self):
        if self._mongo_client is None and self.mongodb_uri:
            self._mongo_client = motor.motor_asyncio.AsyncIOMotorClient(
                self.mongodb_uri
            )

        return self._mongo_client

    @property
    def mongo_db(self):
        if self._mongo_db is None and self.mongo_client is not None:
            self._mongo_db = self.mongo_client[self.db_name]
        return self._mongo_db

    @property
    def collection(self):
        if (
            self._collection is None
            and self.mongo_db is not None
            and self.collection_name
        ):
            self._collection = self.mongo_db[self.collection_name]
        return self._collection

    def is_connected(self):
        return (
            self.mongo_client is not None
            and self.mongo_db is not None
            and self.collection is not None
        )

    def close(self):
        if self._mongo_client is not None:
            self._mongo_client.close()
            self._mongo_client = None
            self._mongo_db = None
            self._collection = None
