from typing import Optional, Dict, Any, List
import motor.motor_asyncio
import config
from utils import logger


class MongoDBManager:
    """Utility class to connect to Mongodb"""

    def __init__(
        self,
        db_name: Optional[str] = config.MONGO_DATABASE_NAME,
        collection_name: str = None,
    ):
        # Simple connection string with just username and password
        if config.MONGO_USER and config.MONGO_PASSWORD:
            self.mongodb_uri = f"mongodb://{config.MONGO_USER}:{config.MONGO_PASSWORD}@{config.MONGO_HOST}:{config.MONGO_PORT}/{db_name}?authSource=admin"
        else:
            self.mongodb_uri = (
                f"mongodb://{config.MONGO_HOST}:{config.MONGO_PORT}/{db_name}"
            )

        logger.info(f"Connecting to MongoDB at {config.MONGO_HOST}:{config.MONGO_PORT}")

        self.db_name = db_name
        self.collection_name = collection_name
        self._mongo_client = None
        self._mongo_db = None
        self._collection = None

    @property
    def mongo_client(self):
        if self._mongo_client is None and self.mongodb_uri:
            try:
                self._mongo_client = motor.motor_asyncio.AsyncIOMotorClient(
                    self.mongodb_uri
                )
                logger.info("Successfully connected to MongoDB")
            except Exception as e:
                logger.error(f"MongoDB connection error: {str(e)}")

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

    async def insert(self, collection_name: str, document: Dict[str, Any]) -> str:
        """Insert a document into the specified collection"""
        collection = self.mongo_db[collection_name]
        result = await collection.insert_one(document)
        return str(result.inserted_id)

    async def find_one(
        self, collection_name: str, filter_query: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Find a single document that matches the filter"""
        collection = self.mongo_db[collection_name]
        document = await collection.find_one(filter_query)
        return document

    async def find(
        self,
        collection_name: str,
        filter_query: Dict[str, Any],
        skip: int = 0,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Find documents that match the filter"""
        collection = self.mongo_db[collection_name]
        cursor = collection.find(filter_query)

        if skip > 0:
            cursor = cursor.skip(skip)

        if limit > 0:
            cursor = cursor.limit(limit)

        documents = await cursor.to_list(length=limit)
        return documents

    async def update_one(
        self,
        collection_name: str,
        filter_query: Dict[str, Any],
        update_data: Dict[str, Any],
    ) -> int:
        """Update a single document that matches the filter"""
        collection = self.mongo_db[collection_name]
        result = await collection.update_one(filter_query, update_data)
        return result.modified_count

    async def delete_one(
        self, collection_name: str, filter_query: Dict[str, Any]
    ) -> int:
        """Delete a single document that matches the filter"""
        collection = self.mongo_db[collection_name]
        result = await collection.delete_one(filter_query)
        return result.deleted_count

    async def count(self, collection_name: str, filter_query: Dict[str, Any]) -> int:
        """Count documents that match the filter"""
        collection = self.mongo_db[collection_name]
        count = await collection.count_documents(filter_query)
        return count


async def initialize_mongodb_indexes():
    """
    Create indexes for MongoDB collections.
    Call this at application startup.
    """
    try:
        # Get the tools collection
        db_client = MongoDBManager().mongo_client
        collection = db_client[config.MONGO_DATABASE_NAME][
            config.MONGO_TOOL_COLLECTION_NAME
        ]

        # Create indexes
        await collection.create_index("id", unique=True)
        await collection.create_index("name")
        await collection.create_index("category")
        await collection.create_index("tags")
        await collection.create_index("enabled")

        logger.info(
            f"MongoDB indexes created for collection: {config.MONGO_TOOL_COLLECTION_NAME}"
        )
    except Exception as e:
        logger.error(f"Failed to create MongoDB indexes: {str(e)}")
