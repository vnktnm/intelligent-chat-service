from typing import Optional
import motor
import motor.motor_asyncio
import config
from utils import logger
from pymongo.errors import ConnectionFailure, ConfigurationError, OperationFailure


class MongoDBManager:
    """Utility class to connect to MongoDB"""

    def __init__(
        self,
        db_name: Optional[str] = config.MONGO_DATABASE_NAME,
        collection_name: str = None,
    ):
        # Get MongoDB configuration from environment variables
        mongo_user = config.MONGO_USER
        mongo_password = config.MONGO_PASSWORD
        mongo_host = config.MONGO_HOST
        mongo_port = config.MONGO_PORT
        mongo_auth_source = "admin"  # Using admin as the auth source

        # Build connection string similar to tool_manager.py
        self.mongodb_uri = f"mongodb://{mongo_user}:{mongo_password}@{mongo_host}:{mongo_port}/?authSource={mongo_auth_source}"

        # Only add authMechanism if specified and not NONE
        if config.MONGO_AUTH_MECH and config.MONGO_AUTH_MECH.upper() != "NONE":
            self.mongodb_uri += f"&authMechanism={config.MONGO_AUTH_MECH}"

        # Add replica set if configured
        if config.MONGO_REPLICA_SET:
            self.mongodb_uri += f"&replicaSet={config.MONGO_REPLICA_SET}"

        # Add read preference if configured
        if config.MONGO_READ_PREFERENCE:
            self.mongodb_uri += f"&readPreference={config.MONGO_READ_PREFERENCE}"

        self.db_name = db_name
        self.collection_name = collection_name
        self._mongo_client = None
        self._mongo_db = None
        self._collection = None

        # Log connection attempt with masked URI (no credentials)
        connection_info = self.mongodb_uri.split("@")[-1]
        logger.debug(f"MongoDB connecting to: {connection_info}")

    @property
    def mongo_client(self):
        if self._mongo_client is None and self.mongodb_uri:
            try:
                self._mongo_client = motor.motor_asyncio.AsyncIOMotorClient(
                    self.mongodb_uri
                )
                # Test connection - this is async so we can't directly check server_info
                logger.info(
                    f"MongoDB connection initialized to {config.MONGO_HOST}:{config.MONGO_PORT}"
                )
            except ConnectionFailure:
                logger.error(
                    "Failed to connect to MongoDB. Check if the server is running."
                )
                self._mongo_client = None
            except ConfigurationError as e:
                logger.error(f"MongoDB configuration error: {str(e)}")
                self._mongo_client = None
            except OperationFailure as e:
                logger.error(f"MongoDB operation failed: {str(e)}")
                if "requires authentication" in str(e):
                    logger.error(
                        "Authentication failed. Please check your MongoDB credentials."
                    )
                self._mongo_client = None
            except Exception as e:
                logger.error(f"Failed to connect to MongoDB: {str(e)}")
                self._mongo_client = None

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
