import argparse
import os
import json
import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import OperationFailure, ConnectionFailure, ConfigurationError

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ToolArgumentType(str, Enum):
    """Types of arguments that can be passed to tools"""

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


class ToolArgument(BaseModel):
    """Model for tool arguments"""

    name: str
    type: ToolArgumentType
    description: str
    required: bool = False
    default: Optional[Any] = None
    enum: Optional[List[Any]] = None


class ToolInfo(BaseModel):
    """Model for tool information"""

    name: str
    description: str
    version: str
    arguments: List[ToolArgument] = Field(default_factory=list)
    capabilities: List[str] = Field(default_factory=list)
    service_url: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class ToolManager:
    """Manager for tool registration and updates in MongoDB"""

    def __init__(self, mongo_uri: Optional[str] = None):
        """Initialize the tool manager"""
        # Get MongoDB configuration from environment variables
        mongo_user = os.getenv("MONGO_USER", "admin")
        mongo_password = os.getenv("MONGO_PASSWORD", "password")
        mongo_host = os.getenv("MONGO_HOST", "localhost")
        mongo_port = os.getenv("MONGO_PORT", "27017")
        mongo_auth_source = os.getenv("MONGO_AUTH_SOURCE", "admin")

        # Build connection string if not provided
        if not mongo_uri:
            self.mongo_uri = f"mongodb://{mongo_user}:{mongo_password}@{mongo_host}:{mongo_port}/?authSource={mongo_auth_source}"
        else:
            self.mongo_uri = mongo_uri

        self.db_name = os.getenv("MONGODB_DB", "intelligent_chat")
        self.collection_name = os.getenv("MONGODB_TOOLS_COLLECTION", "tools")

        # Connection and initialization
        self.client = None
        self.db = None
        self.collection = None

        try:
            logger.info(f"Connecting to MongoDB: {self.mongo_uri.split('@')[-1]}")
            self.client = MongoClient(self.mongo_uri)

            # Test connection with serverInfo
            self.client.server_info()

            self.db: Database = self.client[self.db_name]
            self.collection: Collection = self.db[self.collection_name]

            logger.info(f"Connected to MongoDB successfully")
            logger.info(f"Using database: {self.db_name}")
            logger.info(f"Using collection: {self.collection_name}")

            # Create index on tool name for faster lookups
            try:
                self.collection.create_index("name", unique=True)
                logger.info("Created unique index on 'name' field")
            except OperationFailure as e:
                logger.warning(f"Could not create index: {str(e)}")
                if "requires authentication" in str(e):
                    logger.error(
                        "Authentication failed. Please check your MongoDB credentials."
                    )

        except ConnectionFailure:
            logger.error(
                "Failed to connect to MongoDB. Check if the server is running."
            )
            raise
        except ConfigurationError as e:
            logger.error(f"MongoDB configuration error: {str(e)}")
            raise
        except OperationFailure as e:
            logger.error(f"MongoDB operation failed: {str(e)}")
            if "requires authentication" in str(e):
                logger.error(
                    "Authentication failed. Please check your MongoDB credentials."
                )
            raise

    def register_tool(self, tool_info: ToolInfo) -> Dict[str, Any]:
        """
        Register a new tool or update an existing one

        Args:
            tool_info: The tool information to register

        Returns:
            The registered tool document
        """
        if self.collection is None:
            raise ValueError("MongoDB connection not established")

        # Set timestamps
        now = datetime.utcnow()
        if not tool_info.created_at:
            tool_info.created_at = now
        tool_info.updated_at = now

        # Convert to dict
        tool_dict = tool_info.model_dump()

        try:
            # Check if tool already exists
            existing_tool = self.collection.find_one({"name": tool_info.name})

            if existing_tool:
                # Update existing tool
                logger.info(f"Updating existing tool: {tool_info.name}")
                self.collection.update_one(
                    {"name": tool_info.name},
                    {"$set": tool_dict},
                )
            else:
                # Insert new tool
                logger.info(f"Registering new tool: {tool_info.name}")
                self.collection.insert_one(tool_dict)

            # Return the updated document
            return self.collection.find_one({"name": tool_info.name})
        except OperationFailure as e:
            logger.error(f"Failed to register tool: {str(e)}")
            raise

    def get_tool(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get tool information by name

        Args:
            tool_name: The name of the tool to get

        Returns:
            The tool document or None if not found
        """
        if self.collection is None:
            raise ValueError("MongoDB connection not established")

        try:
            return self.collection.find_one({"name": tool_name})
        except OperationFailure as e:
            logger.error(f"Failed to get tool: {str(e)}")
            raise

    def list_tools(self) -> List[Dict[str, Any]]:
        """
        List all registered tools

        Returns:
            List of all tool documents
        """
        if self.collection is None:
            raise ValueError("MongoDB connection not established")

        try:
            return list(self.collection.find({}))
        except OperationFailure as e:
            logger.error(f"Failed to list tools: {str(e)}")
            raise

    def delete_tool(self, tool_name: str) -> bool:
        """
        Delete a tool by name

        Args:
            tool_name: The name of the tool to delete

        Returns:
            True if the tool was deleted, False otherwise
        """
        if self.collection is None:
            raise ValueError("MongoDB connection not established")

        try:
            result = self.collection.delete_one({"name": tool_name})
            return result.deleted_count > 0
        except OperationFailure as e:
            logger.error(f"Failed to delete tool: {str(e)}")
            raise

    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            self.client = None
            self.db = None
            self.collection = None


def load_tool_from_json(file_path: str) -> ToolInfo:
    """
    Load tool information from a JSON file

    Args:
        file_path: Path to the JSON file

    Returns:
        Tool information
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    # Process arguments to convert to proper Enum values
    if "arguments" in data:
        for arg in data["arguments"]:
            if "type" in arg:
                arg["type"] = ToolArgumentType(arg["type"])

    return ToolInfo(**data)


def main():
    parser = argparse.ArgumentParser(
        description="Tool Manager for registering and updating tools in MongoDB"
    )

    # Add MongoDB connection options
    parser.add_argument(
        "--mongo-uri",
        help="MongoDB URI (overrides environment variables)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Register command
    register_parser = subparsers.add_parser(
        "register", help="Register a tool from a JSON file"
    )
    register_parser.add_argument(
        "file_path", help="Path to JSON file with tool information"
    )

    # List command
    list_parser = subparsers.add_parser("list", help="List all registered tools")

    # Get command
    get_parser = subparsers.add_parser("get", help="Get a tool by name")
    get_parser.add_argument("name", help="Name of the tool to get")

    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a tool by name")
    delete_parser.add_argument("name", help="Name of the tool to delete")

    args = parser.parse_args()

    # Initialize tool manager
    try:
        tool_manager = ToolManager(mongo_uri=getattr(args, "mongo_uri", None))
    except (ConnectionFailure, ConfigurationError, OperationFailure) as e:
        logger.error(f"Failed to initialize tool manager: {str(e)}")
        return 1

    try:
        if args.command == "register":
            # Load tool from JSON file
            tool_info = load_tool_from_json(args.file_path)

            # Register the tool
            result = tool_manager.register_tool(tool_info)

            logger.info(f"Tool registered: {result['name']}")
            logger.info(f"Description: {result['description']}")
            logger.info(f"Version: {result['version']}")
            logger.info(f"Capabilities: {', '.join(result['capabilities'])}")

        elif args.command == "list":
            # List all tools
            tools = tool_manager.list_tools()

            logger.info(f"Found {len(tools)} registered tools:")
            for tool in tools:
                logger.info(
                    f"- {tool['name']} (v{tool['version']}): {tool['description'][:50]}..."
                )

        elif args.command == "get":
            # Get a tool by name
            tool = tool_manager.get_tool(args.name)

            if tool:
                logger.info(f"Tool: {tool['name']}")
                logger.info(f"Description: {tool['description']}")
                logger.info(f"Version: {tool['version']}")
                logger.info(f"Capabilities: {', '.join(tool['capabilities'])}")
                logger.info(f"Service URL: {tool['service_url']}")
                logger.info(f"Arguments: {len(tool['arguments'])} defined")
                for arg in tool["arguments"]:
                    logger.info(
                        f"  - {arg['name']} ({arg['type']}): {arg['description']}"
                    )
            else:
                logger.error(f"Tool not found: {args.name}")

        elif args.command == "delete":
            # Delete a tool by name
            success = tool_manager.delete_tool(args.name)

            if success:
                logger.info(f"Tool deleted: {args.name}")
            else:
                logger.error(f"Tool not found or could not be deleted: {args.name}")

        else:
            parser.print_help()
    finally:
        # Close MongoDB connection
        if tool_manager:
            tool_manager.close()


if __name__ == "__main__":
    main()
