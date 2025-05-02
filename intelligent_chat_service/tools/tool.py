from core import MongoDBManager
from typing import Optional, Dict, Any, Tuple, List
from utils import logger


class ToolManager:
    def __init__(self, collection_name: str):
        self.database_client = MongoDBManager(collection_name=collection_name)

    async def get_tool_by_name(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get a specific tool by name"""
        if not self.database_client.collection:
            return None

        try:
            return await self.database_client.collection.find_one({"name": tool_name})
        except Exception as e:
            logger.error(f"Error fetching tool by name: {str(e)}")
            return None
        finally:
            self.database_client.close()

    async def load_tools(
        self, tools: List[str] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Load tools from MongoDB and return them as a list of dictionaries."""
        logger.debug(f"loading tools - {tools}")
        if self.database_client.collection is None:
            logger.warning("MongoDB collection is not initialized.")
            return [], {}

        try:
            if not tools:
                cursor = self.database_client.collection.find()
            else:
                cursor = self.database_client.collection.find({"name": {"$in": tools}})

            tools_docs = await cursor.to_list(length=100)

            if not tools_docs:
                logger.warning("No tools found in the database.")
                return [], {}

            openai_tools = []
            tool_functions = {}

            for tool_doc in tools_docs:
                tool_name = tool_doc.get("name")
                description = tool_doc.get("description")
                arguments = tool_doc.get("arguments", {})

                if not tool_name or not description:
                    logger.warning(f"Tool {tool_name} is missing name or description.")
                    continue

                service_url = tool_doc.get("service_url")
                if not service_url:
                    logger.warning(f"Tool {tool_name} is missing service_url.")
                    continue

                metadata = tool_doc.get("metadata", {})
                tool_id = str(tool_doc.get("_id", ""))

                tool_functions[tool_name] = {
                    "service_url": service_url,
                    "tool_id": tool_id,
                    "metadata": metadata,
                }

                openai_tool = {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": description,
                        "parameters": arguments,
                    },
                }

                openai_tools.append(openai_tool)
                logger.info(f"Loaded tool {tool_name} with description: {description}")

            return openai_tools, tool_functions
        except Exception as e:
            logger.error(f"Error loading tools: {str(e)}")
            return [], {}
        finally:
            self.database_client.close()
