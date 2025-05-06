from typing import Dict, Any, Optional, Callable, List
from clients.kafka_client import KafkaClient
from utils import logger
import asyncio
import config
import uuid


class KafkaMessageService:
    """Service for handling agent communication via Kafka"""

    _instance = None

    @classmethod
    def get_instance(cls):
        """Get or create singleton instance"""
        if cls._instance is None:
            cls._instance = KafkaMessageService()
        return cls._instance

    def __init__(self):
        self.kafka_client = KafkaClient(
            bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS
        )
        self.response_listeners: Dict[str, asyncio.Future] = {}
        self.agent_listeners: Dict[str, List[Callable]] = {}
        self.initialized = False

    async def initialize(self):
        """Initialize the service"""
        if not self.initialized:
            await self.kafka_client.initialize()
            self.kafka_client.start_consumer(
                topics=[config.KAFKA_RESPONSE_TOPIC],
                group_id=f"agent-service-{uuid.uuid4().hex[:8]}",
                callback=self._handle_response_message,
            )
            self.initialized = True
            logger.info("Kafka message service initialized")

    async def _handle_response_message(self, topic: str, message: Dict[str, Any]):
        """Handle incoming response messages"""
        if topic != config.KAFKA_RESPONSE_TOPIC:
            return

        request_id = message.get("request_id")
        if request_id and request_id in self.response_listeners:
            future = self.response_listeners.pop(request_id)
            if not future.done():
                future.set_result(message)

        agent_id = message.get("agent_id")
        if agent_id and agent_id in self.agent_listeners:
            for callback in self.agent_listeners[agent_id]:
                try:
                    await callback(message)
                except Exception as e:
                    logger.error(f"Error in agent listener callback: {str(e)}")

    async def send_agent_request(
        self,
        agent_name: str,
        context: Dict[str, Any],
        request_id: str = None,
        timeout: float = 60.0,
    ) -> Dict[str, Any]:
        """
        Send request to an agent via Kafka and wait for response
        """
        await self.initialize()

        if not request_id:
            request_id = str(uuid.uuid4())

        message = {
            "request_id": request_id,
            "agent_name": agent_name,
            "context": context,
        }

        # Create a future to await the response
        response_future = asyncio.Future()
        self.response_listeners[request_id] = response_future

        # Send the request
        topic = f"{config.KAFKA_REQUEST_TOPIC_PREFIX}{agent_name}"
        success = await self.kafka_client.send_message(topic, message, key=request_id)

        if not success:
            del self.response_listeners[request_id]
            raise Exception(f"Failed to send request to agent {agent_name}")

        try:
            # Wait for the response with timeout
            response = await asyncio.wait_for(response_future, timeout)
            return response
        except asyncio.TimeoutError:
            del self.response_listeners[request_id]
            raise Exception(
                f"Request to agent {agent_name} timed out after {timeout} seconds"
            )

    async def send_agent_response(
        self, request_id: str, agent_id: str, agent_name: str, result: Dict[str, Any]
    ) -> bool:
        """Send a response back via Kafka"""
        await self.initialize()

        message = {
            "request_id": request_id,
            "agent_id": agent_id,
            "agent_name": agent_name,
            "result": result,
            "status": "success",
        }

        return await self.kafka_client.send_message(
            config.KAFKA_RESPONSE_TOPIC, message, key=request_id
        )

    def register_agent_listener(self, agent_id: str, callback: Callable):
        """Register a callback for messages to a specific agent"""
        if agent_id not in self.agent_listeners:
            self.agent_listeners[agent_id] = []
        self.agent_listeners[agent_id].append(callback)

    def unregister_agent_listener(self, agent_id: str, callback: Callable = None):
        """Unregister a callback for an agent"""
        if agent_id in self.agent_listeners:
            if callback is None:
                del self.agent_listeners[agent_id]
            else:
                self.agent_listeners[agent_id] = [
                    cb for cb in self.agent_listeners[agent_id] if cb != callback
                ]

    async def shutdown(self):
        """Shutdown the service"""
        await self.kafka_client.shutdown()
        self.initialized = False
        logger.info("Kafka message service shutdown complete")
