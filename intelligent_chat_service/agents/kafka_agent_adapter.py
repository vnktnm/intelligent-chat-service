import asyncio
from typing import Dict, Any, Type, Optional, List
import uuid

from utils import logger
from services.kafka_message_service import KafkaMessageService
from core import get_openai_service
from .agent import Agent
import config


class KafkaAgentAdapter:
    """Adapter to enable Kafka-based communication for agents"""

    def __init__(self, agent_class: Type[Agent], agent_name: str = None):
        self.agent_class = agent_class
        self.agent_name = agent_name or agent_class.__name__.lower().replace(
            "agent", ""
        )
        self.kafka_service = KafkaMessageService.get_instance()
        self.request_topic = f"{config.KAFKA_REQUEST_TOPIC_PREFIX}{self.agent_name}"
        self.running = False
        self.consumer_task = None

    async def start(self):
        """Start listening for agent requests"""
        if self.running:
            return

        await self.kafka_service.initialize()

        # Start consuming messages for this agent
        self.consumer_task = self.kafka_service.kafka_client.start_consumer(
            topics=[self.request_topic],
            group_id=f"{self.agent_name}-service",
            callback=self._handle_request,
            auto_offset_reset=(
                "earliest" if config.KAFKA_PROCESS_ALL_MESSAGES else "latest"
            ),
        )

        self.running = True
        logger.info(
            f"Kafka agent adapter started for {self.agent_name}, listening on {self.request_topic}"
        )

    async def _handle_request(self, topic: str, message: Dict[str, Any]):
        """Handle incoming agent request messages"""
        if topic != self.request_topic:
            return

        request_id = message.get("request_id")
        if not request_id:
            logger.error(f"Received message without request_id on topic {topic}")
            return

        agent_name = message.get("agent_name")
        if agent_name != self.agent_name:
            logger.warning(f"Topic/agent name mismatch: {topic} has agent {agent_name}")

        context = message.get("context", {})

        # Extract agent parameters from context if present
        model = context.pop("_agent_model", config.OPENAI_DEFAULT_MODEL)
        temperature = context.pop(
            "_agent_temperature", config.OPENAI_DEFAULT_TEMPERATURE
        )
        max_tokens = context.pop("_agent_max_tokens", config.OPENAI_MAX_TOKENS)
        require_thought = context.pop("_agent_require_thought", True)
        human_in_the_loop = context.pop("_agent_hitl", config.HITL_ENABLED)

        try:
            # Create agent instance
            agent_instance = self.agent_class(
                name=agent_name or self.agent_name,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                require_thought=require_thought,
                human_in_the_loop=human_in_the_loop,
                request=context.get("request") if "request" in context else None,
            )

            # Execute agent with OpenAI service
            openai_service = get_openai_service()
            result_context = await agent_instance.execute(context, openai_service)

            # Send response
            await self.kafka_service.send_agent_response(
                request_id=request_id,
                agent_id=agent_instance.agent_id,
                agent_name=agent_instance.name,
                result=result_context,
            )

            logger.info(f"Agent {agent_instance.name} completed request {request_id}")

        except Exception as e:
            logger.error(f"Error processing agent request {request_id}: {str(e)}")

            # Send error response
            await self.kafka_service.kafka_client.send_message(
                config.KAFKA_RESPONSE_TOPIC,
                {
                    "request_id": request_id,
                    "agent_name": self.agent_name,
                    "status": "error",
                    "message": str(e),
                },
                key=request_id,
            )

    async def stop(self):
        """Stop the adapter"""
        if not self.running:
            return

        self.running = False
        await self.kafka_service.kafka_client.stop_consumer()
        self.consumer_task = None
        logger.info(f"Kafka agent adapter for {self.agent_name} stopped")
