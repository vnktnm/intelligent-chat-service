import aiohttp
import json
from typing import Dict, Any, Optional
import config
from utils import logger
from services.kafka_message_service import KafkaMessageService


class AgentClient:
    """Client for interacting with agent services"""

    @staticmethod
    async def execute_agent(
        agent_url: str,
        context: Dict[str, Any],
        agent_name: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        require_thought: Optional[bool] = None,
        human_in_the_loop: Optional[bool] = None,
        use_kafka: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Execute an agent via its service endpoint or Kafka
        """
        # Determine if we should use Kafka based on config or parameter
        if use_kafka is None:
            use_kafka = config.USE_KAFKA_FOR_AGENTS

        if use_kafka:
            return await AgentClient._execute_agent_via_kafka(
                agent_name=agent_name,
                context=context,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                require_thought=require_thought,
                human_in_the_loop=human_in_the_loop,
            )
        else:
            return await AgentClient._execute_agent_via_http(
                agent_url=agent_url,
                context=context,
                agent_name=agent_name,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                require_thought=require_thought,
                human_in_the_loop=human_in_the_loop,
            )

    @staticmethod
    async def _execute_agent_via_http(
        agent_url: str,
        context: Dict[str, Any],
        agent_name: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        require_thought: Optional[bool] = None,
        human_in_the_loop: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Execute an agent via its HTTP service endpoint
        """
        payload = {
            "context": context,
            "agent_name": agent_name,
        }

        # Add optional parameters if provided
        if model is not None:
            payload["model"] = model
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if require_thought is not None:
            payload["require_thought"] = require_thought
        if human_in_the_loop is not None:
            payload["human_in_the_loop"] = human_in_the_loop

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{agent_url}/execute", json=payload
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        logger.error(f"Agent service error: {error_text}")
                        return {
                            "status": "error",
                            "message": f"Agent service error: {error_text}",
                        }
        except aiohttp.ClientError as e:
            logger.error(f"Network error connecting to agent service: {str(e)}")
            return {"status": "error", "message": f"Network error: {str(e)}"}
        except Exception as e:
            logger.error(f"Error calling agent service: {str(e)}")
            return {"status": "error", "message": f"Error: {str(e)}"}

    @staticmethod
    async def _execute_agent_via_kafka(
        agent_name: str,
        context: Dict[str, Any],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        require_thought: Optional[bool] = None,
        human_in_the_loop: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Execute an agent via Kafka messaging
        """
        # Enhance context with agent parameters
        enhanced_context = dict(context)
        if model is not None:
            enhanced_context["_agent_model"] = model
        if temperature is not None:
            enhanced_context["_agent_temperature"] = temperature
        if max_tokens is not None:
            enhanced_context["_agent_max_tokens"] = max_tokens
        if require_thought is not None:
            enhanced_context["_agent_require_thought"] = require_thought
        if human_in_the_loop is not None:
            enhanced_context["_agent_hitl"] = human_in_the_loop

        # Get Kafka message service and send request
        try:
            kafka_service = KafkaMessageService.get_instance()
            response = await kafka_service.send_agent_request(
                agent_name=agent_name,
                context=enhanced_context,
                timeout=config.KAFKA_REQUEST_TIMEOUT,
            )

            if response and "status" in response:
                return response

            # Format the response to match the HTTP service format
            return {
                "status": "success",
                "agent_id": response.get("agent_id", "unknown"),
                "result": response.get("result", {}),
                "agent_output": response.get("result", {}).get(
                    f"{agent_name}_output", ""
                ),
            }

        except Exception as e:
            logger.error(f"Error executing agent via Kafka: {str(e)}")
            return {"status": "error", "message": f"Kafka error: {str(e)}"}
