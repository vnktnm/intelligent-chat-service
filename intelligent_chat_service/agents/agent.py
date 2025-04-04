from schema import Step
import config
from typing import Optional, List, Dict, Any, Callable
from tools.tool import ToolManager
import uuid
from core import OpenAIService
from utils import logger
import json
import aiohttp


class Agent(Step):
    """Base class for agents."""

    def __init__(
        self,
        name: str,
        description: str,
        role: str,
        system_prompt: str,
        model: str = config.OPENAI_DEFAULT_MODEL,
        temperature: float = config.OPENAI_DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = None,
        tools: List[Dict[str, Any]] = None,
        response_format: Optional[Any] = None,
        require_thought: bool = True,
        tool_calls: List[str] = None,
    ):
        super().__init__(name, description)
        self.role = role
        self.system_prompt = system_prompt
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.tools = tools or []
        self.tool_functions = {}
        self.conversation_history = []

        # Initialize the tool manager
        self.tool_manager = ToolManager(config.MONGO_TOOL_COLLECTION_NAME)

        # Add a unique agent id for better tracking
        self.agent_id = f"agent_{name}_{uuid.uuid4().hex[:6]}"

        self.response_format = response_format
        self.require_thought = require_thought
        self.tool_calls = tool_calls

    async def think(
        self, input_text: str, context: Dict[str, Any], openai_service: OpenAIService
    ) -> str:
        """Internal reasoning process for the agent"""
        messages = [
            {
                "role": "system",
                "content": f"{self.system_prompt}\nYou are thinking internally and you can just respond your thought and response format like JSON can be negated.",
            },
            {"role": "user", "content": f"Think about: {input_text}"},
        ]

        kwargs = {
            "messages": messages,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }

        if self.tools:
            kwargs["tools"] = self.tools
            kwargs["tool_choice"] = config.OPENAI_TOOL_CHOICE

        response = await openai_service.generate_completions(**kwargs)

        logger.debug(f"Agent {self.name} thought: {response}")

        if "tool_calls" in response["choices"][0]["message"]:
            thought = "I should use tools to help with this request"
            tool_calls = response["choices"][0]["message"]["tool_calls"]
            for tool_call in tool_calls:
                tool_name = tool_call["function"]["name"]
                thought += f"I will use {tool_name} to help with this request"
        else:
            thought = response["choices"][0]["message"]["content"]

        logger.debug(f"Agent {self.name} thought: {thought}")
        return thought

    async def execute(
        self,
        context: Dict[str, Any],
        openai_service: OpenAIService,
        callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Execute this agent with the given context."""

        if self.tool_calls:
            await self.load_tools()

        input_text = context.get("input_text", "")

        for step_name, value in context.items():
            if step_name.endswith("_output") and isinstance(value, str):
                input_text = f"{input_text}\n\nPrevious Step Output: {value}"

        agent_info = {
            "step": self.name,
            "description": self.description,
            "role": self.role,
            "status": "starting",
        }

        if callback:
            await callback("step_update", agent_info)

        logger.info(f"Executing agent {self.name} with input: {input_text}")

        if self.require_thought:
            thought = await self.think(input_text, context, openai_service)
            logger.debug(f"Agent {self.name} thought: {thought}")

        content = await self.respond(input_text, context, openai_service, callback)

        context[self.name] = content
        context[f"{self.name}_output"] = content
        self.result = content

        agent_info["status"] = "completed"
        agent_info["result"] = content

        if callback:
            await callback("step_complete", agent_info)

        return context

    async def respond(
        self,
        input_text: str,
        context: Dict[str, Any],
        openai_service: OpenAIService,
        callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> str:
        """Generate a response for the given input text."""
        messages = [{"role": "system", "content": self.system_prompt}]

        if self.conversation_history:
            messages.extend(self.conversation_history)

        messages.append({"role": "user", "content": input_text})

        if callback:
            await callback(
                "content_start",
                {
                    "agent": self.name,
                    "role": self.role,
                    "agent_id": self.agent_id,
                    "description": self.description,
                },
            )

        kwargs = {
            "messages": messages,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        if self.tools:
            kwargs["tools"] = self.tools
            kwargs["tool_choice"] = config.OPENAI_TOOL_CHOICE

        tool_was_used = False
        content = ""

        if callback:
            response = await openai_service.generate_completions(
                **kwargs, stream=False, response_format=self.response_format
            )
            logger.debug(f"Agent {self.agent_id} response: {response}")

            if "tool_calls" in response["choices"][0]["message"]:
                tool_was_used = True
                tool_calls = response["choices"][0]["message"]["tool_calls"]

                await callback(
                    "tool_use",
                    {
                        "agent": self.name,
                        "role": self.role,
                        "agent_id": self.agent_id,
                        "tools": [tc["function"]["name"] for tc in tool_calls],
                    },
                )

                for tool_call in tool_calls:
                    logger.info(
                        f"Agent {self.agent_id} - validating tool calls - {tool_call}"
                    )
                    tool_result = await self.execute_tool(tool_call, context)

                    messages.append(
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [tool_call],
                        }
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": tool_result,
                        }
                    )

                    await callback(
                        "tool_result",
                        {
                            "agent": self.name,
                            "role": self.role,
                            "agent_id": self.agent_id,
                            "tool": tool_call["function"]["name"],
                            "result": tool_result,
                        },
                    )

                kwargs["messages"] = messages
                content_chunks = []

                async for chunk in openai_service.stream_completion(
                    **kwargs, response_format=self.response_format
                ):
                    await callback(
                        "content_chunk",
                        {
                            "chunk": chunk,
                            "agent": self.name,
                            "role": self.role,
                            "agent_id": self.agent_id,
                        },
                    )
                    content_chunks.append(chunk)

                # get the full content for conversation history
                final_response = await openai_service.generate_completions(
                    **kwargs, stream=False, response_format=self.response_format
                )
                content = final_response["choices"][0]["message"]["content"]
            else:
                content = response["choices"][0]["message"]["content"]
                content_chunks = [content]

                await callback(
                    "content_chunk",
                    {
                        "chunk": content,
                        "agent": self.name,
                        "role": self.role,
                        "agent_id": self.agent_id,
                    },
                )

            await callback(
                "content_end",
                {
                    "agent": self.name,
                    "role": self.role,
                    "agent_id": self.agent_id,
                },
            )
        else:
            response = await openai_service.generate_completions(
                **kwargs, stream=False, response_format=self.response_format
            )

            if "tool_calls" in response["choices"][0]["message"]:
                tool_was_used = True
                tool_calls = response["choices"][0]["message"]["tool_calls"]

                for tool_call in tool_calls:
                    tool_result = await self.execute_tool(tool_call, context)

                    messages.append(
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [tool_call],
                        }
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": tool_result,
                        }
                    )

                kwargs["messages"] = messages
                final_response = await openai_service.generate_completions(
                    **kwargs, stream=False, response_format=self.response_format
                )
                content = final_response["choices"][0]["message"]["content"]
            else:
                content = response["choices"][0]["message"]["content"]

        self.conversation_history.append({"role": "user", "content": input_text})

        if tool_was_used:
            tool_messages = [
                msg
                for msg in messages
                if msg.get("role") in ["tool", "assistant"]
                and messages.index(msg) > len(self.conversation_history)
            ]
            self.conversation_history.extend(tool_messages)

        self.conversation_history.append({"role": "assistant", "content": content})

        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]

        return content

    async def load_tools(self) -> None:
        if len(self.tool_calls) > 0:
            loaded_tool, tool_function = await self.tool_manager.load_tools(
                tools=self.tool_calls
            )

            logger.info(f"Loaded tools: {loaded_tool}")

            if loaded_tool:
                self.tools.extend(loaded_tool)
                self.tool_functions.update(tool_function)

    async def execute_tool(
        self, tool_call: Dict[str, Any], context: Dict[str, Any]
    ) -> str:
        """Execute a tool call and return the result."""
        tool_name = tool_call["function"]["name"]
        try:
            arguments = json.loads(tool_call["function"]["arguments"])

            if tool_name not in self.tool_functions:
                return f"Error: Tool {tool_name} not found."

            tool_info = self.tool_functions[tool_name]
            service_url = tool_info["service_url"]
            tool_id = tool_info["tool_id"]
            metadata = tool_info.get("metadata", {})

            logger.info(f"Executing tool {tool_name} with arguments: {arguments}")

            payload = {
                "tool_name": tool_name,
                "tool_id": tool_id,
                "arguments": arguments,
                "agent_id": self.agent_id,
                "context": context.get("user_input", ""),
                "metadata": metadata,
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(service_url, json=payload) as response:
                    if response.status == 200:
                        result_data = await response.json()
                        result = result_data.get("result", "No result provided")
                        return str(result)
                    else:
                        error_text = await response.text()
                        error_msg = f"API call failed with status {response.status}: {error_text}"
                        logger.error(error_msg)
                        return f"Error executing tool: {error_msg}"
        except aiohttp.ClientError as e:
            error_msg = f"Network error occurred: {str(e)}"
            logger.error(error_msg)
            return error_msg
        except json.JSONDecodeError as e:
            error_msg = f"Error decoding JSON response: {str(e)}"
            logger.error(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"An unexpected error occurred: {str(e)}"
            logger.error(error_msg)
            return error_msg
