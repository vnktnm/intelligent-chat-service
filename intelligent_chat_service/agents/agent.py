from schema import Step
import config
from typing import Optional, List, Dict, Any, Callable
from tools.tool import ToolManager
import uuid
from core import OpenAIService, get_human_interaction_service
from utils import logger
import json
import aiohttp
from datetime import datetime
from pydantic import BaseModel
from fastapi import FastAPI, Body, HTTPException, Request, Depends
import uvicorn
import os


class Agent(Step):
    """Base class for agents that can help orchestrate workflow."""

    def __init__(
        self,
        name: str,
        description: str,
        role: str,
        system_prompt: str,
        model: str = config.OPENAI_DEFAULT_MODEL,
        temperature: float = config.OPENAI_DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = None,
        tools: List[Dict[str, Any]] = [],
        response_format: Optional[Any] = None,
        require_thought: bool = True,
        tool_calls: Optional[List[str]] = None,
        human_in_the_loop: bool = config.HITL_ENABLED,
    ):
        super().__init__(name, description)
        self.role = role
        self.system_prompt = system_prompt
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.tools = tools
        self.tool_functions = {}
        self.conversation_history = []
        self.tool_manager = ToolManager(config.MONGO_TOOL_COLLECTION_NAME)
        self.agent_id = f"agent_{name}_{uuid.uuid4().hex[:6]}"
        self.response_format = response_format
        self.require_thought = require_thought
        self.tool_calls = tool_calls
        self.human_in_the_loop = human_in_the_loop

    async def think(
        self, input_text: str, context: Dict[str, Any], openai_service: OpenAIService
    ) -> str:
        """Internal reasoning process for the agent."""
        messages = [
            {
                "role": "system",
                "content": f"{self.system_prompt}\nYou are thinking internally and you can just respond with the thought and response format can be negated.",
            },
            {"role": "user", "content": f"Think about: {input_text}"},
        ]

        kwargs = {
            "messages": messages,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
            "response_format": None,
        }

        if self.tools:
            kwargs["tools"] = self.tools
            kwargs["tool_choice"] = "auto"

        response = await openai_service.generate_completion(**kwargs)

        if "tool_calls" in response["choices"][0]["message"]:
            thought = "I should use tools to help with this request."
            tool_calls = response["choices"][0]["message"]["tool_calls"]
            for tool_call in tool_calls:
                tool_name = tool_call["function"]["name"]
                thought += f"I'll use the {tool_name} tool."
        else:
            thought = response["choices"][0]["message"]["content"]

        return thought

    async def execute(
        self,
        context: Dict[str, Any],
        openai_service: OpenAIService,
        callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Execute this agent with the provided context"""

        # todo: add a logic here so that every time if there is a hitl, it will start from here to get going with multi-round discussions.

        await self.load_tools()

        input_text = context.get("user_input", "")

        for step_name, value in context.items():
            if step_name.endswith("_output") and isinstance(value, str):
                input_text = f"{input_text}\n\nPrevious step output: {value}"

        agent_info = {
            "step": self.name,
            "description": self.description,
            "role": self.role,
            "status": "Starting",
        }

        if callback:
            await callback("step_update", agent_info)

        if self.require_thought:
            thought = await self.think(input_text, context, openai_service)
            logger.info(f"Agent {self.agent_id}\nThought: {thought}")
            # todo: handle thought streaming

        content = await self.respond(input_text, context, openai_service, callback)

        context[self.name] = content
        context[f"{self.name}_output"] = content
        self.result = content

        # hitl
        # todo: all agents having hitl should have response_format enabled.
        if self.human_in_the_loop and self.result:
            formatted_result = json.loads(self.result)

            if (
                formatted_result["type"] in ["clarification", "suggestion"]
                and formatted_result["question"]
            ):
                updated_context = await self._handle_human_help_request(
                    formatted_result["question"], context, openai_service, callback
                )
                if updated_context:
                    context = updated_context

        agent_info["status"] = "completed"
        agent_info["result"] = content

        if callback:
            await callback("step_complete", agent_info)

        return context

    async def _handle_human_help_request(
        self,
        question_text: str,
        context: Dict[str, Any],
        openai_service: OpenAIService,
        callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Handle HITL"""
        interaction_id = str(uuid.uuid4())
        session_id = None
        thread_id = None

        if context and "config" in context:
            config = context.get("config", {})
            session_id = config.get("session_id")
            thread_id = config.get("thread_id")

        if callback:
            event_data = {
                "agent": self.name,
                "agent_id": self.agent_id,
                "question": question_text,
                "interaction_id": interaction_id,
                "timestamp": datetime.now().isoformat() + "Z",
                "session_id": session_id,
                "thread_id": thread_id,
            }

            context["event_data"] = event_data

            await callback("human_input_requested", event_data)

        human_result = await self.ask_human(question_text, context, callback)

        if human_result["status"] == "success":
            follow_up = f"The human has provided this response: {human_result["response"]}. Please refine the query with this information making the query valid."

            refined_response = await self.respond(
                follow_up, context, openai_service, callback
            )

            context[self.name] = refined_response
            context[f"{self.name}_output"] = refined_response
            self.result = refined_response

            if callback:
                await callback(
                    "step_update",
                    {
                        "step": self.name,
                        "status": "updated_with_human_input",
                        "result": refined_response,
                    },
                )

            return context
        return None

    async def respond(
        self,
        input_text: str,
        context: Dict[str, Any],
        openai_service: OpenAIService,
        callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ):
        """Generate response from the Agent."""
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
            kwargs["tool_choice"] = "auto"  # todo: needs configurability

        tool_was_used = False
        content = ""

        if callback:
            response = await openai_service.generate_completion(
                **kwargs, stream=False, response_format=self.response_format
            )

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

                    extracted_content = self.extract_content_from_sse(chunk)
                    content_chunks.append(extracted_content)

                content = "".join(content_chunks)
            else:
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

                    extracted_content = self.extract_content_from_sse(chunk)
                    content_chunks.append(extracted_content)

                content = "".join(content_chunks)

            await callback(
                "content_end",
                {"agent": self.name, "role": self.role, "agent_id": self.agent_id},
            )
        else:
            response = await openai_service.generate_completion(
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
                final_response = await openai_service.generate_completion(
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
                if msg.get("role") in ["assistant", "tool"]
                and messages.index(msg) > len(self.conversation_history)
            ]
            self.conversation_history.extend(tool_messages)

        self.conversation_history.append({"role": "assistant", "content": content})

        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[:10]

        return content

    async def ask_human(
        self,
        question: str,
        context: Dict[str, Any],
        callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Ask Human"""
        if not self.human_in_the_loop:
            return {"response": None, "status": "disabled", "interaction_id": None}

        human_service = get_human_interaction_service()

        result = await human_service.request_human_input(
            agent_id=self.agent_id,
            question=question,
            context=context,
            timeout=config.HITL_RESPONSE_TIMEOUT,
        )

        interaction_id = result.get("interaction_id")

        if callback and interaction_id:
            event_data = {
                "agent": self.name,
                "role": self.role,
                "agent_id": self.agent_id,
                "question": question,
                "interaction_id": interaction_id,
            }

            await callback("human_input_requested", event_data)
        else:
            if not callback:
                logger.warning("No callback provided for human input request.")
            if not interaction_id:
                logger.warning("No interaction_id available for human input request.")

        if callback and result["status"] == "success" and interaction_id:
            await callback(
                "human_input_received",
                {
                    "agent": self.name,
                    "role": self.role,
                    "agent_id": self.agent_id,
                    "interaction_id": interaction_id,
                    "response": result["response"],
                },
            )

        return result

    async def load_tools(self) -> None:
        """Load tools"""
        loaded_tool, tool_function = await self.tool_manager.load_tools(
            tools=self.tool_calls
        )

        if loaded_tool:
            self.tools.extend(loaded_tool)
            self.tool_functions.update(tool_function)

    async def execute_tool(
        self, tool_call: Dict[str, Any], context: Dict[str, Any]
    ) -> str:
        """Execute tool calls"""
        tool_name = tool_call["function"]["name"]

        try:
            arguments = json.loads(tool_call["function"]["arguments"])

            transformed_arguments = {}
            if isinstance(arguments, list):
                transformed_arguments = self.transform_dicts_to_single(arguments)

            tool_name = tool_name.split(".")[1]

            if tool_name not in self.tool_functions:
                return f"Error: Tool {tool_name} is not avaiable"

            tool_info = self.tool_functions[tool_name]
            service_url = tool_info["service_url"]
            tool_id = tool_info["tool_id"]
            metadata = tool_info.get("metadata", {})

            payload = {
                "tool_name": tool_name,
                "tool_id": tool_id,
                "arguments": transformed_arguments,
                "agent_id": self.agent_id,
                "context": context.get("user_input", ""),
                "metadata": metadata,
            }

            transformed_arguments["user_id"] = ""  # todo: need to be removed

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    service_url, json=transformed_arguments
                ) as response:
                    if response.status == 200:
                        result_data = await response.json()
                        result = result_data.get("result", "No result provided")
                    else:
                        error_text = await response.text()
                        error_msg = f"API call failed with status {response.status}: {error_text}"
                        return f"Error executing tool {tool_name}: API Call Failed {error_text}"
        except aiohttp.ClientError as e:
            error_msg = f"Network error executing tool {tool_name}: {str(e)}"
            return error_msg
        except json.JSONDecoderError as e:
            error_msg = f"Error parsing arguments for tool {tool_name}: {str(e)}"
            return error_msg
        except Exception as e:
            error_msg = f"Error executing tool {tool_name}: {str(e)}"
            return error_msg

    def extract_content_from_sse(self, sse_chunk: str) -> str:
        """Extract content from SSE formatted chunk"""
        try:
            if sse_chunk.startswith("data: "):
                json_str = sse_chunk[6:].strip()
                chunk = json.loads(json_str)

                if "content" in chunk:
                    return chunk["content"]
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.debug(f"Failed to parse SSE chunks: {e}")

        return ""

    def transform_dicts_to_single(self, dicts):
        """Transform dicts"""
        result_dict = {}

        for original_dict in dicts:
            new_key = original_dict["key"]
            new_value = original_dict["value"]

            result_dict[new_key] = new_value

        return result_dict

    @classmethod
    def create_service(cls, agent_class):
        """
        Create a FastAPI service for this agent
        """
        app = FastAPI(
            title=f"{agent_class.__name__} Service",
            description=f"API for {agent_class.__name__}",
        )

        @app.post("/execute")
        async def execute_agent(request: Dict[str, Any] = Body(...)):
            try:
                context = request.get("context", {})
                agent_name = request.get("agent_name", agent_class.__name__)
                model = request.get("model", config.OPENAI_DEFAULT_MODEL)
                temperature = request.get(
                    "temperature", config.OPENAI_DEFAULT_TEMPERATURE
                )
                max_tokens = request.get("max_tokens", config.OPENAI_MAX_TOKENS)
                require_thought = request.get("require_thought", True)
                human_in_the_loop = request.get(
                    "human_in_the_loop", config.HITL_ENABLED
                )

                # Create agent instance
                agent = agent_class(
                    name=agent_name,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    require_thought=require_thought,
                    human_in_the_loop=human_in_the_loop,
                    request=context.get("request") if "request" in context else None,
                )

                # Create OpenAI service
                from core import get_openai_service

                openai_service = get_openai_service()

                # Execute the agent
                result = await agent.execute(context, openai_service)

                return {
                    "status": "success",
                    "agent_id": agent.agent_id,
                    "result": result,
                    "agent_output": agent.result,
                }
            except Exception as e:
                logger.error(f"Error executing agent: {str(e)}")
                return {"status": "error", "message": str(e)}

        @app.get("/health")
        async def health_check():
            return {"status": "healthy"}

        return app

    @staticmethod
    def run_service(app, port: int = 8000):
        """
        Run the agent as a standalone service
        """
        uvicorn.run(app, host="0.0.0.0", port=port)
