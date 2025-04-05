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
        human_in_the_loop: bool = config.HUMAN_IN_THE_LOOP_ENABLED,
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
        self.human_in_the_loop = human_in_the_loop

        # Initialize the tool manager
        self.tool_manager = ToolManager(config.MONGO_TOOL_COLLECTION_NAME)

        # Add a unique agent id for better tracking
        self.agent_id = f"agent_{name}_{uuid.uuid4().hex[:6]}"

        self.response_format = response_format
        self.require_thought = require_thought
        self.tool_calls = tool_calls

        # Add human-in-the-loop instruction to the prompt if enabled
        if human_in_the_loop:
            self.system_prompt += self._get_human_help_instructions()

    def _get_human_help_instructions(self) -> str:
        """Get the instructions for human help requests."""
        return "\n\nIMPORTANT: If you encounter a question or topic that requires human expertise, judgment, clarification, or when you're uncertain about something, you MUST ask for human help by including this exact format in your response: 'HUMAN_HELP_NEEDED: [your specific question]'. Be clear and concise with what you need help with."

    async def think(
        self, input_text: str, context: Dict[str, Any], openai_service: OpenAIService
    ) -> str:
        """Internal reasoning process for the agent with human help detection"""
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

        # If human-in-the-loop is enabled, check if this is a question that might benefit from human input
        if self.human_in_the_loop:
            human_help_indicators = [
                "ethical",
                "dilemma",
                "tradeoff",
                "balance",
                "judgment",
                "opinion",
                "controversial",
                "policy",
                "safety concerns",
                "innovation",
                "decision-making",
                "priorities conflict",
            ]

            # Check if the input contains any indicators that human help might be valuable
            if any(
                indicator in input_text.lower() for indicator in human_help_indicators
            ):
                logger.info(
                    f"Detected potential need for human input in: {input_text[:100]}..."
                )
                thought += "\nThis question appears to involve ethical considerations or value judgments where human input would be valuable. I should consider asking for human assistance."

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

        # Check for human-in-the-loop requests
        if self.human_in_the_loop and self.result:
            help_marker = "HUMAN_HELP_NEEDED:"
            if help_marker in self.result:
                updated_context = await self._handle_human_help_request(
                    help_marker, context, openai_service, callback
                )
                if updated_context:
                    context = updated_context

        agent_info["status"] = "completed"
        agent_info["result"] = self.result

        if callback:
            await callback("step_complete", agent_info)

        return context

    async def _handle_human_help_request(
        self,
        help_marker: str,
        context: Dict[str, Any],
        openai_service: OpenAIService,
        callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Handle a human help request."""
        # Extract the question for the human
        question_start = self.result.find(help_marker) + len(help_marker)
        question_text = self.result[question_start:].strip()

        # Truncate to first paragraph or reasonable length
        end_markers = ["\n\n", "\n---", "\n###"]
        for marker in end_markers:
            pos = question_text.find(marker)
            if pos > 0:
                question_text = question_text[:pos].strip()
                break

        # Log the detection of a human help request
        logger.info(f"Detected human help request: {question_text}")

        # Create a consistent interaction ID
        interaction_id = str(uuid.uuid4())

        # Make sure we properly emit the human input requested event here
        if callback:
            # Explicitly trigger the human_input_requested event
            event_data = {
                "agent": self.name,
                "agent_id": self.agent_id,
                "question": question_text,
                "interaction_id": interaction_id,
                "timestamp": datetime.now().isoformat() + "Z",
            }

            # Store event data in context so it can be used by human_interaction_service
            context["event_data"] = event_data

            await callback("human_input_requested", event_data)
            logger.info(f"Emitted human_input_requested event with id {interaction_id}")

        # Ask the human for help
        human_result = await self.ask_human(question_text, context, callback)

        logger.info(f"Human response status: {human_result['status']}")

        # Create follow-up with the human's response
        if human_result["status"] == "success":
            follow_up = f"Thank you for your help. The human has provided this response: '{human_result['response']}'. Please refine your analysis with this information."

            # Get a refined response that incorporates the human feedback
            refined_response = await self.respond(
                follow_up, context, openai_service, callback
            )

            # Update the result
            context[self.name] = refined_response
            context[f"{self.name}_output"] = refined_response
            self.result = refined_response

            # Update the agent info in case callback needs to be called
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

    async def ask_human(
        self,
        question: str,
        context: Dict[str, Any],
        callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Ask a human for input when the agent is uncertain."""
        if not self.human_in_the_loop:
            logger.warning(
                f"Agent {self.name} attempted to ask human but human-in-the-loop is disabled"
            )
            return {"response": None, "status": "disabled", "interaction_id": None}

        logger.info(
            f"Agent {self.agent_id} is requesting human input for question: {question}"
        )

        # Get the human interaction service
        human_service = get_human_interaction_service()

        # Request human input
        result = await human_service.request_human_input(
            agent_id=self.agent_id,
            question=question,
            context=context,
            timeout=config.HUMAN_RESPONSE_TIMEOUT,
        )

        # Ensure we have an interaction_id
        interaction_id = result.get("interaction_id")
        logger.info(
            f"Generated interaction_id: {interaction_id} for human input request"
        )

        # Notify that we're waiting for human input
        if callback and interaction_id:
            logger.info(
                f"Sending human_input_requested event with interaction_id: {interaction_id}"
            )
            event_data = {
                "agent": self.name,
                "role": self.role,
                "agent_id": self.agent_id,
                "question": question,
                "interaction_id": interaction_id,
            }
            logger.debug(f"Event data for human_input_requested: {event_data}")

            await callback("human_input_requested", event_data)
            logger.info("Successfully sent human_input_requested event")
        else:
            if not callback:
                logger.warning("No callback provided for human input request")
            if not interaction_id:
                logger.warning("No interaction_id available for human input request")

        # Notify about the human response if received
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
            logger.info(f"Human response received for interaction {interaction_id}")

        return result

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
