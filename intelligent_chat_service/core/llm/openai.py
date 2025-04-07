from openai import AsyncOpenAI
import config
from utils import logger
from typing import Dict, List, Optional, AsyncGenerator, Any
from fastapi.exceptions import HTTPException
import asyncio
import json


class OpenAIService:
    def __init__(self):
        self.api_key = config.OPENAI_API_KEY

        self.client = AsyncOpenAI(api_key=self.api_key)

    async def generate_completions(
        self,
        messages: List[Dict[str, Any]],
        model: str = config.OPENAI_DEFAULT_MODEL,
        temperature: float = config.OPENAI_DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        response_format: Optional[Any] = None,
        tools: List[Dict[str, Any]] = None,
        tool_choice: str = None,
    ):
        """Generate a completion using OpenAI API.

        Args:
            messages (List[Dict[str, Any]]): List of messages for the chat.
            model (str, optional): Model to use. Defaults to config.OPENAI_DEFAULT_MODEL.
            temperature (float, optional): Temperature for randomness. Defaults to config.OPENAI_DEFAULT_TEMPERATURE.
            max_tokens (Optional[int], optional): Max tokens for the response. Defaults to None.
            stream (bool, optional): Whether to stream the response. Defaults to False.
            response_format (Optional[Any], optional): Format of the response. Defaults to None.
            tools (List[Dict[str, Any]], optional): List of tools for the agent. Defaults to None.
            tool_choice (str, optional): Tool choice for the agent. Defaults to None.

        Raises:
            HTTPException: If an error occurs during API call.

        Returns:
            AsyncGenerator: Streamed response from OpenAI API.
        """
        try:
            logger.info(f"Making OpenAI completion request with Model: {model}")
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                response_format=response_format,
                tools=tools,
                tool_choice=tool_choice,
            )
            logger.info(f"Agent openai call - {response}")

            if not stream:
                return {
                    "id": response.id,
                    "model": response.model,
                    "choices": [
                        {
                            "message": {
                                "role": choice.message.role,
                                "content": choice.message.content,
                            },
                            "finish_reason": choice.finish_reason,
                        }
                        for choice in response.choices
                    ],
                }
            return response
        except Exception as e:
            if "401" in str(e):
                logger.error(f"Openai API authentication error: {str(e)}")
                raise HTTPException(
                    status_code=401, detail=f"OpenAI API authentication error: {str(e)}"
                )

    async def stream_completion(
        self,
        messages: List[Dict[str, Any]],
        model: str = config.OPENAI_DEFAULT_MODEL,
        temperature: float = config.OPENAI_DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = None,
        response_format: Optional[Any] = None,
        tools: List[Dict[str, Any]] = None,
        tool_choice: str = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream completion using OpenAI API.
        """
        response_stream = None
        try:
            response_stream = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                response_format=response_format,
                tools=tools,
                tool_choice=tool_choice,
            )

            async for chunk in response_stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    content = delta.content or ""

                    if content:
                        yield f"data: {json.dumps({"content": content})}\n\n"

                        await asyncio.sleep(0.01)

            yield f"data: {json.dumps({"done": True})}\n\n"
        except Exception as e:
            logger.error(f"Error in streaming completion: {str(e)}")
            yield f"data: {json.dumps({"error": e})}\n\n"
        finally:
            if response_stream and hasattr(response_stream, "aclose"):
                await response_stream.aclose()


def get_openai_service() -> OpenAIService:
    """
    Get OpenAI service instance.
    """
    try:
        return OpenAIService()
    except Exception as e:
        logger.error(f"Error initializing OpenAIService: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error initializing OpenAIService: {str(e)}"
        )
