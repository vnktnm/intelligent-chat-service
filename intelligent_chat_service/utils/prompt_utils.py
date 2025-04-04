import yaml
from .logger_utils import logger
import json
from pydantic import BaseModel
from schema import Tool
from typing import Optional, Dict, Any, Union
from dataclasses import asdict
import re
from utils import logger


class Prompt(BaseModel):
    id: str
    prompt: str
    tags: list[str]
    structured_output: Optional[str] = None


def get_prompt(file_path: str, category, agent_name) -> Union[Dict[str, Any], None]:
    try:
        with open(file_path, "r") as file:
            data = yaml.safe_load(file)

        categories = data.get("categories", {})
        if category in categories:
            agents = categories.get(category, {})
            agent = agents.get(agent_name)

            if agent and isinstance(agent, dict):
                prompt_id = agent.get("id")
                prompt_text = agent.get("prompt")
                tags = agent.get("tags", [])
                structured_output = agent.get("structured_output", {})

                if not prompt_id or not prompt_text:
                    logger.error(
                        f"Missing required prompt fields for agent {agent_name}"
                    )
                    return None

                structured_output_json = json.dumps(structured_output, indent=2)

                result = {
                    "id": prompt_id,
                    "prompt": prompt_text,
                    "tags": tags,
                    "structured_output": structured_output_json,
                }

                return result

            logger.error(
                f"Agent '{agent_name}' not found or invalid in category '{category}'"
            )
            return None
        else:
            logger.error(f"Category '{category}' not found in the YAML file.")
            return None
    except FileNotFoundError:
        logger.error(f"File '{file_path}' not found.")
        return None
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return None


def get_formatted_prompt(prompt: str, variables: dict) -> Optional[str]:
    if not prompt:
        logger.error("Cannot format None prompt")
        return None

    try:
        pattern = re.compile(r"\{\s*(\w+)\s*\}")

        def replace_variable(match):
            variable_name = match.group(1)
            return variables.get(variable_name, match.group(0))

        result = pattern.sub(replace_variable, prompt)
        return result
    except Exception as e:
        logger.error(f"Error formatting prompt: {e}")
        return None
