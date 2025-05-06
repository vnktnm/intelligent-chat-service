import config
from clients.agent_client import AgentClient
from utils import logger


async def execute_analyzer_agent(context):
    """Execute the analyzer agent as a service"""
    result = await AgentClient.execute_agent(
        agent_url=config.ANALYZER_AGENT_URL,
        context=context,
        agent_name="analyzer",
        use_kafka=config.USE_KAFKA_FOR_AGENTS,
    )

    if result["status"] == "success":
        return result["result"]
    else:
        logger.error(
            f"Analyzer agent execution failed: {result.get('message', 'Unknown error')}"
        )
        raise Exception(
            f"Analyzer agent error: {result.get('message', 'Unknown error')}"
        )


async def execute_planner_agent(context):
    """Execute the planner agent as a service"""
    result = await AgentClient.execute_agent(
        agent_url=config.PLANNER_AGENT_URL,
        context=context,
        agent_name="planner",
        use_kafka=config.USE_KAFKA_FOR_AGENTS,
    )

    if result["status"] == "success":
        return result["result"]
    else:
        logger.error(
            f"Planner agent execution failed: {result.get('message', 'Unknown error')}"
        )
        raise Exception(
            f"Planner agent error: {result.get('message', 'Unknown error')}"
        )
