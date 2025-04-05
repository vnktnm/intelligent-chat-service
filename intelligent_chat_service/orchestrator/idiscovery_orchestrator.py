from orchestrator import Orchestrator
from agents.analyzer_agent import AnalyzerAgent
from agents.planner_agent import PlannerAgent
import config
from schema import ChatRequest
from utils import logger


class IDiscoveryOrchestrator(Orchestrator):
    """Base orchestration Framework that uses a series of agents to reason and respond."""

    def __init__(self, request: ChatRequest = None):
        super().__init__(
            name="IDiscovery Orchestrator",
            description="Chatbot with analyzing and reasoning capabilities along with citations.",
        )

        # Get human-in-the-loop configuration from request if available
        human_in_the_loop = config.HUMAN_IN_THE_LOOP_ENABLED
        if request and request.config:
            human_in_the_loop = request.config.human_in_the_loop
            logger.info(
                f"Using human_in_the_loop setting from request: {human_in_the_loop}"
            )
        else:
            logger.info(f"Using default human_in_the_loop setting: {human_in_the_loop}")

        # step 1: analyzer agents
        self.add_step(
            AnalyzerAgent(
                name="analyzer_agent",
                model=config.OPENAI_DEFAULT_MODEL,
                require_thought=True,
                human_in_the_loop=human_in_the_loop,
            )
        )

        # # Step 2: planner agent
        # self.add_step(
        #     PlannerAgent(
        #         name="planner_agent",
        #         model=config.OPENAI_DEFAULT_MODEL,
        #         require_thought=True,
        #         request=request,
        #     )
        # )
