from orchestrator import Orchestrator
from agents.analyzer_agent import AnalyzerAgent
from agents.planner_agent import PlannerAgent
import config
from schema import ChatRequest


class IDiscoveryOrchestrator(Orchestrator):
    """Base orchestration Framework that uses a series of agents to reason and respond."""

    def __init__(self, request: ChatRequest = None):
        super().__init__(
            name="IDiscovery Orchestrator",
            description="Chatbot with analyzing and reasoning capabilities along with citations.",
        )

        # step 1: analyzer agents
        self.add_step(
            AnalyzerAgent(
                name="analyzer_agent",
                model=config.OPENAI_DEFAULT_MODEL,
                require_thought=True,
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
