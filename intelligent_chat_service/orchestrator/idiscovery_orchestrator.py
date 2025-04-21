from orchestrator.graph_orchestrator import GraphOrchestrator
from agents.analyzer_agent import AnalyzerAgent
from agents.planner_agent import PlannerAgent
from agents.executor_agent import ExecutorAgent
import config
from schema import ChatRequest
from utils import logger


class IDiscoveryOrchestrator(GraphOrchestrator):
    """Base orchestration Framework that uses a graph-based approach with analytics and reasoning capabilities."""

    def __init__(self, request: ChatRequest = None):
        super().__init__(
            name="IDiscovery Orchestrator",
            description="Graph-based chatbot with analyzing and reasoning capabilities along with citations.",
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

        # Create analyzer agent node
        analyzer_node = self.add_node(
            AnalyzerAgent(
                name="analyzer_agent",
                model=config.OPENAI_DEFAULT_MODEL,
                require_thought=True,
                human_in_the_loop=human_in_the_loop,
            )
        )

        # Create planner agent node
        planner_node = self.add_node(
            PlannerAgent(
                name="planner_agent",
                model=config.OPENAI_DEFAULT_MODEL,
                require_thought=True,
                request=request,
            )
        )

        # Create executor agent node
        executor_node = self.add_node(
            ExecutorAgent(
                name="executor_agent",
                model=config.OPENAI_DEFAULT_MODEL,
                require_thought=True,
                request=request,
            )
        )

        # Connect analyzer to planner with a default edge
        self.add_edge(analyzer_node, planner_node, label="analysis_complete")

        # Connect planner to executor
        self.add_edge(planner_node, executor_node, label="plan_complete")

        # Store configuration for debugging/monitoring
        self.configuration = {
            "human_in_the_loop": human_in_the_loop,
            "model": config.OPENAI_DEFAULT_MODEL,
        }
