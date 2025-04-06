from typing import List, Dict, Any
from orchestrator import GraphOrchestrator
from schema.graph_orchestrator import GraphNodeDefinition
from agents.analyzer_agent import AnalyzerAgent
from agents.planner_agent import PlannerAgent
import config
from schema import ChatRequest
from utils import logger


class GraphBasicOrchestrator(GraphOrchestrator):
    """A basic graph-based orchestrator implementing the same functionality as IDiscovery."""

    def __init__(self, request: ChatRequest = None):
        super().__init__(
            name="Graph Basic Orchestrator",
            description="Graph-based chatbot with analyzing and reasoning capabilities.",
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

        try:
            # Create analyzer agent with proper configuration
            analyzer_agent = AnalyzerAgent(
                name="analyzer_agent",
                model=config.OPENAI_DEFAULT_MODEL,
                require_thought=True,
                human_in_the_loop=human_in_the_loop,
            )

            # Create planner agent with proper configuration
            planner_agent = PlannerAgent(
                name="planner_agent",
                model=config.OPENAI_DEFAULT_MODEL,
                require_thought=True,
                request=request,
            )

            # Create nodes
            analyzer_node = GraphNodeDefinition(
                id="analyzer",
                step=analyzer_agent,
                dependencies=[],  # No dependencies for first node
                metadata={
                    "description": "Analyzes user input and retrieves relevant information"
                },
            )

            planner_node = GraphNodeDefinition(
                id="planner",
                step=planner_agent,
                dependencies=["analyzer"],  # Depends on analyzer node
                metadata={"description": "Plans and generates the final response"},
            )

            # Add nodes to orchestrator
            self.add_node(analyzer_node)
            self.add_node(planner_node)

            # Validate the graph to ensure all dependencies are correctly set up
            self.validate_graph()

            logger.info(
                f"Graph Basic Orchestrator initialized with {len(self.nodes)} nodes"
            )

        except Exception as e:
            logger.error(f"Error initializing GraphBasicOrchestrator: {str(e)}")
            import traceback

            logger.error(traceback.format_exc())
            raise
