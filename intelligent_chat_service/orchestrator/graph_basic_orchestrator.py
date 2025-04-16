from typing import List, Dict, Any
from orchestrator import GraphOrchestrator
from schema.graph_orchestrator import GraphNodeDefinition, ConditionalEdge
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

            # Create clarification agent if needed
            clarification_agent = PlannerAgent(
                name="clarification_agent",
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

            clarification_node = GraphNodeDefinition(
                id="clarification",
                step=clarification_agent,
                dependencies=[],  # We'll add the dependency conditionally
                metadata={"description": "Clarifies ambiguous user input"},
            )

            # Add nodes to orchestrator
            self.add_node(analyzer_node)
            self.add_node(planner_node)
            self.add_node(clarification_node)

            # Define conditional edges based on analyzer results
            def is_ambiguous(context: Dict[str, Any]) -> bool:
                analyzer_result = context.get("analyzer_agent_result", {})

                is_ambiguous = False
                ambiguity_signals = [
                    "ambiguous",
                    "unclear",
                    "vague",
                    "not specific",
                    "could mean",
                    "multiple interpretations",
                    "not sure",
                ]

                analysis_text = str(analyzer_result).lower()
                for signal in ambiguity_signals:
                    if signal in analysis_text:
                        is_ambiguous = True
                        break

                return is_ambiguous

            self.add_conditional_edge(
                "analyzer",
                ConditionalEdge(
                    target_node="clarification",
                    condition=is_ambiguous,
                    description="Route to clarification if input is ambiguous",
                ),
            )

            clarification_node.dependencies.append("analyzer")
            planner_node.dependencies = []

            def is_not_ambiguous(context: Dict[str, Any]) -> bool:
                return not is_ambiguous(context)

            self.add_conditional_edge(
                "analyzer",
                ConditionalEdge(
                    target_node="planner",
                    condition=is_not_ambiguous,
                    description="Route directly to planner if input is clear",
                ),
            )

            self.add_conditional_edge(
                "clarification",
                ConditionalEdge(
                    target_node="planner",
                    condition=lambda context: True,
                    description="Always go to planner after clarification",
                ),
            )

            self.validate_graph()

            logger.info(
                f"Graph Basic Orchestrator initialized with {len(self.nodes)} nodes and conditional edges"
            )

        except Exception as e:
            logger.error(f"Error initializing GraphBasicOrchestrator: {str(e)}")
            import traceback

            logger.error(traceback.format_exc())
            raise
