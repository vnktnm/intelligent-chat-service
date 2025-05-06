from .graph_orchestrator import GraphOrchestrator
from schema.graph_orchestrator import GraphNodeDefinition
from agents.analyzer_agent import AnalyzerAgent
from agents.clarification_agent import ClarificationAgent
from agents.executor_agent import ExecutorAgent
from agents.planner_agent import PlannerAgent
import config
from schema import ChatRequest
from utils import logger
import json


class IdiscoveryGraphOrchestrator(GraphOrchestrator):
    """A basic graph-based orchestrator implementing the same functionality as IDiscovery."""

    def __init__(self, request: ChatRequest = None):
        super().__init__(
            name="Idiscovery Graph Orchestrator",
            description="Idiscovery Graph Orchestrator with Base Agents.",
        )

        try:
            # Create analyzer agent with proper configuration
            analyzer_agent = AnalyzerAgent(
                name="analyzer_agent",
                model=config.OPENAI_DEFAULT_MODEL,
                require_thought=False,
            )

            # Create clarification agent if needed
            clarification_agent = ClarificationAgent(
                name="clarification_agent",
                model=config.OPENAI_DEFAULT_MODEL,
                require_thought=True,
                request=request,
                human_in_the_loop=True,
            )

            # Create planner agent with proper configuration
            planner_agent = PlannerAgent(
                name="planner_agent",
                model=config.OPENAI_DEFAULT_MODEL,
                require_thought=True,
                request=request,
            )

            executor_agent = ExecutorAgent(
                name="executor_agent",
                model=config.OPENAI_DEFAULT_MODEL,
                require_thought=True,
            )

            def is_request_clear(context):
                analyzer_result = json.loads(context.get("analyzer_agent_output", {}))

                if analyzer_result["analysis"] not in ["ambiguous"]:
                    return True
                return False

            def is_request_unclear(context):
                analyzer_result = json.loads(context.get("analyzer_agent_output", {}))

                if analyzer_result["analysis"] in ["ambiguous"]:
                    return True
                return False

            def has_been_clarified(context):
                clarification_result = json.loads(
                    context.get("clarification_agent_output", {})
                )
                return clarification_result["valid"]

            # Create nodes first without dependencies or conditional edges
            analyzer_node = GraphNodeDefinition(
                id="analyzer",
                step=analyzer_agent,
                dependencies=[],
                priority=10,
                metadata={
                    "description": "Analyzes user input and retrieves the analysis - simple | complex | ambiguous"
                },
            )

            planner_node = GraphNodeDefinition(
                id="planner",
                step=planner_agent,
                dependencies=[],  # We'll add these with add_edge
                priority=5,
                metadata={"description": "Generates a plan based on the query"},
                condition=lambda context: is_request_clear(context)
                or has_been_clarified(context),
            )

            clarification_node = GraphNodeDefinition(
                id="clarification",
                step=clarification_agent,
                dependencies=[],  # We'll add these with add_edge
                priority=5,
                metadata={"description": "Clarifies queries with HITL"},
                condition=is_request_unclear,
            )

            execution_node = GraphNodeDefinition(
                id="executor",
                step=executor_agent,
                dependencies=[],  # We'll add these with add_edge
                priority=5,
                metadata={"description": "Executes the plans from the planner"},
            )

            # Add nodes to the graph
            self.add_node(analyzer_node)
            self.add_node(planner_node)
            self.add_node(clarification_node)
            self.add_node(execution_node)

            # Add standard edges using the new add_edge method
            self.add_edge("analyzer", "planner")
            self.add_edge("analyzer", "clarification")
            self.add_edge("clarification", "planner")
            self.add_edge("planner", "executor")

            # Add conditional edges using the new add_conditional_edge method
            self.add_conditional_edge(
                from_node_id="analyzer",
                to_node_id="planner",
                condition=is_request_clear,
                priority=10,
            )
            self.add_conditional_edge(
                from_node_id="analyzer",
                to_node_id="clarification",
                condition=is_request_unclear,
                priority=5,
            )
            self.add_conditional_edge(
                from_node_id="clarification",
                to_node_id="planner",
                condition=has_been_clarified,
                priority=10,
            )

            self.validate_graph()

            logger.info(
                f"Graph Orchestrator initialized with {len(self.nodes)} nodes and conditional branching"
            )

        except Exception as e:
            logger.error(f"Error initializing Graph Basic Orchestrator: {str(e)}")
            raise
