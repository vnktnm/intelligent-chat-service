from .graph_orchestrator import GraphOrchestrator
from schema.graph_orchestrator import GraphNodeDefinition
from agents.analyzer_agent import AnalyzerAgent
from agents.clarification_agent import ClarificationAgent
from agents.executor_agent import ExecutorAgent
from agents.planner_agent import PlannerAgent
from agents.consolidator_agent import ConsolidatorAgent
import config
from schema import ChatRequest
from utils import logger
import json
from utils.prompt_utils import get_prompt, get_formatted_prompt
from schema.idiscovery_orchestrator import (
    AnalyzerResponse,
    ClarificationResponse,
    PlannerResponse,
    ExecutionResponse,
    ConsolidatorResponse,
)


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
                name="analyzer",
                description="An agent that analyzes the user query and classifies based on the complexity",
                role="analyzer",
                system_prompt=get_prompt(
                    config.PROMPT_PATH,
                    config.PROMPT_AGENT_TYPE,
                    config.PROMPT_ANALYZER_AGENT,
                )["prompt"],
                model=config.OPENAI_DEFAULT_MODEL,
                temperature=config.OPENAI_DEFAULT_TEMPERATURE,
                max_tokens=200,
                tools=None,
                response_format=AnalyzerResponse,
                require_thought=True,
                human_in_the_loop=False,
                stream=False,
            )

            clarification_prompt = get_prompt(
                config.PROMPT_PATH,
                config.PROMPT_AGENT_TYPE,
                config.PROMPT_CLARIFICATION_AGENT,
            )
            formatted_clarification_prompt = get_formatted_prompt(
                clarification_prompt["prompt"],
                variables={"data_source": ", ".join(request.selected_sources)},
            )

            # Create clarification agent if needed
            clarification_agent = ClarificationAgent(
                name="clarification",
                description="An agent that clarifies the query with the human.",
                role="clarification",
                system_prompt=formatted_clarification_prompt,
                model=config.OPENAI_DEFAULT_MODEL,
                temperature=config.OPENAI_DEFAULT_TEMPERATURE,
                max_tokens=1000,
                tools=None,
                response_format=ClarificationResponse,
                require_thought=False,
                human_in_the_loop=True,
                stream=True,
            )

            planner_prompt = get_prompt(
                config.PROMPT_PATH,
                config.PROMPT_AGENT_TYPE,
                config.PROMPT_PLANNER_AGENT,
            )
            formatted_planner_prompt = get_formatted_prompt(
                planner_prompt["prompt"],
                variables={"data_source": ", ".join(request.selected_sources)},
            )

            # Create planner agent with proper configuration
            planner_agent = PlannerAgent(
                name="planner",
                description="An agent that plans the usage of tools based on the user query.",
                role="planner",
                system_prompt=formatted_planner_prompt,
                model=config.OPENAI_DEFAULT_MODEL,
                temperature=config.OPENAI_DEFAULT_TEMPERATURE,
                max_tokens=1000,
                tools=None,
                response_format=PlannerResponse,
                require_thought=False,
                human_in_the_loop=False,
                stream=False,
            )

            executor_agent = ExecutorAgent(
                name="executor",
                description="An agent that executes the plan.",
                role="executor",
                system_prompt=None,
                model=config.OPENAI_DEFAULT_MODEL,
                temperature=config.OPENAI_DEFAULT_TEMPERATURE,
                max_tokens=1000,
                tools=None,
                response_format=ExecutionResponse,
                require_thought=False,
                human_in_the_loop=False,
                stream=False,
            )

            consolidator_agent = ConsolidatorAgent(
                name="consolidator",
                description="An agent that consolidates the final response.",
                role="consolidator",
                system_prompt=get_prompt(
                    config.PROMPT_PATH,
                    config.PROMPT_AGENT_TYPE,
                    config.PROMPT_CONSOLIDATOR_AGENT,
                )["prompt"],
                model=config.OPENAI_DEFAULT_MODEL,
                temperature=config.OPENAI_DEFAULT_TEMPERATURE,
                max_tokens=1000,
                tools=None,
                response_format=ConsolidatorResponse,
                require_thought=False,
                human_in_the_loop=False,
                stream=True,
            )

            def is_request_clear(context):
                # Check if output is already a dict or parse as JSON if it's a string
                analyzer_result = json.loads(context.get("analyzer_output", {}))

                if analyzer_result["analysis"] not in ["ambiguous"]:
                    return True
                return False

            def is_request_unclear(context):
                # Check if output is already a dict or parse as JSON if it's a string
                analyzer_result = json.loads(context.get("analyzer_output", {}))

                if analyzer_result["analysis"] in ["ambiguous"]:
                    return True
                return False

            def has_been_clarified(context):
                # Check if output is already a dict or parse as JSON if it's a string
                clarification_result = json.loads(
                    context.get("clarification_output", {})
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

            consolidator_node = GraphNodeDefinition(
                id="consolidator",
                step=consolidator_agent,
                dependencies=[],
                priority=5,
                metadata={"description": "Consolidates the response from the executor"},
            )

            # Add nodes to the graph
            self.add_node(analyzer_node)
            self.add_node(planner_node)
            self.add_node(clarification_node)
            self.add_node(execution_node)
            self.add_node(consolidator_node)

            # Add standard edges using the new add_edge method
            self.add_edge("analyzer", "planner")
            self.add_edge("analyzer", "clarification")
            self.add_edge("clarification", "planner")
            self.add_edge("planner", "executor")
            self.add_edge("executor", "consolidator")

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
