import yaml
from typing import Dict, Any, List, Optional, Union, Type, Callable
import importlib
import inspect
from schema.graph_orchestrator import GraphNodeDefinition, ConditionalEdge
from schema import Step
from utils import logger
from orchestrator import GraphOrchestrator
import os
import config


def load_agent_class(agent_type: str) -> Optional[Type]:
    """Dynamically load an agent class based on its type name."""
    try:
        # Try to load from common agent locations
        agent_locations = [
            f"agents.{agent_type}",
            f"agents.{agent_type}_agent",
            agent_type,
        ]

        for location in agent_locations:
            try:
                module_path, class_name = location.rsplit(".", 1)
                module = importlib.import_module(module_path)
                agent_class = getattr(module, class_name)
                if inspect.isclass(agent_class) and issubclass(agent_class, Step):
                    return agent_class
            except (ImportError, AttributeError, ValueError):
                continue

        logger.error(f"Could not load agent class for type: {agent_type}")
        return None
    except Exception as e:
        logger.error(f"Error loading agent class {agent_type}: {str(e)}")
        return None


def create_agent_from_config(agent_config: Dict[str, Any]) -> Optional[Step]:
    """Create an agent instance from configuration."""
    agent_type = agent_config.get("type")
    if not agent_type:
        logger.error("Agent configuration missing 'type' field")
        return None

    # Load the agent class
    agent_class = load_agent_class(agent_type)
    if not agent_class:
        return None

    # Get configuration parameters
    config_params = {k: v for k, v in agent_config.items() if k != "type"}

    # Create the agent instance
    try:
        return agent_class(**config_params)
    except Exception as e:
        logger.error(f"Error creating agent {agent_type}: {str(e)}")
        return None


def create_conditional_function(condition_def: Dict[str, Any]) -> Optional[Callable]:
    """
    Create a conditional function from a YAML definition.

    Supported condition types:
    - 'python_expr': Direct Python expression as string
    - 'field_equals': Check if a context field equals a value
    - 'field_contains': Check if a context field contains a value
    - 'complex_condition': Combination of conditions with AND/OR logic

    Args:
        condition_def: Dictionary defining the condition

    Returns:
        A callable function that takes a context dictionary and returns boolean
    """
    condition_type = condition_def.get("type")

    if not condition_type:
        logger.error("Condition definition missing 'type' field")
        return None

    if condition_type == "python_expr":
        # Direct Python expression - most flexible but requires care
        expr = condition_def.get("expression", "False")
        try:
            # Compile the expression for efficiency
            code = compile(expr, "<string>", "eval")

            # Create a function that evaluates the expression with context as locals
            def condition_func(context):
                # Make a copy of context to avoid mutations
                locals_dict = dict(context)
                return bool(eval(code, {"__builtins__": {}}, locals_dict))

            return condition_func
        except Exception as e:
            logger.error(f"Error compiling condition expression: {str(e)}")
            return lambda context: False

    elif condition_type == "field_equals":
        # Check if a field equals a specific value
        field = condition_def.get("field", "")
        value = condition_def.get("value")

        def equals_condition(context):
            # Handle nested fields with dot notation (e.g., "analyzer_result.complexity")
            parts = field.split(".")
            current = context
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return False
            return current == value

        return equals_condition

    elif condition_type == "field_contains":
        # Check if a field contains a specific value (for strings, lists)
        field = condition_def.get("field", "")
        value = condition_def.get("value")
        case_sensitive = condition_def.get("case_sensitive", False)

        def contains_condition(context):
            # Handle nested fields with dot notation
            parts = field.split(".")
            current = context
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return False

            # Handle different container types
            if isinstance(current, str):
                if case_sensitive:
                    return value in current
                return value.lower() in current.lower()
            elif isinstance(current, (list, tuple, set)):
                return value in current
            elif isinstance(current, dict):
                return value in current
            return False

        return contains_condition

    elif condition_type == "complex_condition":
        # Combine multiple conditions with AND/OR logic
        operator = condition_def.get("operator", "and").lower()
        conditions = condition_def.get("conditions", [])

        if not conditions:
            logger.error("Complex condition missing 'conditions' list")
            return lambda context: False

        # Create each sub-condition
        condition_funcs = []
        for cond in conditions:
            func = create_conditional_function(cond)
            if func:
                condition_funcs.append(func)

        if not condition_funcs:
            logger.error("No valid sub-conditions in complex condition")
            return lambda context: False

        # Combine conditions according to operator
        if operator == "and":

            def and_condition(context):
                return all(func(context) for func in condition_funcs)

            return and_condition
        elif operator == "or":

            def or_condition(context):
                return any(func(context) for func in condition_funcs)

            return or_condition
        else:
            logger.error(f"Unsupported operator: {operator}")
            return lambda context: False

    else:
        logger.error(f"Unsupported condition type: {condition_type}")
        return lambda context: False


def load_graph_from_yaml(yaml_content: Union[str, Dict]) -> Optional[GraphOrchestrator]:
    """
    Create a GraphOrchestrator from a YAML definition.

    Args:
        yaml_content: Either a YAML string or already parsed YAML dict

    Returns:
        GraphOrchestrator instance or None if parsing failed
    """
    try:
        # Parse YAML if it's a string
        if isinstance(yaml_content, str):
            graph_def = yaml.safe_load(yaml_content)
        else:
            graph_def = yaml_content

        # Extract basic graph info
        name = graph_def.get("name", "YAML Graph")
        description = graph_def.get("description", "Graph created from YAML definition")

        # Create the orchestrator
        orchestrator = GraphOrchestrator(name=name, description=description)

        # First pass: Create all nodes without conditional edges
        node_definitions = {}
        nodes = graph_def.get("nodes", [])

        for node_def in nodes:
            node_id = node_def.get("id")
            if not node_id:
                logger.error("Node missing required 'id' field")
                continue

            # Get node dependencies
            dependencies = node_def.get("dependencies", [])

            # Get node metadata
            metadata = node_def.get("metadata", {})

            # Get execution condition
            condition_def = node_def.get("condition")
            condition_func = None
            if condition_def:
                condition_func = create_conditional_function(condition_def)

            # Create the step/agent for this node
            agent_config = node_def.get("agent", {})
            if not agent_config:
                logger.error(f"Node {node_id} missing 'agent' configuration")
                continue

            step = create_agent_from_config(agent_config)
            if not step:
                logger.error(f"Failed to create agent for node {node_id}")
                continue

            # Create graph node definition
            graph_node = GraphNodeDefinition(
                id=node_id,
                step=step,
                dependencies=dependencies,
                metadata=metadata,
                condition=condition_func,
                conditional_edges=[],  # Will add these in the second pass
            )

            # Store for later use
            node_definitions[node_id] = graph_node

            # Add to orchestrator
            orchestrator.add_node(graph_node)

        # Second pass: Add conditional edges
        for node_def in nodes:
            node_id = node_def.get("id")
            if node_id not in node_definitions:
                continue

            # Add conditional edges if any
            conditional_edges = node_def.get("conditional_edges", [])
            for edge_def in conditional_edges:
                target_node = edge_def.get("target")
                description = edge_def.get("description", "")
                condition_def = edge_def.get("condition")

                if not target_node:
                    logger.error(
                        f"Conditional edge in node {node_id} missing 'target' field"
                    )
                    continue

                if target_node not in node_definitions:
                    logger.error(
                        f"Conditional edge target '{target_node}' does not exist"
                    )
                    continue

                if not condition_def:
                    logger.error(
                        f"Conditional edge in node {node_id} missing 'condition'"
                    )
                    continue

                # Create the condition function
                condition_func = create_conditional_function(condition_def)
                if not condition_func:
                    logger.error(
                        f"Failed to create condition for edge from {node_id} to {target_node}"
                    )
                    continue

                # Create conditional edge
                edge = ConditionalEdge(
                    target_node=target_node,
                    condition=condition_func,
                    description=description,
                )

                # Add to orchestrator
                orchestrator.add_conditional_edge(node_id, edge)

        # Validate the graph
        try:
            orchestrator.validate_graph()
        except ValueError as e:
            logger.error(f"Graph validation failed: {str(e)}")
            return None

        return orchestrator

    except Exception as e:
        logger.error(f"Error parsing graph YAML: {str(e)}")
        return None


def load_graph_from_yaml_file(file_path: str) -> Optional[GraphOrchestrator]:
    """
    Load a graph from a YAML file.

    Args:
        file_path: Path to the YAML file

    Returns:
        GraphOrchestrator instance or None if loading failed
    """
    try:
        with open(file_path, "r") as f:
            yaml_content = f.read()
        return load_graph_from_yaml(yaml_content)
    except Exception as e:
        logger.error(f"Error loading graph from file {file_path}: {str(e)}")
        return None
