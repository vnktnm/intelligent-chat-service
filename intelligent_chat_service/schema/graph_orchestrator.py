from typing import Dict, Any, Optional, Callable, List, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from schema.orchestrator import Step
from datetime import datetime


class NodeStatus(Enum):
    """Status of a node in the graph orchestration."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass
class ExecutionStats:
    """Statistics for node execution."""

    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    error: Optional[str] = None


@dataclass
class ConditionalEdge:
    """Definition of a conditional edge between nodes."""

    target_node: str
    condition: Callable[[Dict[str, Any]], bool]
    priority: int
    description: str = ""


@dataclass
class GraphNodeDefinition:
    """Definition of a node in the orchestration graph."""

    id: str
    step: Step
    dependencies: List[str] = field(default_factory=list)
    condition: Optional[Callable[[Dict[str, Any]], bool]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    conditional_edges: List[ConditionalEdge] = field(default_factory=list)
    priority: int = 0


@dataclass
class GraphNode:
    """Runtime representation of a node in the orchestration graph."""

    id: str
    step: Step
    dependencies: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)
    status: NodeStatus = NodeStatus.PENDING
    stats: ExecutionStats = field(default_factory=ExecutionStats)
    condition: Optional[Callable[[Dict[str, Any]], bool]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    result: Any = None
    conditional_edges: List[ConditionalEdge] = field(default_factory=list)
    priority: int = 0


@dataclass
class GraphExecutionState:
    """The current state of graph execution."""

    nodes: Dict[str, GraphNode] = field(default_factory=dict)
    pending: Set[str] = field(default_factory=set)
    running: Set[str] = field(default_factory=set)
    completed: Set[str] = field(default_factory=set)
    error: Set[str] = field(default_factory=set)
    skipped: Set[str] = field(default_factory=set)
    execution_order: List[str] = field(default_factory=list)
    conditional_activations: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class GraphSummary:
    """Summary of the graph execution for visualization."""

    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    stats: Dict[str, Any]
