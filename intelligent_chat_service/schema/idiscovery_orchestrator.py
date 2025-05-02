from pydantic import BaseModel, Field
from typing import Literal, Any, List


class AnalyzerResponse(BaseModel):
    explanation: str
    analysis: Literal["simple", "complex", "ambiguous"]


class ClarificationResponse(BaseModel):
    explanation: str
    type: Literal["clarification", "suggestion"]
    question: str = Field(
        description="Question to ask user if the user query is ambiguous based on the context"
    )
    valid: bool = Field(
        description="Identifies if the user query is valid. By default it is false."
    )


class PlannerTaskArgs(BaseModel):
    key: str = Field(description="Key for the arguments")
    value: str = Field(description="Value for the argument.")


class PlannerTask(BaseModel):
    id: str = Field(description="Unique identifier for the task")
    description: str = Field(description="Description of what the task accomplishes")
    type: Literal["tool"] = Field(
        description="Type of task (currently only 'tool' is supported)"
    )
    execution_type: Literal["sequential", "parallel"] = Field(
        description="Whether this task can be executed in parallel with others or must be sequential"
    )
    tool: str = Field(description="Name of the tool from the list of available tools")
    args: List[PlannerTaskArgs] = Field(
        description="Arguments to be passed to the tool when executing this task"
    )
    capabilities: List[str] = Field(
        description="Specific capabilities of the tool that will be used in this task"
    )
    dependencies: List[str] = Field(
        description="IDs of tasks that must be completed before this task can start",
        default_factory=list,
    )


class PlannerResponse(BaseModel):
    explanation: str = Field(description="Explanation of the overall plan")
    tasks: List[PlannerTask] = Field(description="List of tasks to execute the plan")


class ExecutionResponse(BaseModel):
    summary: str = Field(description="Summary for execution task results")
    execution_status: str = Field(description="Status of execution")
    results: dict[str, Any] = Field(
        description="Dictionary of task IDs and their results"
    )
