from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum

# Enums for validation
class SourceType(str, Enum):
    TEXT = "text"
    URL = "url"
    FILE = "file"
    PDF = "pdf"

class WorkflowStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class FallbackCategory(str, Enum):
    GENERAL = "general"
    ERROR = "error"
    TIMEOUT = "timeout"

# Agent Models
class AgentModel(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field("", max_length=500)
    system_prompt: str = Field(..., min_length=1)
    model_name: str = Field(..., min_length=1)
    input_intents: List[str] = Field(default_factory=list)
    output_intents: List[str] = Field(default_factory=list)
    tool_id: Optional[str] = None
    is_active: bool = Field(default=True)

class AgentCreateRequest(AgentModel):
    pass

class AgentResponse(AgentModel):
    id: str
    created_at: datetime
    updated_at: Optional[datetime] = None

# Tool Models
class ToolModel(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field("", max_length=500)
    function_code: str = Field(..., min_length=1)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    is_active: bool = Field(default=True)

class ToolCreateRequest(ToolModel):
    pass

class ToolResponse(ToolModel):
    id: str
    created_at: datetime
    updated_at: Optional[datetime] = None

# Workflow Models
class WorkflowNode(BaseModel):
    id: str
    type: str
    data: Dict[str, Any]
    position: Dict[str, float] = Field(default_factory=dict)

class WorkflowEdge(BaseModel):
    id: str
    source: str
    target: str
    data: Optional[Dict[str, Any]] = Field(default_factory=dict)

class WorkflowDbModel(BaseModel):
    """Model that matches the actual database columns"""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field("", max_length=500)
    is_active: bool = Field(default=True)

class WorkflowModel(BaseModel):
    """Full workflow model with nodes and edges for API"""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field("", max_length=500)
    nodes: List[WorkflowNode]
    edges: List[WorkflowEdge]
    is_active: bool = Field(default=True)

    @validator('nodes')
    def validate_nodes(cls, v):
        if not v:
            raise ValueError('At least one node is required')
        return v

class WorkflowCreateRequest(WorkflowModel):
    """API request model that includes nodes and edges"""
    pass

class WorkflowResponse(WorkflowModel):
    id: str
    created_at: datetime
    updated_at: Optional[datetime] = None

class WorkflowRunModel(BaseModel):
    workflow_id: str = Field(..., min_length=1)
    input_data: Dict[str, Any] = Field(default_factory=dict)
    status: WorkflowStatus = Field(default=WorkflowStatus.PENDING)
    session_id: Optional[str] = None

class WorkflowRunRequest(BaseModel):
    workflow_id: str
    input: Dict[str, Any]
    session_id: Optional[str] = None

class WorkflowRunResponse(WorkflowRunModel):
    id: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    output_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

# Router/Commander Agent Models
class RouterRuleModel(BaseModel):
    intent_name: str = Field(..., min_length=1, max_length=100)
    keywords: List[str] = Field(default_factory=list)
    agent_id: str = Field(..., min_length=1)
    priority: int = Field(default=100, ge=1)
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    is_active: bool = Field(default=True)
    description: Optional[str] = Field("", max_length=500)

class RouterRuleCreateRequest(RouterRuleModel):
    pass

class RouterRuleResponse(RouterRuleModel):
    id: str
    created_at: datetime
    updated_at: Optional[datetime] = None

class IntentLogModel(BaseModel):
    user_query: str = Field(..., min_length=1)
    detected_intent: Optional[str] = None
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    selected_agent_id: Optional[str] = None
    selected_agent_name: Optional[str] = None
    rule_id: Optional[str] = None
    fallback_used: bool = Field(default=False)
    processing_time_ms: Optional[int] = Field(None, ge=0)
    session_id: Optional[str] = None
    user_id: Optional[str] = None

class IntentLogCreateRequest(IntentLogModel):
    pass

class IntentLogResponse(IntentLogModel):
    id: str
    created_at: datetime

class FallbackMessageModel(BaseModel):
    message: str = Field(..., min_length=10, max_length=500)
    category: FallbackCategory = Field(default=FallbackCategory.GENERAL)
    is_active: bool = Field(default=True)

class FallbackMessageCreateRequest(FallbackMessageModel):
    pass

class FallbackMessageResponse(FallbackMessageModel):
    id: str
    created_at: datetime
    updated_at: Optional[datetime] = None

class RouterMetricsModel(BaseModel):
    agent_id: str = Field(..., min_length=1)
    agent_name: str = Field(..., min_length=1)
    success_count: int = Field(default=0, ge=0)
    failure_count: int = Field(default=0, ge=0)
    total_executions: int = Field(default=0, ge=0)
    avg_response_time_ms: float = Field(default=0.0, ge=0.0)
    date_bucket: str = Field(default_factory=lambda: datetime.now().strftime('%Y-%m-%d'))

class RouterMetricsCreateRequest(RouterMetricsModel):
    pass

class RouterMetricsResponse(RouterMetricsModel):
    id: str
    created_at: datetime
    updated_at: Optional[datetime] = None

# Knowledge Base Models
class DocumentModel(BaseModel):
    content: str = Field(..., min_length=10, max_length=100000)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    source_type: SourceType = Field(default=SourceType.TEXT)
    source_reference: Optional[str] = Field("", max_length=500)
    chunk_index: int = Field(default=0, ge=0)
    total_chunks: int = Field(default=1, ge=1)
    session_id: Optional[str] = None
    user_id: Optional[str] = None

class DocumentCreateRequest(DocumentModel):
    pass

class DocumentResponse(DocumentModel):
    id: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    embedding: Optional[List[float]] = None

class KnowledgeQueryModel(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000)
    session_id: str = Field(..., min_length=1)
    user_id: str = Field(default="anonymous")
    limit: int = Field(default=10, ge=1, le=50)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)

class KnowledgeQueryRequest(KnowledgeQueryModel):
    pass

class KnowledgeSearchResult(BaseModel):
    document_id: str
    content: str
    similarity_score: float
    metadata: Dict[str, Any]
    source_type: SourceType
    source_reference: Optional[str]

class KnowledgeQueryResponse(BaseModel):
    query: str
    results: List[KnowledgeSearchResult]
    total_results: int
    processing_time_ms: float

# Workflow Streaming Models
class WorkflowStreamRequest(BaseModel):
    workflow_id: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)
    input: Dict[str, Any]

class WorkflowStepRequest(BaseModel):
    run_id: str = Field(..., min_length=1)
    node_id: str = Field(..., min_length=1)
    input: Optional[Dict[str, Any]] = Field(default_factory=dict)

class WorkflowStepResponse(BaseModel):
    """Response model for workflow step execution"""
    run_id: str
    node_id: str
    input_data: Optional[Dict[str, Any]] = None
    output_data: Dict[str, Any]
    success: bool = True
    error_message: Optional[str] = None
    executed_at: datetime = Field(default_factory=datetime.now)

# Router Classification Models
class RouterClassifyRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    session_id: Optional[str] = None
    user_id: Optional[str] = Field(default="anonymous")

class RouterClassifyResponse(BaseModel):
    query: str = Field(..., description="The original user query")
    detected_intent: Optional[str] = Field(
        None, 
        description="The detected intent name if classification was successful"
    )
    confidence_score: float = Field(
        0.0, 
        ge=0.0, 
        le=1.0,
        description="Confidence score of the classification (0.0 to 1.0)"
    )
    selected_agent: Optional[Dict[str, Any]] = Field(
        None,
        description="Information about the selected agent for this intent"
    )
    rule_used: Optional[Dict[str, Any]] = Field(
        None,
        description="Information about the rule that was matched"
    )
    fallback_used: bool = Field(
        False,
        description="Whether a fallback response was used (no matching intent found)"
    )
    processing_time_ms: float = Field(
        0.0,
        ge=0.0,
        description="Time taken to process the request in milliseconds"
    )
    error: Optional[str] = Field(
        None,
        description="Error message if the classification failed"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "query": "How do I reset my password?",
                "detected_intent": "password_reset",
                "confidence_score": 0.92,
                "selected_agent": {
                    "id": "agent-123",
                    "name": "Password Assistant",
                    "description": "Helps with account and password issues"
                },
                "rule_used": {
                    "id": "rule-456",
                    "intent_name": "password_reset",
                    "confidence_threshold": 0.7
                },
                "fallback_used": False,
                "processing_time_ms": 125.5,
                "error": None
            }
        }

# Generic API Response Models
class APIResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Any] = None
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    service: str
    version: str
    uptime: str
