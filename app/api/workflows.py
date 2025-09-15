import asyncio
import json
import re
from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from ..core.models import (
    WorkflowCreateRequest, WorkflowResponse, WorkflowRunRequest, 
    WorkflowRunResponse, WorkflowStepRequest, WorkflowStreamRequest
)
from ..services.workflow_service import WorkflowService
from ..services.workflow_stream_service import WorkflowStreamService

router = APIRouter()

@router.get("/")
async def workflows_info():
    """Base route for /api/workflows"""
    return {
        "status": "Workflows API is active",
        "endpoints": [
            "GET    /",
            "POST   /run",
            "POST   /step", 
            "POST   /run/stream",
            "GET    /run/stream",
            "GET    /:id",
            "GET    /:id/runs",
            "GET    /runs/:runId"
        ],
        "description": "FSM-based workflow execution with real-time streaming support"
    }

@router.post("/", response_model=WorkflowResponse)
async def create_workflow(workflow: WorkflowCreateRequest):
    """Create workflow with nodes and edges"""
    try:
        result = await WorkflowService.create_workflow(workflow.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/all")
async def get_all_workflows():
    """Get all workflows"""
    try:
        workflows = await WorkflowService.get_all_workflows()
        return workflows
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{workflow_id}")
async def get_workflow(workflow_id: str):
    """Get workflow by ID"""
    try:
        workflow = await WorkflowService.get_workflow(workflow_id)
        return workflow
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/run", response_model=WorkflowRunResponse)
async def run_workflow(request: WorkflowRunRequest):
    """Run complete workflow (traditional JSON response)"""
    try:
        result = await WorkflowService.run_workflow(
            request.workflow_id, 
            request.input, 
            request.session_id
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/step")
async def execute_workflow_step(request: WorkflowStepRequest):
    """Execute single workflow step"""
    try:
        result = await WorkflowService.execute_workflow_step(
            request.run_id, 
            request.node_id, 
            request.input
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def sanitize_id(id_string: str) -> str:
    """Sanitize ID strings to prevent injection attacks"""
    return re.sub(r'[^a-zA-Z0-9-_]', '', str(id_string))

async def generate_workflow_stream(workflow_id: str, input_data: Dict[str, Any], session_id: str):
    """Generate SSE events for workflow execution"""
    
    # Sanitize inputs
    sanitized_workflow_id = sanitize_id(workflow_id)
    sanitized_session_id = sanitize_id(session_id)
    
    if not sanitized_workflow_id or not sanitized_session_id:
        yield {
            "event": "error",
            "data": json.dumps({
                "message": "Invalid workflow_id or session_id format",
                "timestamp": datetime.now().isoformat()
            })
        }
        return

    try:
        # Use async generator from WorkflowStreamService
        async for event_data in WorkflowStreamService.run_workflow_stream(
            sanitized_workflow_id, 
            input_data, 
            sanitized_session_id
        ):
            yield {
                "event": event_data.get("event", "data"),
                "data": json.dumps({
                    **event_data.get("payload", {}),
                    "timestamp": datetime.now().isoformat()
                })
            }
            
            # Add small delay to prevent overwhelming the client
            await asyncio.sleep(0.01)
            
    except Exception as error:
        yield {
            "event": "workflow_error",
            "data": json.dumps({
                "message": "Workflow execution failed",
                "error": str(error),
                "timestamp": datetime.now().isoformat()
            })
        }

@router.post("/run/stream")
async def stream_workflow_post(request: WorkflowStreamRequest):
    """Stream workflow execution with Server-Sent Events (POST)"""
    
    # Validate required parameters
    if not request.workflow_id:
        raise HTTPException(status_code=400, detail="workflow_id is required")
    
    if not request.session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    
    if not request.input:
        raise HTTPException(status_code=400, detail="input is required")

    return EventSourceResponse(
        generate_workflow_stream(request.workflow_id, request.input, request.session_id),
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control, Content-Type",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS"
        }
    )

@router.get("/run/stream")
async def stream_workflow_get(
    workflow_id: str = Query(..., description="Workflow ID to execute"),
    session_id: str = Query(..., description="Session ID for tracking"),
    input: str = Query("{}", description="Input data as JSON string")
):
    """Stream workflow execution with Server-Sent Events (GET for testing)"""
    
    # Parse input JSON
    try:
        parsed_input = json.loads(input) if input else {}
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid input JSON format")

    # Validate required parameters
    if not workflow_id:
        raise HTTPException(status_code=400, detail="workflow_id query parameter is required")
    
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id query parameter is required")

    return EventSourceResponse(
        generate_workflow_stream(workflow_id, parsed_input, session_id),
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control, Content-Type",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS"
        }
    )

@router.get("/{workflow_id}/runs")
async def get_workflow_runs(workflow_id: str):
    """Get workflow runs for a specific workflow"""
    try:
        runs = await WorkflowService.get_workflow_runs(workflow_id)
        return runs
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.get("/runs/{run_id}")
async def get_workflow_run(run_id: str):
    """Get specific workflow run details"""
    try:
        run = await WorkflowService.get_workflow_run(run_id)
        return run
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.get("/stream/status/{run_id}")
async def get_stream_status(run_id: str):
    """Get workflow execution status for streaming"""
    try:
        status = await WorkflowStreamService.get_stream_status(run_id)
        return status
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
