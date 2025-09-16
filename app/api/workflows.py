import asyncio
import json
import os
import re
from datetime import datetime
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Query, Depends, Request
from ..core.database import _supabase, _initialized, get_supabase_client
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from ..core.models import (
    WorkflowCreateRequest, WorkflowResponse, WorkflowRunRequest, 
    WorkflowRunResponse, WorkflowStepRequest, WorkflowStreamRequest
)
from ..services.workflow_service import WorkflowService
from ..services.workflow_stream_service import WorkflowStreamService
import logging

# Configure logger
logger = logging.getLogger(__name__)

router = APIRouter()

# Global variables for fallback/mock data
MOCK_WORKFLOWS = [
    {
        "id": "mock-workflow-1",
        "name": "Sample Workflow 1",
        "description": "This is a sample workflow for testing",
        "is_active": True,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "nodes": [],
        "edges": []
    },
    {
        "id": "mock-workflow-2",
        "name": "Sample Workflow 2",
        "description": "Another sample workflow",
        "is_active": True,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "nodes": [],
        "edges": []
    }
]

@router.get("/")
async def workflows_info():
    """Base route for /api/workflows"""
    # Check environment variables
    env_vars = {
        "SUPABASE_URL": bool(os.getenv("SUPABASE_URL") or os.getenv("VITE_SUPABASE_URL")),
        "SUPABASE_KEY": bool(os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("VITE_SUPABASE_ANON_KEY")),
        "DB_INITIALIZED": _initialized if '_initialized' in globals() else False,
        "SUPABASE_CLIENT": "Initialized" if '_supabase' in globals() and _supabase is not None else "Not Initialized"
    }
    
    return {
        "status": "Workflows API is active",
        "environment": env_vars,
        "endpoints": [
            "GET    /",
            "GET    /all",
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
        # Pass the workflow model object directly instead of converting to dict here
        result = await WorkflowService.create_workflow(workflow)
        # Ensure the result has the required structure for the response model
        if "nodes" not in result:
            result["nodes"] = []
        if "edges" not in result:
            result["edges"] = []
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/test-db")
async def test_db_connection():
    """Test database connection and return status"""
    try:
        # Try to get a new client
        client = get_supabase_client()
        
        # Test the connection
        result = client.from_("workflows").select("id").limit(1).execute()
        
        # Check for errors in the result
        if hasattr(result, 'error') and result.error:
            return {
                "status": "error",
                "message": f"Database query failed: {result.error}",
                "connection": "Established but query failed"
            }
            
        return {
            "status": "success",
            "message": "Successfully connected to database",
            "data": result.data if hasattr(result, 'data') else "No data"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to connect to database: {str(e)}",
            "connection": "Failed"
        }

@router.get("/all")
async def get_all_workflows():
    """Get all workflows"""
    from datetime import datetime
    
    def create_mock_workflow(id_num: int):
        return {
            "id": f"mock-workflow-{id_num}",
            "name": f"Sample Workflow {id_num} (Mock Data)",
            "description": f"This is a sample workflow (mock data from workflows.py)",
            "is_active": True,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "nodes": [],
            "edges": []
        }
    
    try:
        logger.info("Attempting to get workflows from database")
        
        # Import our database module
        from ..core.database import get_supabase_client, supabase
        
        client = None
        try:
            # First try the global client
            logger.info(f"Global supabase client: {supabase}")
            logger.info(f"Global _supabase: {_supabase}")
            
            if supabase is not None:
                client = supabase
                logger.info("Using global Supabase client")
            else:
                # If global is None, try to get a fresh client
                logger.info("Global supabase client is None, getting a fresh client")
                try:
                    client = get_supabase_client()
                    logger.info(f"Got new client: {client is not None}")
                except Exception as e:
                    logger.error(f"Error getting Supabase client: {str(e)}")
                    client = None
                
            # Verify client is valid
            logger.info(f"Client before verification: {client}")
            if client is None:
                logger.error("Supabase client is None after initialization")
                # Try one more time with direct initialization
                try:
                    from supabase import create_client as create_supabase_client
                    supabase_url = os.getenv("SUPABASE_URL") or os.getenv("VITE_SUPABASE_URL")
                    supabase_key = os.getenv("SUPABASE_KEY") or os.getenv("VITE_SUPABASE_ANON_KEY")
                    logger.info(f"Direct init - URL: {bool(supabase_url)}, Key: {bool(supabase_key)}")
                    if supabase_url and supabase_key:
                        client = create_supabase_client(supabase_url, supabase_key)
                        logger.info("Direct client created")
                except Exception as e:
                    logger.error(f"Direct client creation failed: {str(e)}")
                
                if client is None:
                    logger.error("All client initialization attempts failed")
                    return [create_mock_workflow(1), create_mock_workflow(2)]
                
            # Test the connection
            try:
                logger.info("Testing Supabase connection...")
                test_result = client.from_("workflows").select("id").limit(1).execute()
                if hasattr(test_result, 'error') and test_result.error:
                    raise Exception(f"Test query failed: {test_result.error}")
                logger.info("Successfully connected to Supabase")
                
            except Exception as test_error:
                logger.error(f"Supabase connection test failed: {str(test_error)}")
                return [create_mock_workflow(1), create_mock_workflow(2)]
            
            # Query the workflows table
            logger.info("Querying workflows table")
            result = client.from_("workflows").select("*").execute()
            
            # Process workflows
            workflows = result.data if hasattr(result, 'data') else []
            
            # If no workflows found, return mock data
            if not workflows:
                logger.warning("No workflows found in database, returning mock data")
                return [create_mock_workflow(1), create_mock_workflow(2)]
            
            # Process each workflow
            for workflow in workflows:
                # Extract nodes and edges from data if they exist
                if "data" in workflow and workflow["data"] is not None:
                    if "nodes" in workflow["data"]:
                        workflow["nodes"] = workflow["data"]["nodes"]
                    if "edges" in workflow["data"]:
                        workflow["edges"] = workflow["data"]["edges"]
                
                # Ensure nodes and edges are always present
                if "nodes" not in workflow:
                    workflow["nodes"] = []
                if "edges" not in workflow:
                    workflow["edges"] = []
            
            logger.info(f"Successfully retrieved {len(workflows)} workflows")
            return workflows
            
        except Exception as query_error:
            logger.error(f"Error executing query: {str(query_error)}")
            logger.warning("Returning mock data due to query error")
            return [create_mock_workflow(1), create_mock_workflow(2)]
            
    except Exception as ve:
        logger.error(f"Unexpected error in get_all_workflows: {str(ve)}")
        return [create_mock_workflow(1), create_mock_workflow(2)]

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
