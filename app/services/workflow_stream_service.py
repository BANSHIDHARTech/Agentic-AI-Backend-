import asyncio
import json
import logging
from typing import Dict, Any, AsyncGenerator, Optional
from datetime import datetime
from .workflow_service import WorkflowService

# Configure logger
logger = logging.getLogger(__name__)

class WorkflowStreamService:
    """Workflow streaming service for real-time SSE execution"""
    
    @classmethod
    async def run_workflow_stream(
        cls,
        workflow_id: str,
        input_data: Dict[str, Any],
        session_id: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Run workflow with real-time streaming using async generator"""
        
        try:
            # Send start event
            yield {
                "event": "workflow_started",
                "payload": {
                    "workflow_id": workflow_id,
                    "session_id": session_id,
                    "input_data": input_data
                }
            }
            
            # Get workflow definition
            workflow = await WorkflowService.get_workflow(workflow_id)
            
            # Send workflow info
            yield {
                "event": "workflow_info",
                "payload": {
                    "name": workflow.get("name", "Unknown"),
                    "description": workflow.get("description", ""),
                    "total_nodes": len(workflow.get("nodes", []))
                }
            }
            
            # Execute workflow with streaming
            async for event in cls._execute_workflow_stream(workflow, input_data, session_id):
                yield event
            
            # Send completion event
            yield {
                "event": "workflow_completed",
                "payload": {
                    "workflow_id": workflow_id,
                    "session_id": session_id,
                    "completed_at": datetime.now().isoformat()
                }
            }
            
        except Exception as error:
            yield {
                "event": "workflow_error",
                "payload": {
                    "error": str(error),
                    "workflow_id": workflow_id,
                    "session_id": session_id
                }
            }
    
    @classmethod
    async def _execute_workflow_stream(
        cls,
        workflow: Dict[str, Any],
        input_data: Dict[str, Any],
        session_id: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute workflow with streaming events"""
        
        nodes = {node["id"]: node for node in workflow["nodes"]}
        edges = workflow["edges"]
        
        # Find start node
        incoming_edges = {edge["target"] for edge in edges}
        start_nodes = [node_id for node_id in nodes.keys() if node_id not in incoming_edges]
        
        if not start_nodes:
            yield {
                "event": "error",
                "payload": {"message": "No start node found in workflow"}
            }
            return
        
        current_node_id = start_nodes[0]
        current_data = input_data
        step_count = 0
        
        # Execute nodes with streaming
        while current_node_id:
            node = nodes[current_node_id]
            step_count += 1
            
            # Send node start event
            yield {
                "event": "node_started",
                "payload": {
                    "node_id": current_node_id,
                    "node_type": node["type"],
                    "step_number": step_count,
                    "input_data": current_data
                }
            }
            
            try:
                # Execute node with streaming
                async for node_event in cls._execute_node_stream(node, current_data):
                    yield node_event
                
                # Get final node result
                node_result = await WorkflowService._execute_workflow_node(node, current_data)
                
                # Send node completion event
                yield {
                    "event": "node_completed",
                    "payload": {
                        "node_id": current_node_id,
                        "node_type": node["type"],
                        "step_number": step_count,
                        "output_data": node_result
                    }
                }
                
                # Update current data
                current_data = node_result
                
                # Find next node
                next_edges = [edge for edge in edges if edge["source"] == current_node_id]
                
                if not next_edges:
                    # End of workflow
                    yield {
                        "event": "workflow_end",
                        "payload": {
                            "final_output": current_data,
                            "total_steps": step_count
                        }
                    }
                    break
                
                current_node_id = next_edges[0]["target"]
                
                # Add delay between nodes for better UX
                await asyncio.sleep(0.1)
                
            except Exception as error:
                yield {
                    "event": "node_error",
                    "payload": {
                        "node_id": current_node_id,
                        "error": str(error),
                        "step_number": step_count
                    }
                }
                break
    
    @classmethod
    async def _execute_node_stream(
        cls,
        node: Dict[str, Any],
        input_data: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute a single node with streaming events"""
        
        node_type = node["type"]
        node_data = node["data"]
        
        if node_type == "agent":
            # Stream agent execution
            yield {
                "event": "agent_thinking",
                "payload": {
                    "node_id": node["id"],
                    "agent_id": node_data.get("agent_id"),
                    "message": "Agent is processing your request..."
                }
            }
            
            # Simulate agent processing with progress updates
            for i in range(3):
                await asyncio.sleep(0.5)
                yield {
                    "event": "agent_progress",
                    "payload": {
                        "node_id": node["id"],
                        "progress": (i + 1) * 33,
                        "message": f"Processing step {i + 1}/3..."
                    }
                }
        
        elif node_type == "condition":
            yield {
                "event": "condition_evaluating",
                "payload": {
                    "node_id": node["id"],
                    "condition": node_data.get("condition", "true"),
                    "message": "Evaluating condition..."
                }
            }
            
            await asyncio.sleep(0.2)
        
        elif node_type == "transform":
            yield {
                "event": "transform_processing",
                "payload": {
                    "node_id": node["id"],
                    "transformation": node_data.get("transformation", {}),
                    "message": "Transforming data..."
                }
            }
            
            await asyncio.sleep(0.3)
        
        # Send node processing complete
        yield {
            "event": "node_processing_complete",
            "payload": {
                "node_id": node["id"],
                "node_type": node_type
            }
        }
    
    @classmethod
    async def get_stream_status(cls, run_id: str) -> Dict[str, Any]:
        """Get workflow execution status for streaming"""
        try:
            # Get workflow run details
            run = await WorkflowService.get_workflow_run(run_id)
            
            # Get step details
            from ..core.database import supabase
            
            # Use created_at instead of executed_at which doesn't exist
            try:
                steps_result = supabase.from_("workflow_steps").select("*").eq("workflow_run_id", run_id).order("created_at").execute()
                
                # Handle different response structures
                if hasattr(steps_result, 'data'):
                    steps = steps_result.data
                elif isinstance(steps_result, dict) and "data" in steps_result:
                    steps = steps_result["data"]
                else:
                    steps = []
                    
                # Check for errors in a way that works with the current Supabase structure
                error_occurred = False
                if hasattr(steps_result, 'error') and steps_result.error:
                    error_occurred = True
                    logger.error(f"Error fetching workflow steps: {steps_result.error}")
                
                if error_occurred:
                    steps = []
            except Exception as step_error:
                logger.error(f"Exception fetching workflow steps: {str(step_error)}")
                steps = []
            
            return {
                "run_id": run_id,
                "status": run.get("status", "unknown"),
                "workflow_id": run.get("workflow_id"),
                "session_id": run.get("session_id"),
                "started_at": run.get("started_at"),
                "completed_at": run.get("completed_at"),
                "total_steps": len(steps),
                "completed_steps": len([s for s in steps if s.get("success", False)]),
                "current_step": steps[-1] if steps else None,
                "error_message": run.get("error_message")
            }
            
        except Exception as error:
            return {
                "run_id": run_id,
                "status": "error",
                "error": str(error)
            }
