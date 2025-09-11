from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid
from ..core.database import supabase, db_insert, db_update, db_select, db_delete
from ..core.models import WorkflowModel, WorkflowRunModel
from .agent_service import AgentService

class WorkflowService:
    """Workflow management service for FSM-based workflow execution"""
    
    @classmethod
    async def create_workflow(cls, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new workflow"""
        # Validate using Pydantic model
        validated_data = WorkflowModel(**workflow_data).dict()
        
        result = await db_insert("workflows", validated_data)
        return result[0] if result else None
    
    @classmethod
    async def get_workflow(cls, workflow_id: str) -> Dict[str, Any]:
        """Get workflow by ID"""
        result = supabase.from_("workflows").select("*").eq("id", workflow_id).execute()
        
        if result.error:
            raise Exception(result.error)
        
        if not result.data:
            raise Exception(f"Workflow with ID {workflow_id} not found")
        
        return result.data[0]
    
    @classmethod
    async def get_all_workflows(cls) -> List[Dict[str, Any]]:
        """Get all workflows"""
        result = supabase.from_("workflows").select("*").eq("is_active", True).order("created_at", desc=True).execute()
        
        if result.error:
            raise Exception(result.error)
        
        return result.data or []
    
    @classmethod
    async def run_workflow(cls, workflow_id: str, input_data: Dict[str, Any], session_id: Optional[str] = None) -> Dict[str, Any]:
        """Run complete workflow (traditional JSON response)"""
        try:
            # Get workflow definition
            workflow = await cls.get_workflow(workflow_id)
            
            # Create workflow run record
            run_data = {
                "workflow_id": workflow_id,
                "input_data": input_data,
                "status": "running",
                "session_id": session_id or str(uuid.uuid4()),
                "started_at": datetime.now().isoformat()
            }
            
            run_result = await db_insert("workflow_runs", run_data)
            run_id = run_result[0]["id"] if run_result else None
            
            if not run_id:
                raise Exception("Failed to create workflow run record")
            
            # Execute workflow
            execution_result = await cls._execute_workflow_fsm(workflow, input_data, run_id)
            
            # Update run record with results
            await db_update("workflow_runs", run_id, {
                "status": "completed" if execution_result["success"] else "failed",
                "output_data": execution_result,
                "completed_at": datetime.now().isoformat(),
                "error_message": execution_result.get("error_message")
            })
            
            return {
                "run_id": run_id,
                "workflow_id": workflow_id,
                "status": "completed" if execution_result["success"] else "failed",
                "input_data": input_data,
                "output_data": execution_result,
                "session_id": session_id
            }
            
        except Exception as error:
            # Update run record with error
            if 'run_id' in locals():
                await db_update("workflow_runs", run_id, {
                    "status": "failed",
                    "error_message": str(error),
                    "completed_at": datetime.now().isoformat()
                })
            
            raise Exception(f"Workflow execution failed: {error}")
    
    @classmethod
    async def execute_workflow_step(cls, run_id: str, node_id: str, input_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute single workflow step"""
        try:
            # Get workflow run
            run_result = supabase.from_("workflow_runs").select("*").eq("id", run_id).execute()
            
            if run_result.error:
                raise Exception(run_result.error)
            
            if not run_result.data:
                raise Exception(f"Workflow run with ID {run_id} not found")
            
            run = run_result.data[0]
            
            # Get workflow definition
            workflow = await cls.get_workflow(run["workflow_id"])
            
            # Find the node
            node = None
            for n in workflow["nodes"]:
                if n["id"] == node_id:
                    node = n
                    break
            
            if not node:
                raise Exception(f"Node with ID {node_id} not found in workflow")
            
            # Execute the node
            step_result = await cls._execute_workflow_node(node, input_data or run["input_data"])
            
            # Log the step execution
            await cls._log_workflow_step(run_id, node_id, input_data, step_result)
            
            return {
                "run_id": run_id,
                "node_id": node_id,
                "input_data": input_data,
                "output_data": step_result,
                "success": True
            }
            
        except Exception as error:
            # Log the failed step
            await cls._log_workflow_step(run_id, node_id, input_data, {}, success=False, error_message=str(error))
            
            raise Exception(f"Workflow step execution failed: {error}")
    
    @classmethod
    async def get_workflow_runs(cls, workflow_id: str) -> List[Dict[str, Any]]:
        """Get workflow runs for a specific workflow"""
        result = supabase.from_("workflow_runs").select("*").eq("workflow_id", workflow_id).order("created_at", desc=True).execute()
        
        if result.error:
            raise Exception(result.error)
        
        return result.data or []
    
    @classmethod
    async def get_workflow_run(cls, run_id: str) -> Dict[str, Any]:
        """Get specific workflow run details"""
        result = supabase.from_("workflow_runs").select("*").eq("id", run_id).execute()
        
        if result.error:
            raise Exception(result.error)
        
        if not result.data:
            raise Exception(f"Workflow run with ID {run_id} not found")
        
        return result.data[0]
    
    @classmethod
    async def _execute_workflow_fsm(cls, workflow: Dict[str, Any], input_data: Dict[str, Any], run_id: str) -> Dict[str, Any]:
        """Execute workflow using Finite State Machine logic"""
        try:
            nodes = {node["id"]: node for node in workflow["nodes"]}
            edges = workflow["edges"]
            
            # Find start node (node with no incoming edges)
            incoming_edges = {edge["target"] for edge in edges}
            start_nodes = [node_id for node_id in nodes.keys() if node_id not in incoming_edges]
            
            if not start_nodes:
                raise Exception("No start node found in workflow")
            
            current_node_id = start_nodes[0]
            current_data = input_data
            execution_path = []
            
            # Execute nodes in sequence
            while current_node_id:
                node = nodes[current_node_id]
                
                # Execute current node
                node_result = await cls._execute_workflow_node(node, current_data)
                
                execution_path.append({
                    "node_id": current_node_id,
                    "node_type": node["type"],
                    "input": current_data,
                    "output": node_result,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Log the step
                await cls._log_workflow_step(run_id, current_node_id, current_data, node_result)
                
                # Update current data with node output
                current_data = node_result
                
                # Find next node
                next_edges = [edge for edge in edges if edge["source"] == current_node_id]
                
                if not next_edges:
                    # End of workflow
                    break
                
                # For simplicity, take the first edge (in a real FSM, you'd evaluate conditions)
                current_node_id = next_edges[0]["target"]
            
            return {
                "success": True,
                "execution_path": execution_path,
                "final_output": current_data,
                "total_steps": len(execution_path)
            }
            
        except Exception as error:
            return {
                "success": False,
                "error_message": str(error),
                "execution_path": execution_path if 'execution_path' in locals() else []
            }
    
    @classmethod
    async def _execute_workflow_node(cls, node: Dict[str, Any], input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single workflow node"""
        node_type = node["type"]
        node_data = node["data"]
        
        if node_type == "agent":
            # Execute agent node
            agent_id = node_data.get("agent_id")
            if not agent_id:
                raise Exception("Agent ID not specified in agent node")
            
            agent = await AgentService.get_agent(agent_id)
            result = await AgentService.execute_agent(agent, {"input": input_data})
            
            return {
                "type": "agent_response",
                "agent_id": agent_id,
                "response": result["output"],
                "processing_time_ms": result["processing_time_ms"]
            }
        
        elif node_type == "condition":
            # Execute condition node (simple evaluation)
            condition = node_data.get("condition", "true")
            # For simplicity, always return true (in real implementation, evaluate the condition)
            return {
                "type": "condition_result",
                "condition": condition,
                "result": True,
                "input_data": input_data
            }
        
        elif node_type == "transform":
            # Execute transformation node
            transformation = node_data.get("transformation", {})
            # Apply simple transformations
            transformed_data = input_data.copy()
            
            for key, value in transformation.items():
                if isinstance(value, str) and value.startswith("{{") and value.endswith("}}"):
                    # Simple template substitution
                    field_name = value[2:-2].strip()
                    if field_name in input_data:
                        transformed_data[key] = input_data[field_name]
                else:
                    transformed_data[key] = value
            
            return {
                "type": "transform_result",
                "transformed_data": transformed_data
            }
        
        else:
            # Unknown node type, pass through
            return {
                "type": "passthrough",
                "input_data": input_data,
                "node_type": node_type
            }
    
    @classmethod
    async def _log_workflow_step(
        cls,
        run_id: str,
        node_id: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        success: bool = True,
        error_message: Optional[str] = None
    ):
        """Log workflow step execution"""
        try:
            log_data = {
                "workflow_run_id": run_id,
                "node_id": node_id,
                "input_data": input_data,
                "output_data": output_data,
                "success": success,
                "error_message": error_message,
                "executed_at": datetime.now().isoformat()
            }
            
            result = supabase.from_("workflow_steps").insert(log_data).execute()
            if result.error:
                print(f"Failed to log workflow step: {result.error}")
                
        except Exception as error:
            print(f"Failed to log workflow step: {error}")
