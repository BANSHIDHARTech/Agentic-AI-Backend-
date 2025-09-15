from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid
from ..core.database import supabase, db_insert, db_update, db_select, db_delete
from ..core.models import WorkflowModel, WorkflowRunModel
from .agent_service import AgentService
import asyncio
import logging

logger = logging.getLogger(__name__)

class WorkflowService:
    """Workflow management service for FSM-based workflow execution with auto-save and node management"""
    
    # Auto-save configuration
    AUTO_SAVE_INTERVAL = 30  # seconds
    _auto_save_tasks = {}
    _auto_save_queue = asyncio.Queue()
    _auto_save_workers = 2
    _shutdown_event = asyncio.Event()

    @classmethod
    async def initialize(cls):
        """Initialize auto-save workers"""
        for _ in range(cls._auto_save_workers):
            asyncio.create_task(cls._auto_save_worker())

    @classmethod
    async def shutdown(cls):
        """Shutdown auto-save workers"""
        cls._shutdown_event.set()
        await asyncio.gather(
            *[task for task in cls._auto_save_tasks.values()],
            return_exceptions=True
        )

    @classmethod
    async def _auto_save_worker(cls):
        """Background worker for processing auto-save tasks"""
        while not cls._shutdown_event.is_set():
            try:
                workflow_id, nodes, edges, user_id = await asyncio.wait_for(
                    cls._auto_save_queue.get(), 
                    timeout=1.0
                )
                await cls._save_workflow_internal(
                    workflow_id, 
                    nodes, 
                    edges, 
                    user_id,
                    auto_save=True
                )
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Auto-save error for {workflow_id}: {e}")
            finally:
                cls._auto_save_queue.task_done()

    @classmethod
    async def queue_auto_save(
        cls, 
        workflow_id: str,
        nodes: List[Dict],
        edges: List[Dict],
        user_id: str
    ) -> None:
        """Queue a workflow for auto-saving"""
        if workflow_id in cls._auto_save_tasks:
            cls._auto_save_tasks[workflow_id].cancel()
        
        # Create a new task that will execute after the auto-save interval
        task = asyncio.create_task(
            cls._delayed_auto_save(workflow_id, nodes, edges, user_id)
        )
        cls._auto_save_tasks[workflow_id] = task

    @classmethod
    async def _delayed_auto_save(
        cls,
        workflow_id: str,
        nodes: List[Dict],
        edges: List[Dict],
        user_id: str
    ) -> None:
        """Wait for the auto-save delay and then queue the save"""
        try:
            await asyncio.sleep(cls.AUTO_SAVE_INTERVAL)
            await cls._auto_save_queue.put((workflow_id, nodes, edges, user_id))
        except asyncio.CancelledError:
            pass
        finally:
            cls._auto_save_tasks.pop(workflow_id, None)

    @classmethod
    async def _save_workflow_internal(
        cls, 
        workflow_id: str,
        nodes: List[Dict],
        edges: List[Dict],
        user_id: str,
        auto_save: bool = False
    ) -> Dict[str, Any]:
        """Internal method to save workflow data"""
        try:
            # Validate nodes and edges
            validated_data = {
                "nodes": nodes,
                "edges": edges,
                "last_updated": datetime.utcnow().isoformat(),
                "updated_by": user_id,
                "auto_save": auto_save
            }
            
            # Update database
            result = await db_update(
                "workflows",
                {"id": workflow_id},
                {"data": validated_data, "updated_at": "now()"}
            )
            
            logger.info(f"Workflow {'auto-' if auto_save else ''}saved: {workflow_id}")
            return result[0] if result else None
            
        except Exception as e:
            logger.error(f"Error saving workflow {workflow_id}: {str(e)}")
            raise

    @classmethod
    async def save_workflow(
        cls, 
        workflow_id: str,
        nodes: List[Dict],
        edges: List[Dict],
        user_id: str,
        auto_save: bool = False
    ) -> Dict[str, Any]:
        """Save workflow with auto-save support"""
        if auto_save:
            # Queue for auto-save
            await cls.queue_auto_save(workflow_id, nodes, edges, user_id)
            return {"status": "queued_for_auto_save"}
        
        # Immediate save
        return await cls._save_workflow_internal(
            workflow_id, nodes, edges, user_id, auto_save=False
        )

    @classmethod
    def validate_connection(
        cls,
        source_node: Dict,
        target_node: Dict,
        source_handle: str = "output",
        target_handle: str = "input"
    ) -> bool:
        """Validate if connection between nodes is allowed"""
        try:
            # Check if source node has the output handle
            if source_handle not in (source_node.get("data", {}).get("outputs", {}) or {}):
                return False
                
            # Check if target node has the input handle
            if target_handle not in (target_node.get("data", {}).get("inputs", {}) or {}):
                return False
                
            # Add custom validation logic here
            source_type = source_node.get("type")
            target_type = target_node.get("type")
            
            # Example: Don't allow connection from output to output or input to input
            if source_handle == target_handle:
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Connection validation error: {e}")
            return False

    @classmethod
    async def update_node_data(
        cls,
        workflow_id: str,
        node_id: str,
        updates: Dict[str, Any],
        user_id: str
    ) -> Dict[str, Any]:
        """Update node data and trigger auto-save"""
        try:
            # Get current workflow
            workflow = await cls.get_workflow(workflow_id)
            if not workflow:
                raise ValueError("Workflow not found")
                
            # Find and update node
            nodes = workflow.get("data", {}).get("nodes", [])
            updated = False
            
            for node in nodes:
                if node.get("id") == node_id:
                    # Update node data
                    node_data = node.get("data", {})
                    node_data.update(updates)
                    node["data"] = node_data
                    updated = True
                    break
                    
            if not updated:
                raise ValueError("Node not found")
                
            # Save changes
            edges = workflow.get("data", {}).get("edges", [])
            await cls.save_workflow(
                workflow_id=workflow_id,
                nodes=nodes,
                edges=edges,
                user_id=user_id,
                auto_save=True
            )
            
            return {"status": "success", "node_id": node_id}
            
        except Exception as e:
            logger.error(f"Error updating node {node_id}: {e}")
            raise

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
                node_id = current_node_id
                
                try:
                    # Log node start
                    await cls._log_workflow_step(
                        run_id=run_id,
                        node_id=node_id,
                        input_data=current_data,
                        output_data={"status": "starting"},
                        success=True
                    )
                    
                    # Execute current node
                    node_result = await cls._execute_workflow_node(node, current_data)
                    
                    execution_path.append({
                        "node_id": node_id,
                        "node_type": node["type"],
                        "input": current_data,
                        "output": node_result,
                        "timestamp": datetime.now().isoformat(),
                        "status": "completed"
                    })
                    
                    # Log successful completion
                    await cls._log_workflow_step(
                        run_id=run_id,
                        node_id=node_id,
                        input_data=current_data,
                        output_data=node_result,
                        success=True
                    )
                    
                    # Update current data with node output
                    current_data = node_result
                    
                except Exception as error:
                    error_msg = str(error)
                    logger.error(f"Error executing node {node_id}: {error_msg}")
                    
                    execution_path.append({
                        "node_id": node_id,
                        "node_type": node.get("type", "unknown"),
                        "input": current_data,
                        "error": error_msg,
                        "timestamp": datetime.now().isoformat(),
                        "status": "failed"
                    })
                    
                    # Log the failure
                    await cls._log_workflow_step(
                        run_id=run_id,
                        node_id=node_id,
                        input_data=current_data,
                        output_data={"error": error_msg},
                        success=False,
                        error_message=error_msg
                    )
                    
                    # Re-raise to be handled by the outer try/except
                    raise
                
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
        """Execute a single workflow node with support for multiple node types"""
        node_type = node["type"]
        node_data = node["data"] or {}
        
        try:
            if node_type == "agent":
                # Execute agent node
                agent_id = node_data.get("agent_id")
                if not agent_id:
                    raise ValueError("Agent ID not specified in agent node")
                
                agent = await AgentService.get_agent(agent_id)
                result = await AgentService.execute_agent(agent, {"input": input_data})
                
                return {
                    "type": "agent_response",
                    "agent_id": agent_id,
                    "response": result["output"],
                    "processing_time_ms": result["processing_time_ms"]
                }
            
            elif node_type == "llm":
                # Execute LLM node
                from langchain.chat_models import ChatOpenAI
                from langchain.schema import HumanMessage, SystemMessage
                
                model_name = node_data.get("model", "gpt-3.5-turbo")
                temperature = float(node_data.get("temperature", 0.7))
                system_prompt = node_data.get("system_prompt", "")
                user_prompt = node_data.get("prompt_template", "{input}")
                
                # Format prompt with input data
                try:
                    formatted_prompt = user_prompt.format(**input_data)
                except KeyError as e:
                    raise ValueError(f"Missing required input variable in prompt: {e}")
                
                # Initialize LLM
                llm = ChatOpenAI(
                    model_name=model_name,
                    temperature=temperature,
                    openai_api_key=os.getenv("OPENAI_API_KEY")
                )
                
                # Prepare messages
                messages = []
                if system_prompt:
                    messages.append(SystemMessage(content=system_prompt))
                messages.append(HumanMessage(content=formatted_prompt))
                
                # Call LLM
                response = await llm.agenerate([messages])
                
                return {
                    "type": "llm_response",
                    "model": model_name,
                    "response": response.generations[0][0].text,
                    "usage": response.llm_output.get("token_usage", {}) if hasattr(response, 'llm_output') else {}
                }
            
            elif node_type in ["database", "sql"]:
                # Execute database query
                import json
                from ..core.database import execute_sql
                
                query_template = node_data.get("query", "")
                db_connection = node_data.get("connection_id") or "default"
                
                # Format query with input data
                try:
                    query = query_template.format(**input_data)
                except KeyError as e:
                    raise ValueError(f"Missing required variable in SQL query: {e}")
                
                # Execute query
                result = await execute_sql(query, db_connection)
                
                return {
                    "type": "query_result",
                    "query_type": "sql" if node_type == "sql" else "database",
                    "result": result,
                    "row_count": len(result) if isinstance(result, list) else 1
                }
            
            elif node_type == "api":
                # Execute API call
                import httpx
                
                url = node_data.get("url", "")
                method = node_data.get("method", "GET").upper()
                headers = node_data.get("headers", {})
                body_template = node_data.get("body", "")
                
                # Format URL and body with input data
                try:
                    formatted_url = url.format(**input_data)
                    if body_template:
                        if isinstance(body_template, str):
                            body = body_template.format(**input_data)
                            headers.setdefault("Content-Type", "application/json")
                        else:
                            # Handle JSON body templates
                            import json
                            body = json.dumps(body_template).format(**input_data)
                    else:
                        body = None
                except KeyError as e:
                    raise ValueError(f"Missing required variable in API call: {e}")
                
                # Make the HTTP request
                async with httpx.AsyncClient() as client:
                    response = await client.request(
                        method=method,
                        url=formatted_url,
                        headers=headers,
                        content=body,
                        timeout=30.0
                    )
                
                # Parse response
                try:
                    response_data = response.json()
                except ValueError:
                    response_data = response.text
                
                return {
                    "type": "api_response",
                    "status_code": response.status_code,
                    "response": response_data,
                    "headers": dict(response.headers)
                }
            
            elif node_type == "knowledge":
                # Execute knowledge base query
                from ..services.knowledge_service import KnowledgeService
                
                query = node_data.get("query", "")
                kb_id = node_data.get("kb_id")
                limit = int(node_data.get("limit", 5))
                
                if not kb_id:
                    raise ValueError("Knowledge base ID (kb_id) is required")
                
                # Format query with input data
                try:
                    formatted_query = query.format(**input_data)
                except KeyError as e:
                    raise ValueError(f"Missing required variable in knowledge query: {e}")
                
                # Query knowledge base
                results = await KnowledgeService.query(
                    kb_id=kb_id,
                    query=formatted_query,
                    limit=limit
                )
                
                return {
                    "type": "knowledge_result",
                    "kb_id": kb_id,
                    "query": formatted_query,
                    "results": results,
                    "result_count": len(results)
                }
            
            elif node_type == "condition":
                # Execute condition node
                condition = node_data.get("condition", "true")
                # In a real implementation, you'd evaluate the condition
                # For now, we'll just return the condition and input
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
                # Unknown node type, pass through with warning
                logger.warning(f"Unknown node type: {node_type}, passing through input data")
                return {
                    "type": "passthrough",
                    "input_data": input_data,
                    "node_type": node_type
                }
                
        except Exception as e:
            logger.error(f"Error in _execute_workflow_node ({node_type}): {str(e)}")
            raise
    
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
        """Log workflow step execution with enhanced context and status tracking
        
        Uses the database function log_workflow_step to handle status transitions and timestamps
        """
        try:
            # Determine status based on success flag
            status = "completed" if success else "failed"
            
            # Prepare execution context
            execution_context = {
                "input": input_data,
                "output": output_data,
                "metadata": {
                    "success": success,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            # Call the database function to handle the log
            result = await supabase.rpc(
                "log_workflow_step",
                {
                    "p_workflow_run_id": run_id,
                    "p_node_id": node_id,
                    "p_status": status,
                    "p_execution_context": execution_context,
                    "p_error_message": error_message
                }
            ).execute()
            
            if result.error:
                logger.error(f"Failed to log workflow step: {result.error}")
                
            return result.data[0] if result.data else None
                
        except Exception as error:
            logger.error(f"Error in _log_workflow_step: {error}")
            raise
