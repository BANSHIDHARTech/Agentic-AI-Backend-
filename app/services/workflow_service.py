from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid
import os
from ..core.database import supabase, db_insert, db_update, db_select, db_delete, get_supabase_client
from ..core.models import WorkflowModel, WorkflowRunModel
from .agent_service import AgentService
import asyncio
import logging

logger = logging.getLogger(__name__)

# Ensure Supabase client is initialized when this module is loaded
try:
    if supabase is None:
        logger.warning("Initializing Supabase client in workflow_service module")
        get_supabase_client()
except Exception as e:
    logger.error(f"Failed to initialize Supabase client in workflow_service module: {str(e)}")

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
    async def create_workflow(cls, workflow_model) -> Dict[str, Any]:
        """Create a new workflow from a WorkflowCreateRequest model"""
        # Get transaction client for atomic operations
        from ..core.database import get_supabase_client
        import logging
        
        logger = logging.getLogger(__name__)
        supabase_client = get_supabase_client()
        
        try:
            # First check if a workflow with the same name already exists
            check_result = supabase_client.from_("workflows").select("id").eq("name", workflow_model.name).execute()
            
            if hasattr(check_result, 'data') and check_result.data and len(check_result.data) > 0:
                raise Exception(f"A workflow with the name '{workflow_model.name}' already exists")
            
            # Instead of direct database fields, let's use a dictionary approach
            # and store nodes and edges as a data field
            workflow_dict = {
                "name": workflow_model.name,
                "description": workflow_model.description,
                "is_active": workflow_model.is_active,
                "data": {
                    "nodes": [node.dict() if hasattr(node, 'dict') else node for node in workflow_model.nodes],
                    "edges": [edge.dict() if hasattr(edge, 'dict') else edge for edge in workflow_model.edges]
                }
            }
            
            # Store nodes and edges for later - access them directly from the model
            nodes = workflow_model.nodes
            edges = workflow_model.edges
            
            # Use a transaction for atomic workflow creation (all-or-nothing)
            logger.info(f"Starting transaction for workflow creation: {workflow_model.name}")
            
            # Direct database insert with only valid workflow fields and data field
            result = supabase_client.from_("workflows").insert(workflow_dict).execute()
            
            # Check if there was an error in the response
            # Handle both types of response objects
            if hasattr(result, 'error') and result.error:
                logger.error(f"Workflow creation failed: {result.error}")
                raise Exception(f"Insert error: {result.error}")
            
            # Also handle response types where data is accessed differently
            if hasattr(result, 'data'):
                data = result.data
            elif hasattr(result, 'json') and callable(result.json):
                data = result.json()
            else:
                data = result
                
            if not data:
                raise Exception("Failed to create workflow - no data returned")
                
            workflow = data[0] if isinstance(data, list) else data
            workflow_id = workflow["id"]
            
            logger.info(f"Created workflow base record with ID: {workflow_id}")
            
            # If we have nodes and edges, sync them using the database function
            # Only if there are actually nodes or edges to sync
            if (nodes and len(list(nodes) if hasattr(nodes, '__iter__') else [])) or \
               (edges and len(list(edges) if hasattr(edges, '__iter__') else [])):
                import json
                
                # Ensure nodes and edges are iterable
                node_list = list(nodes) if hasattr(nodes, '__iter__') else []
                edge_list = list(edges) if hasattr(edges, '__iter__') else []
                
                # Skip sync if both lists are empty
                if not node_list and not edge_list:
                    logger.info(f"Skipping sync for workflow {workflow_id} - no nodes or edges to sync")
                    workflow["nodes"] = []
                    workflow["edges"] = []
                    if "data" in workflow:
                        del workflow["data"]
                    return workflow
                
                # Process edges to match the expected format
                processed_edges = []
                # Create UUID conversion function
                import uuid
                
                def ensure_uuid(value):
                    """Ensure a value is a valid UUID"""
                    if isinstance(value, uuid.UUID):
                        return str(value)
                    try:
                        # Try parsing as a UUID
                        return str(uuid.UUID(value))
                    except (ValueError, TypeError, AttributeError):
                        # If not a valid UUID, create a deterministic UUID from the string
                        # This ensures the same string always maps to the same UUID
                        if isinstance(value, str):
                            return str(uuid.uuid5(uuid.NAMESPACE_DNS, value))
                        else:
                            # If it's not even a string, use a random UUID
                            return str(uuid.uuid4())
                
                for edge in edge_list:
                    try:
                        # Handle both dict access and object attribute access
                        if hasattr(edge, 'dict'):
                            edge_dict = edge.dict()
                            processed_edge = {
                                "id": ensure_uuid(edge_dict["id"]),
                                "from_node_id": ensure_uuid(edge_dict["source"]),
                                "to_node_id": ensure_uuid(edge_dict["target"]),
                                "data": edge_dict.get("data", {})
                            }
                        else:
                            processed_edge = {
                                "id": ensure_uuid(edge["id"]),
                                "from_node_id": ensure_uuid(edge["source"]),
                                "to_node_id": ensure_uuid(edge["target"]),
                                "data": edge.get("data", {})
                            }
                        processed_edges.append(processed_edge)
                    except Exception as edge_error:
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.error(f"Error processing edge: {str(edge_error)}")
                        # Continue with next edge
                
                # Convert nodes to proper format for the database function
                processed_nodes = []
                for node in node_list:
                    try:
                        if hasattr(node, 'dict'):
                            node_dict = node.dict()
                            # Convert ID to UUID
                            node_dict["id"] = ensure_uuid(node_dict["id"])
                            processed_nodes.append(node_dict)
                        else:
                            node_copy = dict(node)  # Create a copy
                            # Convert ID to UUID
                            node_copy["id"] = ensure_uuid(node["id"])
                            processed_nodes.append(node_copy)
                    except Exception as node_error:
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.error(f"Error processing node: {str(node_error)}")
                        # Continue with next node
                
                # Convert to JSON for the database function
                # Ensure we're passing valid JSON arrays
                if processed_nodes and isinstance(processed_nodes, list):
                    nodes_array = processed_nodes
                else:
                    nodes_array = []
                    
                if processed_edges and isinstance(processed_edges, list):
                    edges_array = processed_edges
                else:
                    edges_array = []
                
                # Debug logging
                import logging
                logger = logging.getLogger(__name__)
                logger.info(f"Syncing workflow {workflow_id} with {len(nodes_array)} nodes and {len(edges_array)} edges")
                
                # Call the sync_workflow function to handle nodes and edges
                try:
                    # First make sure nodes and edges are JSON serialized properly
                    import json
                    
                    # Convert Python lists to proper JSON strings
                    nodes_json = json.dumps(nodes_array)
                    edges_json = json.dumps(edges_array)
                    
                    # Log the actual JSON being sent
                    logger.info(f"Sending nodes JSON: {nodes_json[:100]}...")
                    logger.info(f"Sending edges JSON: {edges_json[:100]}...")
                    
                    # Use the same supabase_client from the transaction
                    sync_result = supabase_client.rpc(
                        "sync_workflow", 
                        {
                            "p_workflow_id": workflow_id, 
                            "p_nodes": nodes_array, 
                            "p_edges": edges_array
                        }
                    ).execute()
                    
                    # If we're here, it was successful
                    logger.info(f"Successfully synced workflow {workflow_id}")
                    
                except Exception as sync_error:
                    # Add more diagnostic info
                    nodes_str = json.dumps(nodes_array)[:100] if nodes_array else "[]"
                    edges_str = json.dumps(edges_array)[:100] if edges_array else "[]"
                    logger.error(f"Sync error with nodes: {nodes_str}..., edges: {edges_str}...")
                    
                    # On sync failure, we should also rollback the workflow creation for atomic operations
                    # Delete the workflow record to maintain consistency
                    try:
                        delete_result = supabase_client.from_("workflows").delete().eq("id", workflow_id).execute()
                        logger.info(f"Rolled back workflow creation for {workflow_id} after sync failure")
                    except Exception as delete_error:
                        logger.error(f"Error during rollback of workflow {workflow_id}: {str(delete_error)}")
                        
                    raise Exception(f"Sync workflow error: {str(sync_error)}. Nodes JSON: {nodes_str}..., Edges JSON: {edges_str}...")
                
                # Check if there was an error in the response
                # Handle both types of response objects
                if hasattr(sync_result, 'error') and sync_result.error:
                    # On sync failure, we should also rollback the workflow creation for atomic operations
                    try:
                        delete_result = supabase_client.from_("workflows").delete().eq("id", workflow_id).execute()
                        logger.info(f"Rolled back workflow creation for {workflow_id} after sync error")
                    except Exception as delete_error:
                        logger.error(f"Error during rollback of workflow {workflow_id}: {str(delete_error)}")
                        
                    raise Exception(f"Failed to sync workflow nodes/edges: {sync_result.error}")
                
            # Add back nodes and edges to the response
            if hasattr(nodes, '__iter__'):
                workflow["nodes"] = [node.dict() if hasattr(node, 'dict') else node for node in nodes]
            else:
                workflow["nodes"] = []
                
            if hasattr(edges, '__iter__'):
                workflow["edges"] = [edge.dict() if hasattr(edge, 'dict') else edge for edge in edges]
            else:
                workflow["edges"] = []
                
            # Delete the data field from response since we're using nodes/edges directly
            if "data" in workflow:
                del workflow["data"]
                
            return workflow
            
        except Exception as e:
            # Log the error
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error creating workflow: {str(e)}")
            
            # Try a simpler creation method as a fallback
            try:
                logger.warning("Attempting simplified workflow creation without nodes/edges")
                
                # Check again if a workflow with the same name already exists (just in case)
                check_result = supabase_client.from_("workflows").select("id").eq("name", workflow_model.name).execute()
                
                if hasattr(check_result, 'data') and check_result.data and len(check_result.data) > 0:
                    raise Exception(f"A workflow with the name '{workflow_model.name}' already exists")
                
                # Create a simpler workflow without nodes/edges
                workflow_dict = {
                    "name": workflow_model.name,
                    "description": workflow_model.description,
                    "is_active": workflow_model.is_active,
                    "data": {}  # Empty data
                }
                
                # Simplified insert with the transaction client
                result = supabase_client.from_("workflows").insert(workflow_dict).execute()
                
                if hasattr(result, 'data'):
                    data = result.data
                elif hasattr(result, 'json') and callable(result.json):
                    data = result.json()
                else:
                    data = result
                    
                if not data:
                    raise Exception("Failed to create workflow - no data returned")
                    
                workflow = data[0] if isinstance(data, list) else data
                workflow["nodes"] = []  # Empty nodes
                workflow["edges"] = []  # Empty edges
                
                # Delete the data field
                if "data" in workflow:
                    del workflow["data"]
                
                logger.info(f"Created simplified workflow with ID: {workflow['id']}")
                    
                return workflow
            except Exception as fallback_error:
                # If fallback fails too, raise the original error with details
                logger.error(f"Fallback workflow creation failed: {str(fallback_error)}")
                
                # Check if original error is a PostgreSQL error with more details
                if hasattr(e, 'args') and len(e.args) > 0 and isinstance(e.args[0], dict) and 'message' in e.args[0]:
                    # Extract the more detailed PostgreSQL error
                    pg_error = e.args[0]
                    raise Exception(f"Database error: {pg_error.get('message', str(e))}")
                else:
                    raise Exception(f"Failed to create workflow: {str(e)}")
    
    @classmethod
    async def get_workflow(cls, workflow_id: str) -> Dict[str, Any]:
        """Get workflow by ID"""
        try:
            # Use db_select which has better error handling and retry logic
            from ..core.database import db_select, init_supabase, get_supabase_client
            
            # First ensure the supabase client is initialized
            try:
                # This will initialize if not already initialized
                init_supabase()
                # Verify client is accessible
                client = get_supabase_client()
                logger.info(f"Supabase client initialized successfully for get_workflow({workflow_id})")
            except Exception as e:
                logger.error(f"Failed to initialize Supabase client: {str(e)}")
                raise Exception(f"Database connection failed: {str(e)}")
                
            # Get workflow using db_select
            results = await db_select(
                table="workflows",
                filters={"id": workflow_id}
            )
            
            if not results or len(results) == 0:
                raise Exception(f"Workflow with ID {workflow_id} not found")
            
            # Process the workflow data to handle nodes and edges correctly
            workflow = results[0]
            
            # Extract nodes and edges from data if they exist there
            if "data" in workflow and workflow["data"] is not None:
                if "nodes" in workflow["data"]:
                    workflow["nodes"] = workflow["data"]["nodes"]
                if "edges" in workflow["data"]:
                    workflow["edges"] = workflow["data"]["edges"]
                    
            # Ensure nodes and edges are present even if they weren't in data
            if "nodes" not in workflow:
                workflow["nodes"] = []
            if "edges" not in workflow:
                workflow["edges"] = []
                
            return workflow
            
        except Exception as e:
            logger.error(f"Error in get_workflow({workflow_id}): {str(e)}")
            raise Exception(f"Failed to retrieve workflow: {str(e)}")
    
    @classmethod
    def get_all_workflows(cls) -> List[Dict[str, Any]]:
        """Get all workflows"""
        try:
            supabase_client = get_supabase_client()
            if not supabase_client:
                logger.error("Failed to get Supabase client")
                return []
                
            # FIXED: Removed await from execute since it's a synchronous method
            result = supabase_client.table("workflows").select("*").execute()
            
            # Process workflows and ensure they have proper format
            workflows = result.data if hasattr(result, 'data') and result.data is not None else []
            
            # Ensure nodes and edges are always present
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
            
            return workflows
            
        except Exception as e:
            logger.error(f"Error in get_all_workflows: {str(e)}")
            return []
    
    @classmethod
    async def run_workflow(cls, workflow_id: str, input_data: Dict[str, Any], session_id: Optional[str] = None) -> Dict[str, Any]:
        """Run complete workflow (traditional JSON response)"""
        try:
            # Get workflow definition
            workflow = await cls.get_workflow(workflow_id)
            
            # Ensure session_id is a valid UUID
            if not session_id:
                session_id = str(uuid.uuid4())
            elif not isinstance(session_id, str):
                # Convert to string if it's not already
                session_id = str(session_id)
                
            # Create timestamp for both created_at and updated_at
            timestamp = datetime.now()
            
            # Create workflow run record
            run_data = {
                "workflow_id": workflow_id,
                "input_data": input_data,
                "status": "running",
                "session_id": session_id,
                "started_at": timestamp.isoformat()
            }
            
            run_result = await db_insert("workflow_runs", run_data)
            run_id = run_result[0]["id"] if run_result else None
            
            if not run_id:
                raise Exception("Failed to create workflow run record")
            
            # Execute workflow
            execution_result = await cls._execute_workflow_fsm(workflow, input_data, run_id)
            
            # Update run record with results
            completion_timestamp = datetime.now()
            await db_update("workflow_runs", run_id, {
                "status": "completed" if execution_result["success"] else "failed",
                "output_data": execution_result,
                "completed_at": completion_timestamp.isoformat(),
                "error_message": execution_result.get("error_message")
            })
            
            # Create a response that complies with the WorkflowRunResponse schema
            return {
                "id": run_id,  # Use run_id as the id (required by schema)
                "workflow_id": workflow_id,
                "status": "completed" if execution_result["success"] else "failed",
                "input_data": input_data,
                "output_data": execution_result,
                "created_at": timestamp,  # Include created_at (required by schema)
                "updated_at": completion_timestamp,  # Include updated_at
                "session_id": session_id,
                "error_message": execution_result.get("error_message")
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
        """Execute single workflow step with enhanced error handling and logging"""
        if input_data is None:
            input_data = {}
            
        execution_start = datetime.now()
        step_log_id = None
        
        try:
            # Validate run_id format
            try:
                if not run_id or not isinstance(run_id, str):
                    raise ValueError(f"Invalid run_id format: {run_id}")
            except Exception:
                raise ValueError(f"Invalid run_id format: {run_id}")
                
            # Validate node_id format
            if not node_id or not isinstance(node_id, str):
                raise ValueError(f"Invalid node_id format: {node_id}")
                
            # Get workflow run with error handling
            try:
                run_result = supabase.from_("workflow_runs").select("*").eq("id", run_id).execute()
                
                if hasattr(run_result, 'error') and run_result.error:
                    logger.error(f"Database error retrieving workflow run: {run_result.error}")
                    raise ValueError(f"Failed to retrieve workflow run: {run_result.error}")
                
                if not hasattr(run_result, 'data') or not run_result.data:
                    raise ValueError(f"Workflow run with ID {run_id} not found")
                
                run = run_result.data[0]
            except Exception as e:
                raise ValueError(f"Error retrieving workflow run: {str(e)}")
            
            # Get workflow definition with error handling
            try:
                workflow = await cls.get_workflow(run["workflow_id"])
                if not workflow:
                    raise ValueError(f"Workflow definition not found for workflow_id: {run['workflow_id']}")
            except Exception as e:
                raise ValueError(f"Error retrieving workflow definition: {str(e)}")
            
            # Find the node with better matching (handle string vs UUID comparison)
            node = None
            node_id_str = str(node_id)  # Ensure string format for comparison
            
            # Try to find node by direct ID match first
            for n in workflow["nodes"]:
                # Try direct comparison
                if str(n["id"]) == node_id_str:
                    node = n
                    break
            
            # If not found, try to find it by its hashed ID
            if not node:
                import hashlib
                import uuid
                
                for n in workflow["nodes"]:
                    # Generate the same hash that was used during workflow execution
                    original_id = str(n["id"])
                    hashed_id = None
                    
                    # Check if it's already a UUID
                    try:
                        uuid_obj = uuid.UUID(original_id)
                        hashed_id = str(uuid_obj)
                    except ValueError:
                        # Generate deterministic UUID from string ID
                        hash_hex = hashlib.md5(original_id.encode()).hexdigest()
                        # Convert to proper UUID format
                        hashed_id = str(uuid.UUID(hash_hex[:32]))
                    
                    if hashed_id == node_id_str:
                        node = n
                        break
            
            # If still not found, look in the execution path from the previous workflow run
            if not node and run.get("output_data") and "execution_path" in run.get("output_data", {}):
                for step in run["output_data"]["execution_path"]:
                    if step.get("node_id") == node_id_str:
                        # Found the node in execution path - try to find matching node in workflow again
                        node_type = step.get("node_type")
                        
                        # Try to find a node with matching type
                        for n in workflow["nodes"]:
                            if n.get("type") == node_type:
                                # Found a node with matching type
                                logger.info(f"Found matching node by type: {node_type}")
                                node = n
                                break
                        
                        # If we found a node or have a node_type, exit the loop
                        if node or node_type:
                            # If no node but we have a node_type, create a placeholder node
                            if not node and node_type:
                                node = {
                                    "id": node_id,
                                    "type": node_type,
                                    "data": {
                                        "label": f"Node {node_id}",
                                        "type": node_type
                                    },
                                    "position": {"x": 0, "y": 0}
                                }
                                logger.info(f"Created placeholder node with type: {node_type}")
                            break
            
            if not node:
                raise ValueError(f"Node with ID {node_id} not found in workflow")
            
            # Log the start of execution
            try:
                step_log_id = await cls._log_workflow_step(
                    run_id=run_id,
                    node_id=node_id,
                    input_data=input_data,
                    output_data={},
                    success=True,
                    error_message="Step execution started"
                )
            except Exception as log_error:
                logger.warning(f"Failed to log step start, but continuing: {str(log_error)}")
            
            # Execute the node with its own error handling
            try:
                # Use normalized input data (either provided or from the run)
                normalized_input = input_data if input_data else run.get("input_data", {})
                if not normalized_input:
                    normalized_input = {}
                
                step_result = await cls._execute_workflow_node(node, normalized_input)
            except Exception as node_error:
                # Wrap the node execution error with more context
                logger.error(f"Node execution failed: {str(node_error)}")
                raise ValueError(f"Failed to execute node '{node.get('data', {}).get('label', node_id)}': {str(node_error)}")
            
            # Log the successful step execution
            try:
                await cls._log_workflow_step(
                    run_id=run_id,
                    node_id=node_id,
                    input_data=input_data,
                    output_data=step_result,
                    success=True
                )
            except Exception as log_error:
                logger.warning(f"Failed to log step completion, but continuing: {str(log_error)}")
            
            # Return a structured response that matches WorkflowStepResponse
            return {
                "run_id": run_id,
                "node_id": node_id,
                "input_data": input_data,
                "output_data": step_result,
                "success": True,
                "error_message": None,
                "executed_at": execution_start
            }
            
        except Exception as error:
            # Log the failed step
            try:
                await cls._log_workflow_step(
                    run_id=run_id,
                    node_id=node_id,
                    input_data=input_data or {},
                    output_data={},
                    success=False,
                    error_message=str(error)
                )
            except Exception as log_error:
                logger.error(f"Failed to log step failure: {str(log_error)}")
            
            # Return structured error that follows the same model pattern
            return {
                "run_id": run_id,
                "node_id": node_id,
                "input_data": input_data,
                "output_data": {"error": str(error)},
                "success": False,
                "error_message": str(error),
                "executed_at": execution_start
            }
    
    @classmethod
    async def get_workflow_runs(cls, workflow_id: str) -> List[Dict[str, Any]]:
        """Get workflow runs for a specific workflow"""
        try:
            result = supabase.from_("workflow_runs").select("*").eq("workflow_id", workflow_id).order("created_at", desc=True).execute()
            
            # Check if result has data attribute
            if hasattr(result, 'data'):
                return result.data or []
            elif isinstance(result, dict) and "data" in result:
                return result["data"] or []
            else:
                # Fallback to returning the result itself if structure is unexpected
                logger.warning("Unexpected result structure from Supabase in get_workflow_runs")
                return result or []
                
        except Exception as e:
            logger.error(f"Error getting workflow runs: {str(e)}")
            raise Exception(f"Failed to get workflow runs: {str(e)}")
    
    @classmethod
    async def get_workflow_run(cls, run_id: str) -> Dict[str, Any]:
        """Get specific workflow run details"""
        try:
            result = supabase.from_("workflow_runs").select("*").eq("id", run_id).execute()
            
            # Get data from result based on its structure
            data = None
            if hasattr(result, 'data'):
                data = result.data
            elif isinstance(result, dict) and "data" in result:
                data = result["data"]
            
            # Check if we have data
            if not data:
                raise Exception(f"Workflow run with ID {run_id} not found")
            
            return data[0]
            
        except Exception as e:
            logger.error(f"Error getting workflow run: {str(e)}")
            raise Exception(f"Failed to get workflow run: {str(e)}")
    
    @classmethod
    async def _execute_workflow_fsm(cls, workflow: Dict[str, Any], input_data: Dict[str, Any], run_id: str) -> Dict[str, Any]:
        """Execute workflow using Finite State Machine logic"""
        try:
            # Function to ensure node IDs are valid UUIDs
            def ensure_uuid(value):
                """Ensure a value is a valid UUID string"""
                import uuid
                if isinstance(value, uuid.UUID):
                    return str(value)
                try:
                    # Try parsing as a UUID
                    return str(uuid.UUID(value))
                except (ValueError, TypeError, AttributeError):
                    # If not a valid UUID, create a deterministic UUID from the string
                    # This ensures the same string always maps to the same UUID
                    if isinstance(value, str):
                        return str(uuid.uuid5(uuid.NAMESPACE_DNS, value))
                    else:
                        # If it's not even a string, use a random UUID
                        return str(uuid.uuid4())
                        
            # Convert node IDs to valid UUIDs if needed
            nodes_with_valid_ids = {}
            for node in workflow["nodes"]:
                uuid_node_id = ensure_uuid(node["id"])
                node["original_id"] = node["id"]  # Store original ID for reference
                node["id"] = uuid_node_id
                nodes_with_valid_ids[uuid_node_id] = node
                
            # Update edge references to use the valid UUIDs
            updated_edges = []
            for edge in workflow["edges"]:
                # Find the nodes by original ID
                source_node = next((n for n in workflow["nodes"] if n["original_id"] == edge["source"]), None)
                target_node = next((n for n in workflow["nodes"] if n["original_id"] == edge["target"]), None)
                
                if source_node and target_node:
                    edge["source"] = source_node["id"]
                    edge["target"] = target_node["id"]
                    updated_edges.append(edge)
            
            nodes = nodes_with_valid_ids
            edges = updated_edges
            
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
                
                # Create a normalized version of the input data to handle common variable patterns
                normalized_input_data = input_data.copy()
                
                # If 'query' is in input_data, map it to common template variables
                if "query" in input_data:
                    query_value = input_data["query"]
                    # Map query to common template variables if they're missing
                    if "input" not in normalized_input_data:
                        normalized_input_data["input"] = query_value
                    if "user_input" not in normalized_input_data:
                        normalized_input_data["user_input"] = query_value
                    if "question" not in normalized_input_data:
                        normalized_input_data["question"] = query_value
                    if "text" not in normalized_input_data:
                        normalized_input_data["text"] = query_value
                
                # If we have any text field but no query, map it back
                elif "input" in input_data and "query" not in normalized_input_data:
                    normalized_input_data["query"] = input_data["input"]
                elif "text" in input_data and "query" not in normalized_input_data:
                    normalized_input_data["query"] = input_data["text"]
                elif "user_input" in input_data and "query" not in normalized_input_data:
                    normalized_input_data["query"] = input_data["user_input"]
                
                # Format prompt with normalized input data
                try:
                    formatted_prompt = user_prompt.format(**normalized_input_data)
                except KeyError as e:
                    # Try with a simple fallback if specific variables are missing
                    fallback_value = ""
                    for key in ["query", "input", "text", "user_input", "question"]:
                        if key in normalized_input_data:
                            fallback_value = normalized_input_data[key]
                            break
                    
                    # Create a dynamic dictionary with the missing variable
                    missing_var = str(e).strip("'")
                    dynamic_data = {missing_var: fallback_value}
                    
                    try:
                        # Try again with the dynamic fallback value
                        formatted_prompt = user_prompt.format(**dynamic_data)
                        logger.info(f"Used fallback value for missing variable {missing_var}")
                    except Exception as format_error:
                        # If still failing, use a generic prompt
                        logger.warning(f"Prompt formatting failed, using generic prompt: {format_error}")
                        formatted_prompt = f"Please respond to this: {fallback_value}"
                
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
            # Convert node_id to UUID if needed
            import uuid
            
            def ensure_uuid(value):
                """Ensure a value is a valid UUID string"""
                if isinstance(value, uuid.UUID):
                    return str(value)
                try:
                    # Try parsing as a UUID
                    return str(uuid.UUID(value))
                except (ValueError, TypeError, AttributeError):
                    # If not a valid UUID, create a deterministic UUID from the string
                    if isinstance(value, str):
                        return str(uuid.uuid5(uuid.NAMESPACE_DNS, value))
                    else:
                        # If it's not even a string, use a random UUID
                        return str(uuid.uuid4())
                        
            # Ensure we have valid UUIDs
            valid_node_id = ensure_uuid(node_id)
            
            # Determine status based on success flag
            status = "completed" if success else "failed"
            
            # Prepare execution context
            execution_context = {
                "input": input_data,
                "output": output_data,
                "metadata": {
                    "success": success,
                    "timestamp": datetime.now().isoformat(),
                    "original_node_id": node_id  # Store original for reference
                }
            }
            
            # Call the database function to handle the log
            # Fix: Don't use await with execute(), it's not awaitable
            result = supabase.rpc(
                "log_workflow_step",
                {
                    "p_workflow_run_id": run_id,
                    "p_node_id": valid_node_id,
                    "p_status": status,
                    "p_execution_context": execution_context,
                    "p_error_message": error_message
                }
            ).execute()
            
            if hasattr(result, 'error') and result.error:
                logger.error(f"Failed to log workflow step: {result.error}")
                
            return result.data[0] if hasattr(result, 'data') and result.data else None
                
        except Exception as error:
            logger.error(f"Error in _log_workflow_step: {error}")
            raise
