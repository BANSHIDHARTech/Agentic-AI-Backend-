"""
Node Service

Handles dynamic tool linking, node creation, and node data management
for the workflow system.
"""

from typing import Dict, List, Optional, Any, Tuple
import logging
import uuid
from datetime import datetime

from ..core.database import get_supabase_client
from ..core.models import ToolModel
from .log_service import LogService

logger = logging.getLogger(__name__)

class NodeService:
    """Service for managing workflow nodes and dynamic tool integration"""
    
    @classmethod
    async def create_node(
        cls,
        node_type: str,
        position: Dict[str, float],
        data: Optional[Dict[str, Any]] = None,
        workflow_id: Optional[str] = None,
        tool_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a new workflow node
        
        Args:
            node_type: Type of node (e.g., 'tool', 'input', 'output', 'condition')
            position: Position of the node in the canvas {x: number, y: number}
            data: Optional node data
            workflow_id: Optional workflow ID this node belongs to
            tool_id: Optional tool ID if this is a tool node
            
        Returns:
            Created node dictionary
        """
        node_id = f"node_{uuid.uuid4().hex}"
        
        node = {
            "id": node_id,
            "type": node_type,
            "position": position,
            "data": data or {},
            "workflow_id": workflow_id,
            "tool_id": tool_id,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        # If this is a tool node, load tool configuration
        if tool_id and node_type == "tool":
            tool = await cls._load_tool_config(tool_id)
            if tool:
                node["data"]["tool"] = tool
                node["data"]["inputs"] = tool.get("inputs", {})
                node["data"]["outputs"] = tool.get("outputs", {})
        
        return node
    
    @classmethod
    async def update_node(
        cls,
        node_id: str,
        updates: Dict[str, Any],
        workflow_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Update node data
        
        Args:
            node_id: ID of the node to update
            updates: Dictionary of fields to update
            workflow_id: Optional workflow ID for validation
            
        Returns:
            Updated node or None if not found
        """
        try:
            supabase = get_supabase_client()
            
            # Get current node
            query = supabase.table("workflow_nodes").select("*").eq("id", node_id)
            if workflow_id:
                query = query.eq("workflow_id", workflow_id)
                
            result = await query.single().execute()
            
            if not result.data:
                return None
                
            node = result.data[0]
            
            # Update fields
            updated_node = {**node, **updates, "updated_at": datetime.utcnow().isoformat()}
            
            # Save to database
            result = await supabase.table("workflow_nodes")\
                .update(updated_node)\
                .eq("id", node_id)\
                .execute()
                
            return result.data[0] if result.data else None
            
        except Exception as e:
            logger.error(f"Error updating node {node_id}: {e}")
            await LogService.log(
                "node_error",
                {
                    "node_id": node_id,
                    "workflow_id": workflow_id,
                    "error": str(e),
                    "action": "update_node"
                }
            )
            raise
    
    @classmethod
    async def link_tool_to_node(
        cls,
        node_id: str,
        tool_id: str,
        workflow_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Link a tool to a node
        
        Args:
            node_id: ID of the node to update
            tool_id: ID of the tool to link
            workflow_id: Optional workflow ID for validation
            
        Returns:
            Updated node with tool configuration or None if not found
        """
        try:
            # Load tool configuration
            tool = await cls._load_tool_config(tool_id)
            if not tool:
                raise ValueError(f"Tool {tool_id} not found")
            
            # Update node with tool configuration
            updates = {
                "tool_id": tool_id,
                "type": "tool",
                "data": {
                    **tool,
                    "inputs": tool.get("inputs", {}),
                    "outputs": tool.get("outputs", {})
                },
                "updated_at": datetime.utcnow().isoformat()
            }
            
            return await cls.update_node(node_id, updates, workflow_id)
            
        except Exception as e:
            logger.error(f"Error linking tool {tool_id} to node {node_id}: {e}")
            await LogService.log(
                "tool_link_error",
                {
                    "node_id": node_id,
                    "tool_id": tool_id,
                    "workflow_id": workflow_id,
                    "error": str(e)
                }
            )
            raise
    
    @classmethod
    async def _load_tool_config(cls, tool_id: str) -> Optional[Dict[str, Any]]:
        """Load tool configuration from database or cache"""
        try:
            supabase = get_supabase_client()
            result = await supabase.table("tools").select("*").eq("id", tool_id).single().execute()
            
            if not result.data:
                return None
                
            tool = result.data[0]
            
            # Parse function code if needed
            if isinstance(tool.get("function_code"), str):
                try:
                    # Here you would compile/load the function code
                    # This is a placeholder - actual implementation depends on your requirements
                    tool["function"] = compile(tool["function_code"], "<string>", "exec")
                except Exception as e:
                    logger.error(f"Error compiling tool function: {e}")
                    tool["function"] = None
            
            return tool
            
        except Exception as e:
            logger.error(f"Error loading tool {tool_id}: {e}")
            return None
    
    @classmethod
    async def get_node_connections(
        cls,
        node_id: str,
        workflow_id: str,
        direction: str = "both"
    ) -> Dict[str, List[Dict]]:
        """
        Get connections for a node
        
        Args:
            node_id: ID of the node
            workflow_id: Workflow ID
            direction: 'input', 'output', or 'both'
            
        Returns:
            Dictionary with 'inputs' and 'outputs' lists
        """
        try:
            supabase = get_supabase_client()
            
            connections = {"inputs": [], "outputs": []}
            
            if direction in ["both", "input"]:
                # Get edges where this node is the target
                result = await supabase.table("workflow_edges")\
                    .select("*")\
                    .eq("workflow_id", workflow_id)\
                    .eq("target", node_id)\
                    .execute()
                connections["inputs"] = result.data or []
            
            if direction in ["both", "output"]:
                # Get edges where this node is the source
                result = await supabase.table("workflow_edges")\
                    .select("*")\
                    .eq("workflow_id", workflow_id)\
                    .eq("source", node_id)\
                    .execute()
                connections["outputs"] = result.data or []
            
            return connections
            
        except Exception as e:
            logger.error(f"Error getting connections for node {node_id}: {e}")
            return {"inputs": [], "outputs": []}
    
    @classmethod
    async def validate_connection(
        cls,
        source_node_id: str,
        target_node_id: str,
        workflow_id: str,
        source_handle: str = "output",
        target_handle: str = "input"
    ) -> Tuple[bool, str]:
        """
        Validate if a connection between nodes is allowed
        
        Args:
            source_node_id: ID of the source node
            target_node_id: ID of the target node
            workflow_id: Workflow ID
            source_handle: Handle ID on the source node
            target_handle: Handle ID on the target node
            
        Returns:
            Tuple of (is_valid, message)
        """
        try:
            supabase = get_supabase_client()
            
            # Get both nodes
            result = await supabase.table("workflow_nodes")\
                .select("*")\
                .in_("id", [source_node_id, target_node_id])\
                .execute()
            
            nodes = {node["id"]: node for node in result.data}
            
            if source_node_id not in nodes or target_node_id not in nodes:
                return False, "One or both nodes not found"
            
            source_node = nodes[source_node_id]
            target_node = nodes[target_node_id]
            
            # Check if source has the output handle
            source_outputs = source_node.get("data", {}).get("outputs", {})
            if source_handle not in source_outputs:
                return False, f"Source node does not have output handle '{source_handle}'"
            
            # Check if target has the input handle
            target_inputs = target_node.get("data", {}).get("inputs", {})
            if target_handle not in target_inputs:
                return False, f"Target node does not have input handle '{target_handle}'"
            
            # Check for circular connections
            if await cls._creates_circular_connection(
                workflow_id, source_node_id, target_node_id
            ):
                return False, "This would create a circular dependency"
            
            # Type checking (simplified example)
            source_type = source_outputs.get(source_handle, {}).get("type")
            target_type = target_inputs.get(target_handle, {}).get("type")
            
            if source_type != target_type:
                return False, f"Type mismatch: {source_type} != {target_type}"
            
            return True, "Connection is valid"
            
        except Exception as e:
            logger.error(f"Error validating connection: {e}")
            return False, f"Validation error: {str(e)}"
    
    @classmethod
    async def _creates_circular_connection(
        cls,
        workflow_id: str,
        source_node_id: str,
        target_node_id: str
    ) -> bool:
        """Check if adding an edge would create a circular dependency"""
        # This is a simplified implementation
        # In a real system, you'd want to do a proper graph traversal
        
        # Quick check for direct circular reference
        if source_node_id == target_node_id:
            return True
            
        # Get all edges in the workflow
        supabase = get_supabase_client()
        result = await supabase.table("workflow_edges")\
            .select("source, target")\
            .eq("workflow_id", workflow_id)\
            .execute()
        
        edges = result.data or []
        
        # Build adjacency list
        graph = {}
        for edge in edges:
            if edge["source"] not in graph:
                graph[edge["source"]] = []
            graph[edge["source"]].append(edge["target"])
        
        # Add the potential new edge
        if source_node_id not in graph:
            graph[source_node_id] = []
        graph[source_node_id].append(target_node_id)
        
        # Check for cycles using DFS
        visited = set()
        rec_stack = set()
        
        def is_cyclic_util(node):
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if is_cyclic_util(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        # Check all nodes in case the graph is disconnected
        for node in graph:
            if node not in visited:
                if is_cyclic_util(node):
                    return True
        
        return False
