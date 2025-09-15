"""
Workflow Nodes API

Handles CRUD operations for workflow nodes and connections.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4

from ...core.database import get_supabase_client
from ...core.models import APIResponse
from ...services.node_service import NodeService
from ...services.workflow_service import WorkflowService
from ...core.auth import get_current_user

router = APIRouter(prefix="/workflows/{workflow_id}/nodes", tags=["Workflow Nodes"])

@router.post("", response_model=APIResponse)
async def create_node(
    workflow_id: str,
    node_data: Dict[str, Any],
    current_user: Dict = Depends(get_current_user)
):
    """
    Create a new node in the workflow
    
    - **workflow_id**: ID of the workflow
    - **node_data**: Node configuration including type, position, and data
    """
    try:
        # Validate workflow exists and user has access
        workflow = await WorkflowService.get_workflow(workflow_id)
        if not workflow:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workflow {workflow_id} not found"
            )
        
        # Create the node
        node = await NodeService.create_node(
            node_type=node_data.get("type", "default"),
            position=node_data.get("position", {"x": 0, "y": 0}),
            data=node_data.get("data", {}),
            workflow_id=workflow_id,
            tool_id=node_data.get("tool_id")
        )
        
        return APIResponse(
            success=True,
            message="Node created successfully",
            data=node
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.put("/{node_id}", response_model=APIResponse)
async def update_node(
    workflow_id: str,
    node_id: str,
    updates: Dict[str, Any],
    current_user: Dict = Depends(get_current_user)
):
    """
    Update a node's data
    
    - **workflow_id**: ID of the workflow
    - **node_id**: ID of the node to update
    - **updates**: Fields to update
    """
    try:
        updated_node = await NodeService.update_node(
            node_id=node_id,
            updates=updates,
            workflow_id=workflow_id
        )
        
        if not updated_node:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Node {node_id} not found in workflow {workflow_id}"
            )
            
        return APIResponse(
            success=True,
            message="Node updated successfully",
            data=updated_node
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.post("/{node_id}/link-tool", response_model=APIResponse)
async def link_tool_to_node(
    workflow_id: str,
    node_id: str,
    tool_data: Dict[str, str],
    current_user: Dict = Depends(get_current_user)
):
    """
    Link a tool to a node
    
    - **workflow_id**: ID of the workflow
    - **node_id**: ID of the node to update
    - **tool_id**: ID of the tool to link
    """
    try:
        tool_id = tool_data.get("tool_id")
        if not tool_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Tool ID is required"
            )
            
        updated_node = await NodeService.link_tool_to_node(
            node_id=node_id,
            tool_id=tool_id,
            workflow_id=workflow_id
        )
        
        if not updated_node:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Node {node_id} or tool {tool_id} not found"
            )
            
        return APIResponse(
            success=True,
            message="Tool linked to node successfully",
            data=updated_node
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.post("/connections", response_model=APIResponse)
async def create_connection(
    workflow_id: str,
    connection_data: Dict[str, Any],
    current_user: Dict = Depends(get_current_user)
):
    """
    Create a connection between two nodes
    
    - **workflow_id**: ID of the workflow
    - **source**: Source node ID
    - **target**: Target node ID
    - **source_handle**: Handle ID on source node (default: 'output')
    - **target_handle**: Handle ID on target node (default: 'input')
    - **data**: Additional connection data
    """
    try:
        source = connection_data.get("source")
        target = connection_data.get("target")
        source_handle = connection_data.get("source_handle", "output")
        target_handle = connection_data.get("target_handle", "input")
        
        if not source or not target:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Source and target node IDs are required"
            )
            
        # Validate the connection
        is_valid, message = await NodeService.validate_connection(
            source_node_id=source,
            target_node_id=target,
            workflow_id=workflow_id,
            source_handle=source_handle,
            target_handle=target_handle
        )
        
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid connection: {message}"
            )
        
        # Create the connection
        supabase = get_supabase_client()
        connection_id = f"conn_{uuid4().hex}"
        
        connection = {
            "id": connection_id,
            "workflow_id": workflow_id,
            "source": source,
            "target": target,
            "source_handle": source_handle,
            "target_handle": target_handle,
            "data": connection_data.get("data", {}),
            "created_by": current_user.get("id")
        }
        
        result = await supabase.table("workflow_edges").insert(connection).execute()
        
        if not result.data:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create connection"
            )
            
        return APIResponse(
            success=True,
            message="Connection created successfully",
            data=result.data[0] if result.data else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.get("/{node_id}/connections", response_model=APIResponse)
async def get_node_connections(
    workflow_id: str,
    node_id: str,
    direction: str = "both",
    current_user: Dict = Depends(get_current_user)
):
    """
    Get connections for a node
    
    - **workflow_id**: ID of the workflow
    - **node_id**: ID of the node
    - **direction**: 'input', 'output', or 'both' (default: 'both')
    """
    try:
        if direction not in ["input", "output", "both"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Direction must be 'input', 'output', or 'both'"
            )
            
        connections = await NodeService.get_node_connections(
            node_id=node_id,
            workflow_id=workflow_id,
            direction=direction
        )
        
        return APIResponse(
            success=True,
            message="Node connections retrieved successfully",
            data=connections
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.put("/{node_id}/data", response_model=APIResponse)
async def update_node_data(
    workflow_id: str,
    node_id: str,
    data: Dict[str, Any],
    current_user: Dict = Depends(get_current_user)
):
    """
    Update a node's data
    
    - **workflow_id**: ID of the workflow
    - **node_id**: ID of the node to update
    - **data**: Data to update
    """
    try:
        # Get current node data
        supabase = get_supabase_client()
        result = await supabase.table("workflow_nodes")\
            .select("*")\
            .eq("id", node_id)\
            .eq("workflow_id", workflow_id)\
            .single()\
            .execute()
            
        if not result.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Node {node_id} not found in workflow {workflow_id}"
            )
            
        # Update node data
        current_data = result.data.get("data", {})
        updated_data = {**current_data, **data}
        
        result = await supabase.table("workflow_nodes")\
            .update({
                "data": updated_data,
                "updated_at": "now()",
                "updated_by": current_user.get("id")
            })\
            .eq("id", node_id)\
            .execute()
            
        if not result.data:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update node data"
            )
            
        return APIResponse(
            success=True,
            message="Node data updated successfully",
            data=result.data[0] if result.data else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
