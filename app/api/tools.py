from fastapi import APIRouter, HTTPException
from ..core.models import ToolCreateRequest, ToolResponse
from ..services.tool_service import ToolService
from ..core.database import get_supabase_client

router = APIRouter()

@router.post("/", response_model=ToolResponse, status_code=201)
async def create_tool(tool: ToolCreateRequest):
    """Register a callable tool"""
    try:
        result = await ToolService.create_tool(tool.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/")
async def get_all_tools():
    """Get all tools"""
    try:
        # Direct implementation to avoid any import issues
        supabase_client = get_supabase_client()
        
        # Query the tools table for active tools
        result = supabase_client.table("tools").select("*").eq("is_active", True).order("created_at", desc=True).execute()
        
        # Check for database errors
        if hasattr(result, 'error') and result.error:
            raise HTTPException(status_code=500, detail=f"Database error: {result.error}")
        
        # Return the tools
        return result.data if hasattr(result, 'data') else []
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Log and wrap other errors
        import logging
        logging.error(f"Error getting all tools: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{tool_id}", response_model=ToolResponse)
async def get_tool(tool_id: str):
    """Get tool by ID"""
    try:
        # Direct implementation to avoid any import issues
        supabase_client = get_supabase_client()
        
        # Query the tools table for the specified ID
        result = supabase_client.table("tools").select("*").eq("id", tool_id).execute()
        
        # Check for database errors
        if hasattr(result, 'error') and result.error:
            raise HTTPException(status_code=500, detail=f"Database error: {result.error}")
        
        # Check if the tool exists
        if not hasattr(result, 'data') or not result.data:
            raise HTTPException(status_code=404, detail=f"Tool not found: {tool_id}")
            
        # Return the first matching tool
        return result.data[0]
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Log and wrap other errors
        import logging
        logging.error(f"Error getting tool {tool_id}: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))

@router.put("/{tool_id}", response_model=ToolResponse)
async def update_tool(tool_id: str, tool: ToolCreateRequest):
    """Update tool"""
    try:
        result = await ToolService.update_tool(tool_id, tool.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.delete("/{tool_id}")
async def delete_tool(tool_id: str):
    """Delete tool"""
    try:
        await ToolService.delete_tool(tool_id)
        return {"message": "Tool deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/{tool_id}/execute")
async def execute_tool(tool_id: str, execution_data: dict):
    """Execute tool"""
    try:
        result = await ToolService.execute_tool(tool_id, execution_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/register-builtin")
async def register_builtin_tools():
    """Register builtin tools"""
    try:
        result = await ToolService.register_builtin_tools()
        return {
            "message": "Builtin tools registered successfully",
            "tools": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
