from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from ..services.log_service import LogService
from ..core.database import get_supabase_client

router = APIRouter()

@router.get("/")
async def get_logs(
    event_type: Optional[str] = Query(None),
    workflow_run_id: Optional[str] = Query(None),
    limit: Optional[int] = Query(100)
):
    """View run logs"""
    try:
        # Direct implementation to avoid any import issues
        supabase_client = get_supabase_client()
        
        # Build query
        query = supabase_client.table('logs').select('*').order('created_at', desc=True)
        
        # Apply filters
        if event_type:
            query = query.eq('event_type', event_type)
            
        if workflow_run_id:
            query = query.eq('workflow_run_id', workflow_run_id)
            
        if limit:
            query = query.limit(limit)
        
        # Execute query
        result = query.execute()
        
        # Check for database errors
        if hasattr(result, 'error') and result.error:
            raise HTTPException(status_code=500, detail=f"Database error: {result.error}")
            
        # Return the logs
        return result.data if hasattr(result, 'data') else []
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Log and wrap other errors
        import logging
        logging.error(f"Error getting logs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/workflow-run/{run_id}")
async def get_workflow_run_logs(run_id: str):
    """Get logs for specific workflow run"""
    try:
        logs = LogService.get_workflow_run_logs(run_id)
        return logs
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.get("/event-type/{event_type}")
async def get_logs_by_event_type(
    event_type: str,
    limit: Optional[int] = Query(100)
):
    """Get logs by event type"""
    try:
        logs = LogService.get_logs_by_event_type(event_type, limit)
        return logs
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/", status_code=201)
async def create_log(log_data: dict):
    """Create log entry"""
    try:
        event_type = log_data.get("event_type")
        details = log_data.get("details")
        workflow_run_id = log_data.get("workflow_run_id")
        
        result = LogService.create_log(event_type, details, workflow_run_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/stats")
async def get_log_stats():
    """Get log statistics"""
    try:
        stats = LogService.get_log_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/cleanup")
async def cleanup_old_logs(days: Optional[int] = Query(30)):
    """Delete old logs"""
    try:
        LogService.delete_old_logs(days)
        return {"message": f"Logs older than {days} days deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
