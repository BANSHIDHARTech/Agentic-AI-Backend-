from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from ..services.log_service import LogService

router = APIRouter()

@router.get("/")
async def get_logs(
    event_type: Optional[str] = Query(None),
    workflow_run_id: Optional[str] = Query(None),
    limit: Optional[int] = Query(100)
):
    """View run logs"""
    try:
        filters = {
            "event_type": event_type,
            "workflow_run_id": workflow_run_id,
            "limit": limit
        }
        logs = await LogService.get_logs(filters)
        return logs
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/workflow-run/{run_id}")
async def get_workflow_run_logs(run_id: str):
    """Get logs for specific workflow run"""
    try:
        logs = await LogService.get_workflow_run_logs(run_id)
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
        logs = await LogService.get_logs_by_event_type(event_type, limit)
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
        
        result = await LogService.create_log(event_type, details, workflow_run_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/stats")
async def get_log_stats():
    """Get log statistics"""
    try:
        stats = await LogService.get_log_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/cleanup")
async def cleanup_old_logs(days: Optional[int] = Query(30)):
    """Delete old logs"""
    try:
        await LogService.delete_old_logs(days)
        return {"message": f"Logs older than {days} days deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
