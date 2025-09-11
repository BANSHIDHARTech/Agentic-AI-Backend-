from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from ..core.models import RouterClassifyRequest, RouterClassifyResponse, RouterRuleCreateRequest, RouterRuleResponse
from ..services.router_service import RouterService
from ..core.database import supabase

router = APIRouter()

@router.get("/")
async def router_info():
    """Base route for /api/router"""
    return {
        "status": "Router API is active",
        "endpoints": [
            "GET    /rules",
            "GET    /rules/:id",
            "POST   /rules",
            "PUT    /rules/:id",
            "DELETE /rules/:id",
            "GET    /fallback",
            "GET    /fallback/:id",
            "POST   /fallback",
            "PUT    /fallback/:id",
            "DELETE /fallback/:id",
            "POST   /classify",
            "GET    /analytics",
            "GET    /logs",
            "GET    /metrics",
            "POST   /test"
        ]
    }

@router.get("/rules")
async def get_router_rules(
    limit: Optional[int] = Query(None),
    offset: Optional[int] = Query(None),
    search: Optional[str] = Query(None),
    active: Optional[bool] = Query(None)
):
    """Get all router rules"""
    try:
        options = {
            "limit": limit,
            "offset": offset,
            "search": search,
            "active": active
        }
        result = await RouterService.get_router_rules(options)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/rules/{rule_id}")
async def get_router_rule(rule_id: str):
    """Get specific router rule"""
    try:
        result = supabase.from_('router_rules').select('*').eq('id', rule_id).execute()
        if result.error:
            raise Exception(result.error)
        
        if not result.data:
            raise HTTPException(status_code=404, detail="Router rule not found")
        
        return result.data[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/rules", response_model=RouterRuleResponse)
async def create_router_rule(rule: RouterRuleCreateRequest):
    """Create new router rule"""
    try:
        result = await RouterService.create_router_rule(rule.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.put("/rules/{rule_id}", response_model=RouterRuleResponse)
async def update_router_rule(rule_id: str, rule: RouterRuleCreateRequest):
    """Update router rule"""
    try:
        result = await RouterService.update_router_rule(rule_id, rule.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.delete("/rules/{rule_id}")
async def delete_router_rule(rule_id: str):
    """Delete router rule"""
    try:
        await RouterService.delete_router_rule(rule_id)
        return {"message": "Router rule deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/fallback")
async def get_fallback_messages():
    """Get all fallback messages"""
    try:
        result = await RouterService.get_fallback_messages()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/fallback")
async def create_fallback_message(message_data: dict):
    """Create new fallback message"""
    try:
        result = await RouterService.create_fallback_message(message_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/classify", response_model=RouterClassifyResponse)
async def classify_intent(request: RouterClassifyRequest):
    """Classify user intent and route to appropriate agent"""
    try:
        result = await RouterService.classify_intent(request.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analytics")
async def get_router_analytics():
    """Get router analytics and metrics"""
    try:
        result = await RouterService.get_analytics()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/logs")
async def get_router_logs(
    limit: Optional[int] = Query(100),
    offset: Optional[int] = Query(0)
):
    """Get router classification logs"""
    try:
        result = await RouterService.get_logs(limit, offset)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics")
async def get_router_metrics():
    """Get router performance metrics"""
    try:
        result = await RouterService.get_metrics()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/test")
async def test_router():
    """Test router functionality with sample queries"""
    try:
        result = await RouterService.test_router()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
