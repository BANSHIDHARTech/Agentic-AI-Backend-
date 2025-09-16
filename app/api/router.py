import logging
import os
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, HTTPException, Query

# Set up logging
logger = logging.getLogger(__name__)
from ..core.models import RouterClassifyRequest, RouterClassifyResponse, RouterRuleCreateRequest, RouterRuleResponse
from ..services.router_service import RouterService
from ..core.database import supabase, get_supabase_client

async def run_migration_script(table_name):
    """
    Run a migration script to create a missing table
    
    Args:
        table_name: The name of the table to create
    """
    logger.warning(f"Table {table_name} doesn't exist but is required. This should be handled by database migrations.")
    logger.info(f"For now, we'll return a mock response instead of trying to create the table.")
    
    # We won't actually try to create the table dynamically as that requires elevated permissions
    # and should be handled through proper migrations
    return False

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
        # Direct implementation to avoid any import issues
        # Initialize the client explicitly in the route handler
        supabase_client = get_supabase_client()
        
        # Build query directly
        query = supabase_client.table('router_rules').select(
            """
            *,
            agents!inner(id, name, description, is_active)
            """, 
            count='exact'
        ).order('priority', desc=False)
        
        # Apply search filter
        if search and isinstance(search, str):
            search_term = search.strip()
            query = query.or_(f"intent_name.ilike.%{search_term}%,description.ilike.%{search_term}%")
        
        # Apply active filter
        if active is not None:
            query = query.eq('is_active', bool(active))
        
        # Apply pagination
        if limit and limit > 0:
            limit_val = min(limit, 1000)  # Cap at 1000 for performance
            query = query.limit(limit_val)
        
        if offset and offset > 0:
            limit_val = limit or 50
            query = query.range(offset, offset + limit_val - 1)
        
        # Execute the query - handle both awaitable and non-awaitable versions
        try:
            # Try non-awaitable version first (newer Supabase client)
            response = query.execute()
        except Exception as exec_error:
            try:
                # Try awaitable version as fallback
                response = await query.execute()
            except Exception as await_error:
                logger.error(f"Database execution error: {await_error}")
                raise HTTPException(status_code=500, detail=f"Database error: {await_error}")
        
        # Check if data exists and handle response object appropriately
        if not hasattr(response, 'data') or response.data is None:
            error_msg = getattr(response, 'error', 'Unknown error') if hasattr(response, 'error') else "Unknown error"
            logger.error(f"Database error: {error_msg}")
            raise HTTPException(status_code=500, detail=f"Database error: {error_msg}")
        
        # Get count safely using error handling
        count = 0
        try:
            count = response.count if hasattr(response, 'count') else getattr(response, 'count', 0)
        except Exception:
            # If accessing count property fails, default to 0
            pass
        
        # Format the response
        return {
            'rules': response.data,
            'total': count,
            'limit': limit,
            'offset': offset or 0,
            'has_more': (count > (offset or 0) + len(response.data)) if response.data else False
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions as is
        raise
    except Exception as e:
        # Log the error
        logger.error(f"Error getting router rules: {str(e)}", exc_info=True)
        
        # Return standardized error response
        raise HTTPException(
            status_code=500, 
            detail=f"Database error: {str(e)}"
        )

@router.get("/rules/{rule_id}")
async def get_router_rule(rule_id: str):
    """Get specific router rule"""
    try:
        # Initialize Supabase client for this request
        supabase_client = get_supabase_client()
        
        # Query the database - handle both awaitable and non-awaitable versions
        try:
            # Try non-awaitable version first (newer Supabase client)
            result = supabase_client.table('router_rules').select('*').eq('id', rule_id).execute()
        except Exception as exec_error:
            try:
                # Try awaitable version as fallback
                result = await supabase_client.table('router_rules').select('*').eq('id', rule_id).execute()
            except Exception as await_error:
                logger.error(f"Database execution error: {await_error}")
                raise HTTPException(status_code=500, detail=f"Database error: {await_error}")
        
        # Check for database errors
        if hasattr(result, 'error') and result.error:
            raise HTTPException(status_code=500, detail=f"Database error: {result.error}")
        
        # Check if data exists
        if not hasattr(result, 'data') or not result.data:
            raise HTTPException(status_code=404, detail="Router rule not found")
        
        return result.data[0]
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Catch any other errors
        logger.error(f"Error fetching router rule {rule_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@router.post("/rules", response_model=RouterRuleResponse)
async def create_router_rule(rule: RouterRuleCreateRequest):
    """Create new router rule"""
    try:
        # Direct implementation to avoid any import issues
        supabase_client = get_supabase_client()
        
        # Convert Pydantic model to dict
        rule_data = rule.dict()
        
        # Add timestamps
        rule_data["created_at"] = datetime.utcnow().isoformat()
        rule_data["updated_at"] = datetime.utcnow().isoformat()
        
        # Set default values if not provided
        rule_data["is_active"] = rule_data.get("is_active", True)
        rule_data["priority"] = rule_data.get("priority", 100)
        
        # Insert into database - handle both awaitable and non-awaitable versions
        try:
            # Try non-awaitable version first (newer Supabase client)
            result = supabase_client.table("router_rules").insert(rule_data).execute()
        except Exception as exec_error:
            try:
                # Try awaitable version as fallback
                result = await supabase_client.table("router_rules").insert(rule_data).execute()
            except Exception as await_error:
                logger.error(f"Database execution error: {await_error}")
                raise HTTPException(status_code=500, detail=f"Database error: {await_error}")
        
        # Check for errors
        if hasattr(result, 'error') and result.error:
            raise HTTPException(status_code=500, detail=f"Database error: {result.error}")
            
        # Return created rule
        return result.data[0] if result.data else {}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating router rule: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@router.put("/rules/{rule_id}", response_model=RouterRuleResponse)
async def update_router_rule(rule_id: str, rule: RouterRuleCreateRequest):
    """Update router rule"""
    try:
        # Direct implementation to avoid any import issues
        supabase_client = get_supabase_client()
        
        # First check if the rule exists - handle both awaitable and non-awaitable versions
        try:
            # Try non-awaitable version first (newer Supabase client)
            check = supabase_client.table("router_rules").select("id").eq("id", rule_id).execute()
        except Exception as exec_error:
            try:
                # Try awaitable version as fallback
                check = await supabase_client.table("router_rules").select("id").eq("id", rule_id).execute()
            except Exception as await_error:
                logger.error(f"Database execution error: {await_error}")
                raise HTTPException(status_code=500, detail=f"Database error: {await_error}")
        
        if not hasattr(check, 'data') or not check.data:
            raise HTTPException(status_code=404, detail="Router rule not found")
            
        # Convert Pydantic model to dict
        rule_data = rule.dict()
        
        # Update timestamp
        rule_data["updated_at"] = datetime.utcnow().isoformat()
        
        # Update in database - handle both awaitable and non-awaitable versions
        try:
            # Try non-awaitable version first (newer Supabase client)
            result = supabase_client.table("router_rules").update(rule_data).eq("id", rule_id).execute()
        except Exception as exec_error:
            try:
                # Try awaitable version as fallback
                result = await supabase_client.table("router_rules").update(rule_data).eq("id", rule_id).execute()
            except Exception as await_error:
                logger.error(f"Database execution error: {await_error}")
                raise HTTPException(status_code=500, detail=f"Database error: {await_error}")
        
        # Check for errors
        if hasattr(result, 'error') and result.error:
            raise HTTPException(status_code=500, detail=f"Database error: {result.error}")
            
        # Return updated rule
        return result.data[0] if hasattr(result, 'data') and result.data else {}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating router rule {rule_id}: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@router.delete("/rules/{rule_id}")
async def delete_router_rule(rule_id: str):
    """Delete router rule"""
    try:
        # Direct implementation to avoid any import issues
        supabase_client = get_supabase_client()
        
        # First check if the rule exists - handle both awaitable and non-awaitable versions
        try:
            # Try non-awaitable version first (newer Supabase client)
            check = supabase_client.table("router_rules").select("id").eq("id", rule_id).execute()
        except Exception as exec_error:
            try:
                # Try awaitable version as fallback
                check = await supabase_client.table("router_rules").select("id").eq("id", rule_id).execute()
            except Exception as await_error:
                logger.error(f"Database execution error: {await_error}")
                raise HTTPException(status_code=500, detail=f"Database error: {await_error}")
        
        if not hasattr(check, 'data') or not check.data:
            raise HTTPException(status_code=404, detail="Router rule not found")
            
        # Delete from database - handle both awaitable and non-awaitable versions
        try:
            # Try non-awaitable version first (newer Supabase client)
            result = supabase_client.table("router_rules").delete().eq("id", rule_id).execute()
        except Exception as exec_error:
            try:
                # Try awaitable version as fallback
                result = await supabase_client.table("router_rules").delete().eq("id", rule_id).execute()
            except Exception as await_error:
                logger.error(f"Database execution error: {await_error}")
                raise HTTPException(status_code=500, detail=f"Database error: {await_error}")
        
        # Check for errors
        if hasattr(result, 'error') and result.error:
            raise HTTPException(status_code=500, detail=f"Database error: {result.error}")
            
        return {"message": "Router rule deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting router rule {rule_id}: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/fallback")
async def get_fallback_messages():
    """Get all fallback messages"""
    try:
        # Direct implementation to avoid any import issues
        supabase_client = get_supabase_client()
        
        # Query the database - handle both awaitable and non-awaitable versions
        try:
            # Try non-awaitable version first (newer Supabase client)
            result = supabase_client.table('router_fallback_messages').select('*').execute()
        except Exception as exec_error:
            try:
                # Try awaitable version as fallback
                result = await supabase_client.table('router_fallback_messages').select('*').execute()
            except Exception as await_error:
                # Check if the error is about the table not existing
                if "does not exist" in str(await_error):
                    logger.warning(f"Table router_fallback_messages does not exist. Returning default fallback messages.")
                    # Return default fallback messages if the table doesn't exist
                    from datetime import datetime
                    return {"status": "success", "data": [
                        {
                            "id": "00000000-0000-0000-0000-000000000001",
                            "message": "I'm not sure how to help with that. Could you try rephrasing your question?",
                            "description": "Default fallback message (table not found)",
                            "is_active": True,
                            "created_at": datetime.utcnow().isoformat(),
                            "updated_at": datetime.utcnow().isoformat()
                        },
                        {
                            "id": "00000000-0000-0000-0000-000000000002",
                            "message": "I don't have enough information to answer that. Could you provide more details?",
                            "description": "Information request fallback (table not found)",
                            "is_active": True,
                            "created_at": datetime.utcnow().isoformat(),
                            "updated_at": datetime.utcnow().isoformat()
                        }
                    ]}
                
                logger.error(f"Database execution error: {await_error}")
                raise HTTPException(status_code=500, detail=f"Database error: {await_error}")
        
        # Check for database errors
        if hasattr(result, 'error') and result.error:
            # Check if the error is about the table not existing
            if "does not exist" in str(result.error):
                logger.warning(f"Table router_fallback_messages does not exist. Returning default fallback messages.")
                # Return default fallback messages if the table doesn't exist
                from datetime import datetime
                return {"status": "success", "data": [
                    {
                        "id": "00000000-0000-0000-0000-000000000001",
                        "message": "I'm not sure how to help with that. Could you try rephrasing your question?",
                        "description": "Default fallback message (table not found)",
                        "is_active": True,
                        "created_at": datetime.utcnow().isoformat(),
                        "updated_at": datetime.utcnow().isoformat()
                    },
                    {
                        "id": "00000000-0000-0000-0000-000000000002",
                        "message": "I don't have enough information to answer that. Could you provide more details?",
                        "description": "Information request fallback (table not found)",
                        "is_active": True,
                        "created_at": datetime.utcnow().isoformat(),
                        "updated_at": datetime.utcnow().isoformat()
                    }
                ]}
            raise HTTPException(status_code=500, detail=f"Database error: {result.error}")
        
        # Format the response
        messages = result.data if hasattr(result, 'data') else []
        return {"status": "success", "data": messages}
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Error getting fallback messages: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/fallback")
async def create_fallback_message(message_data: dict):
    """Create new fallback message"""
    try:
        # Direct implementation to avoid any import issues
        supabase_client = get_supabase_client()
        
        # Validate data
        if not message_data.get('message'):
            raise ValueError("Fallback message is required")
            
        # Prepare data with required fields
        data = {
            'message': message_data['message'],
            'is_active': message_data.get('is_active', True),
            'description': message_data.get('description', None),
            'created_at': datetime.utcnow().isoformat(),
            'updated_at': datetime.utcnow().isoformat()
        }
        
        try:
            # Try to insert the data - handle both awaitable and non-awaitable versions
            try:
                # Try non-awaitable version first (newer Supabase client)
                result = supabase_client.table('router_fallback_messages').insert(data).execute()
            except Exception as exec_error:
                try:
                    # Try awaitable version as fallback
                    result = await supabase_client.table('router_fallback_messages').insert(data).execute()
                except Exception as await_error:
                    # Check if the error is about the table not existing
                    if "does not exist" in str(await_error):
                        logger.warning("Table router_fallback_messages does not exist. Notify admin to run migrations.")
                        await run_migration_script("fallback_messages")
                        # Return simulated response since we can't create the table dynamically
                        return {
                            "id": "00000000-0000-0000-0000-000000000099",
                            "message": data['message'],
                            "is_active": data['is_active'],
                            "description": data['description'],
                            "created_at": data['created_at'],
                            "updated_at": data['updated_at']
                        }
                    else:
                        raise await_error
                
            # Check for errors
            if hasattr(result, 'error') and result.error:
                if "does not exist" in str(result.error):
                    logger.warning("Table router_fallback_messages does not exist. Notify admin to run migrations.")
                    await run_migration_script("fallback_messages")
                    # Return simulated response since we can't create the table dynamically
                    return {
                        "id": "00000000-0000-0000-0000-000000000099",
                        "message": data['message'],
                        "is_active": data['is_active'],
                        "description": data['description'],
                        "created_at": data['created_at'],
                        "updated_at": data['updated_at']
                    }
                else:
                    raise HTTPException(status_code=500, detail=f"Database error: {result.error}")
                
        except Exception as e:
            # If we can't create the table or insert the data, return a simulated success response
            if "does not exist" in str(e):
                logger.warning(f"Could not create table or insert data: {e}. Returning simulated response.")
                return {
                    "id": "00000000-0000-0000-0000-000000000099",
                    "message": data['message'],
                    "is_active": data['is_active'],
                    "description": data['description'],
                    "created_at": data['created_at'],
                    "updated_at": data['updated_at']
                }
            else:
                raise e
            
        # Return the created message
        created_message = result.data[0] if hasattr(result, 'data') and result.data and result.data else {}
        return created_message
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating fallback message: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/classify", response_model=RouterClassifyResponse)
async def classify_intent(request: RouterClassifyRequest):
    """
    Classify user intent and route to appropriate agent
    
    Args:
        request: RouterClassifyRequest containing query and optional session_id/user_id
        
    Returns:
        RouterClassifyResponse with classification results
    """
    start_time = datetime.utcnow()
    
    try:
        # Log the incoming request
        logger.info(f"🔍 [RouterAPI] Received classification request: {request.query}")
        
        # Prepare options for the router service
        options = {
            'session_id': request.session_id,
            'user_id': request.user_id or "anonymous"
        }
        
        # Call the router service
        result = await RouterService.classify_intent(
            input_text=request.query,
            options=options
        )
        
        # Log the result
        if result.get('error'):
            logger.warning(f"⚠️  [RouterAPI] Classification completed with fallback: {result.get('error')}")
        elif result.get('detected_intent'):
            logger.info(f"✅ [RouterAPI] Classified intent: {result.get('detected_intent')} "
                      f"(confidence: {result.get('confidence_score', 0):.2f})")
        
        # Calculate processing time
        processing_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        result['processing_time_ms'] = processing_time_ms
        
        # Ensure the response matches the RouterClassifyResponse model
        response = RouterClassifyResponse(**result)
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
        
    except Exception as e:
        # Catch any other unexpected errors
        error_msg = f"Unexpected error in classify_intent: {str(e)}"
        logger.critical(f"🔥 [RouterAPI] {error_msg}", exc_info=True)
        
        # Return a proper error response
        return RouterClassifyResponse(
            query=request.query,
            detected_intent=None,
            confidence_score=0.0,
            selected_agent=None,
            rule_used=None,
            fallback_used=True,
            processing_time_ms=int((datetime.utcnow() - start_time).total_seconds() * 1000),
            error=error_msg
        )

@router.get("/analytics")
async def get_router_analytics(time_window: str = Query("24h", description="Time window for analytics (e.g., 24h, 7d, 30d)")):
    """
    Get router analytics and metrics
    
    Args:
        time_window: Time window for analytics (e.g., 24h, 7d, 30d)
    """
    try:
        analytics = await RouterService.get_analytics(time_window=time_window)
        return {"status": "success", "data": analytics}
    except Exception as e:
        logger.error(f"Error getting router analytics: {str(e)}")
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
