import os
import time
from datetime import datetime
from typing import Callable
from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global error handler for FastAPI"""
    logger.error(f"Error: {exc}")

    # Supabase errors (check for common error patterns)
    if hasattr(exc, 'code') or 'PGRST' in str(exc):
        return JSONResponse(
            status_code=400,
            content={
                "error": "Database error",
                "message": str(exc),
                "code": getattr(exc, 'code', None)
            }
        )

    # Validation errors
    if 'ValidationError' in str(type(exc)):
        return JSONResponse(
            status_code=400,
            content={
                "error": "Validation error",
                "message": str(exc)
            }
        )

    # HTTP exceptions
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail,
                "message": str(exc)
            }
        )

    # Default error
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc) if os.getenv("NODE_ENV") == "development" else "Something went wrong"
        }
    )

def validate_request(schema_validator: Callable):
    """Request validation decorator"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            try:
                # This would be used with Pydantic models in FastAPI
                # The validation is handled automatically by FastAPI when using Pydantic models
                return await func(*args, **kwargs)
            except Exception as err:
                raise HTTPException(status_code=400, detail=f"Validation failed: {str(err)}")
        return wrapper
    return decorator

def async_handler(func: Callable):
    """Async route handler wrapper (FastAPI handles this automatically)"""
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as exc:
            raise exc
    return wrapper

async def request_logger(request: Request, call_next):
    """Request logging middleware"""
    start_time = time.time()
    
    response = await call_next(request)
    
    duration = (time.time() - start_time) * 1000  # Convert to milliseconds
    logger.info(f"{request.method} {request.url.path} - {response.status_code} ({duration:.2f}ms)")
    
    return response

# Custom middleware for additional headers
async def add_security_headers(request: Request, call_next):
    """Add security headers to responses"""
    response = await call_next(request)
    
    # Add security headers (equivalent to helmet.js)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    
    return response
