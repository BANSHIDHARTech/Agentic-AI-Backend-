import asyncio
import os
import signal
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from dotenv import load_dotenv

# Import API routers
from app.api.agents import router as agents_router
from app.api.tools import router as tools_router
from app.api.workflows import router as workflows_router
from app.api.logs import router as logs_router
from app.api.router import router as router_router
from app.api.knowledge import router as knowledge_router

# Import core modules
from app.core.middleware import error_handler, request_logger
from app.core.database import test_connection

# Import services for initialization
from app.services.tool_service import ToolService
from app.services.scheduler_service import SchedulerService

# Load environment variables
load_dotenv()

# Global variables for graceful shutdown
shutdown_event = asyncio.Event()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    # Startup
    print("🚀 Starting AgentFlow Backend Server...")
    
    # Test database connection
    print("📊 Testing database connection...")
    db_connected = await test_connection()
    if not db_connected:
        print("❌ Database connection failed")
        raise RuntimeError("Database connection failed")
    print("✅ Database connection successful")

    # Register builtin tools
    print("🔧 Registering builtin tools...")
    try:
        registered_tools = await ToolService.register_builtin_tools()
        print(f"✅ Registered {len(registered_tools)} builtin tools")
    except Exception as error:
        print(f"⚠️  Failed to register builtin tools: {error}")

    # Initialize scheduler service
    print("⏰ Initializing scheduler service...")
    try:
        await SchedulerService.initialize()
        print("✅ Scheduler service initialized")
    except Exception as error:
        print(f"⚠️  Failed to initialize scheduler: {error}")

    print("🎉 AgentFlow Backend Server Started Successfully!")
    print("=" * 60)
    
    yield
    
    # Shutdown
    print("🛑 Shutting down gracefully...")
    try:
        await SchedulerService.shutdown()
        print("✅ Scheduler service shut down successfully")
    except Exception as e:
        print(f"⚠️  Error during scheduler shutdown: {e}")
    print("✅ Shutdown complete")

# Create FastAPI app with lifespan
app = FastAPI(
    title="AgentFlow Backend",
    description="AI Agent Flow Management System",
    version="1.0.0",
    lifespan=lifespan,
    swagger_ui_parameters={"defaultModelsExpandDepth": -1, "deepLinking": True, "displayRequestDuration": True}
)

# CORS configuration with SSE support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "Cache-Control"],
    expose_headers=["Cache-Control", "Content-Type"]
)

# Add custom middleware
app.middleware("http")(request_logger)

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "AgentFlow Backend",
        "version": "1.0.0",
        "uptime": "N/A"  # Can be implemented with process start time tracking
    }

# Root route
@app.get("/")
async def root():
    return {
        "message": "🚀 AgentFlow Backend is running",
        "version": "1.0.0",
        "available_endpoints": [
            "/api",
            "/health",
            "/docs",
            "/redoc"
        ]
    }

# Base API route
@app.get("/api")
async def api_info():
    return {
        "message": "AgentFlow API",
        "version": "1.0.0",
        "endpoints": [
            {"path": "/api/agents", "methods": ["GET", "POST"]},
            {"path": "/api/agents/{agent_id}", "methods": ["GET", "PUT", "DELETE"]},
            {"path": "/api/tools", "methods": ["GET", "POST"]},
            {"path": "/api/tools/{tool_id}", "methods": ["GET", "PUT", "DELETE"]},
            {"path": "/api/workflows", "methods": ["GET", "POST"]},
            {"path": "/api/workflows/{workflow_id}", "methods": ["GET", "PUT", "DELETE"]},
            {"path": "/api/workflows/{workflow_id}/run", "methods": ["POST"]},
            {"path": "/api/knowledge/search", "methods": ["POST"]},
            {"path": "/api/knowledge/documents", "methods": ["GET", "POST"]},
            {"path": "/api/knowledge/documents/{document_id}", "methods": ["GET", "DELETE"]},
            {"path": "/health", "methods": ["GET"]},
            {"path": "/docs", "methods": ["GET"]},
            {"path": "/redoc", "methods": ["GET"]}
        ]
    }

# Include API routers
app.include_router(agents_router, prefix="/api/agents", tags=["agents"])
app.include_router(tools_router, prefix="/api/tools", tags=["tools"])
app.include_router(workflows_router, prefix="/api/workflows", tags=["workflows"])
app.include_router(logs_router, prefix="/api/logs", tags=["logs"])
app.include_router(router_router, prefix="/api/router", tags=["router"])
app.include_router(knowledge_router, prefix="/api/knowledge", tags=["knowledge"])

# Legacy routes for backward compatibility (excluding knowledge router which is now primary at /knowledge)
app.include_router(agents_router, prefix="/agents", tags=["agents-legacy"])
app.include_router(tools_router, prefix="/tools", tags=["tools-legacy"])
app.include_router(workflows_router, prefix="/workflows", tags=["workflows-legacy"])
app.include_router(logs_router, prefix="/logs", tags=["logs-legacy"])
app.include_router(router_router, prefix="/router", tags=["router-legacy"])

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return await error_handler(request, exc)

# 404 handler
@app.get("/{path:path}")
async def catch_all(request: Request, path: str):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Route not found",
            "path": request.url.path,
            "method": request.method,
            "timestamp": datetime.now().isoformat(),
            "available_endpoints": ["/api", "/health"]
        }
    )

# Graceful shutdown signal handlers
def signal_handler(signum, frame):
    print(f"🛑 Signal {signum} received, shutting down gracefully...")
    shutdown_event.set()

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 3001))
    
    print(f"📊 Health check: http://localhost:{port}/health")
    print(f"🔧 API Base URL: http://localhost:{port}/api")
    print(f"🎯 Workflows API: http://localhost:{port}/api/workflows")
    print(f"🧠 Knowledge API: http://localhost:{port}/api/knowledge")
    print(f"📡 SSE Streaming: http://localhost:{port}/api/workflows/run/stream")
    print(f"🔍 Router API: http://localhost:{port}/api/router")
    print(f"📝 Interactive docs: http://localhost:{port}/docs")
    
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )
