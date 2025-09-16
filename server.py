import asyncio
import os
import signal
from contextlib import asynccontextmanager
from datetime import datetime
import logging

# Configure basic logging immediately
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables immediately, with override to ensure the latest values
from dotenv import load_dotenv
load_dotenv(override=True)

# Log environment variables for debugging
logger.info(f"SUPABASE_URL: {os.getenv('SUPABASE_URL')[:10] if os.getenv('SUPABASE_URL') else 'Not set'}")
logger.info(f"SUPABASE_KEY: {'Set' if os.getenv('SUPABASE_KEY') else 'Not set'}")
logger.info(f"VITE_SUPABASE_URL: {os.getenv('VITE_SUPABASE_URL')[:10] if os.getenv('VITE_SUPABASE_URL') else 'Not set'}")
logger.info(f"VITE_SUPABASE_ANON_KEY: {'Set' if os.getenv('VITE_SUPABASE_ANON_KEY') else 'Not set'}")

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Import API routers
from app.api.agents import router as agents_router
from app.api.tools import router as tools_router
from app.api.workflows import router as workflows_router
from app.api.logs import router as logs_router
from app.api.router import router as router_router
from app.api.knowledge import router as knowledge_router

# Import core modules
from app.core.middleware import error_handler, request_logger

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
    
    # Test database connection with more detailed information
    print("📊 Testing database connection...")
    try:
        # Direct database connection test without using problematic import
        from supabase import create_client
        
        # Force reload environment variables to get the latest values
        load_dotenv(override=True)
        
        # Log environment variable presence for debugging
        print(f"🔑 SUPABASE_URL: {'Found' if os.getenv('SUPABASE_URL') else 'Not found'}")
        print(f"🔑 SUPABASE_KEY: {'Found' if os.getenv('SUPABASE_KEY') else 'Not found'}")
        print(f"🔑 VITE_SUPABASE_URL: {'Found' if os.getenv('VITE_SUPABASE_URL') else 'Not found'}")
        print(f"🔑 VITE_SUPABASE_ANON_KEY: {'Found' if os.getenv('VITE_SUPABASE_ANON_KEY') else 'Not found'}")
        
        # Get Supabase credentials with multiple fallbacks
        supabase_url = os.getenv("SUPABASE_URL") or os.getenv("VITE_SUPABASE_URL")
        supabase_key = (os.getenv("SUPABASE_KEY") or 
                    os.getenv("SUPABASE_SERVICE_ROLE_KEY") or 
                    os.getenv("VITE_SUPABASE_ANON_KEY"))
        
        if not supabase_url or not supabase_key:
            print(f"⚠️ Missing Supabase credentials - app will run in mock data mode")
            app.state.supabase_client = None
        else:
            try:
                # Create client directly
                client = create_client(supabase_url, supabase_key)
                print("✅ Supabase client initialized successfully")
                
                # Store the client in app.state for reuse
                app.state.supabase_client = client
                
                # Test the client with a simple query
                test_result = client.from_("workflows").select("id").limit(1).execute()
                print(f"✅ Successfully tested Supabase client with a workflows query")
                print(f"✅ Database connection successful")
            except Exception as client_error:
                print(f"⚠️ Database client error: {str(client_error)}")
                print(f"⚠️ App will run in mock data mode")
                app.state.supabase_client = None
        
    except Exception as e:
        print(f"⚠️ Database connection setup error: {str(e)}")
        print(f"⚠️ App will run in mock data mode")
        app.state.supabase_client = None
    
    # Continue with startup regardless of database connection status
    
    # Skip vector search function check for now
    print("🔍 Vector search function check skipped")

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

# Create FastAPI app with lifespan and proper tags configuration
app = FastAPI(
    title="AgentFlow Backend",
    description="AI Agent Flow Management System",
    version="1.0.0",
    lifespan=lifespan,
    swagger_ui_parameters={"defaultModelsExpandDepth": -1, "deepLinking": True, "displayRequestDuration": True},
    openapi_tags=[
        {"name": "default", "description": "Default endpoints"},
        {"name": "agents", "description": "Agent management endpoints"},
        {"name": "tools", "description": "Tool management endpoints"},
        {"name": "workflows", "description": "Workflow management endpoints"},
        {"name": "logs", "description": "Log management endpoints"},
        {"name": "router", "description": "Router management endpoints"},
        {"name": "knowledge", "description": "Knowledge base endpoints"}
    ]
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
@app.get("/health", tags=["default"])
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "AgentFlow Backend",
        "version": "1.0.0",
        "uptime": "N/A"  # Can be implemented with process start time tracking
    }

# Root route
@app.get("/", tags=["default"])
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
@app.get("/api", tags=["default"])
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

# The /api/workflows/all endpoint is now handled by the workflows router

# Remove or comment out diagnostic endpoints since they're not in your desired Swagger UI
'''
@app.get("/api/diagnostics/db", tags=["diagnostics"])
async def get_database_diagnostics(request: Request):
    """Get database connection diagnostics"""
    # ...code removed...

@app.get("/api/diagnostics/env", tags=["diagnostics"])
async def get_environment_diagnostics():
    """Get environment variable diagnostics"""
    # ...code removed...
'''

# Remove direct_routes registration to avoid unexpected endpoints

# Include API routers with correct tags
app.include_router(agents_router, prefix="/api/agents", tags=["agents"])
app.include_router(tools_router, prefix="/api/tools", tags=["tools"])
app.include_router(workflows_router, prefix="/api/workflows", tags=["workflows"])
app.include_router(logs_router, prefix="/api/logs", tags=["logs"])
app.include_router(router_router, prefix="/api/router", tags=["router"])
app.include_router(knowledge_router, prefix="/api/knowledge", tags=["knowledge"])

# Legacy routes are hidden from documentation but still functional for backward compatibility
app.include_router(agents_router, prefix="/agents", tags=["agents-legacy"], include_in_schema=False)
app.include_router(tools_router, prefix="/tools", tags=["tools-legacy"], include_in_schema=False)
app.include_router(workflows_router, prefix="/workflows", tags=["workflows-legacy"], include_in_schema=False)
app.include_router(logs_router, prefix="/logs", tags=["logs-legacy"], include_in_schema=False)
app.include_router(router_router, prefix="/router", tags=["router-legacy"], include_in_schema=False)
app.include_router(knowledge_router, prefix="/knowledge", tags=["knowledge-legacy"], include_in_schema=False)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return await error_handler(request, exc)

# 404 handler
@app.get("/{path:path}", tags=["default"])
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