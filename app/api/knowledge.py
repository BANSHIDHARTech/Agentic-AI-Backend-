from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query, Body
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any, Union
import json
import uuid
import logging
from datetime import datetime
from pydantic import BaseModel, Field

from ..services.knowledge_service import KnowledgeService
from ..core.database import db_insert, db_update, db_select, db_delete, db_rpc, get_supabase_client
from ..core.security import get_current_user

# Configure logging
logger = logging.getLogger(__name__)

# Pydantic models for request/response validation
class DocumentUpload(BaseModel):
    type: str = Field(..., description="Type of upload: 'text', 'url', or 'file'")
    content: Optional[str] = Field(None, description="Text content (for type='text')")
    url: Optional[str] = Field(None, description="URL to fetch content from (for type='url')")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")
    session_id: Optional[str] = Field(None, description="Optional session ID to group documents")

class SearchQuery(BaseModel):
    query: str = Field(..., description="Search query string")
    session_id: Optional[str] = Field(None, description="Filter by session ID")
    top_k: int = Field(5, ge=1, le=20, description="Number of results to return")
    min_score: float = Field(0.5, ge=0.0, le=1.0, description="Minimum similarity score (0.0-1.0)")

class DeleteRequest(BaseModel):
    session_id: Optional[str] = Field(None, description="Delete all documents in this session")
    document_ids: Optional[List[str]] = Field(None, description="List of specific document IDs to delete")
    user_id: Optional[str] = Field(None, description="Delete all documents for this user")

# Create router without prefix since it will be added by the app
router = APIRouter(tags=["knowledge"])

@router.post("/upload/text", summary="Upload Text", description="Upload and process text content directly")
async def upload_text(
    content: str = Form(..., description="Text content to upload"),
    session_id: str = Form(..., description="Session ID to group documents"),
    user_id: Optional[str] = Form("anonymous", description="User identifier"),
    metadata: Optional[str] = Form("{}", description="JSON metadata string")
):
    """Upload and process text content directly"""
    try:
        if not content or not content.strip():
            raise HTTPException(status_code=400, detail="Content cannot be empty")
        
        # Process the text upload
        result = await KnowledgeService.upload_document({
            "type": "text",
            "content": content,
            "session_id": session_id,
            "user_id": user_id,
            "metadata": json.loads(metadata) if metadata else {}
        })
        
        return {
            "success": True,
            "message": "Text content uploaded and processed successfully",
            **result
        }

    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to upload text content: {str(error)}"
        )

@router.get("/")
async def knowledge_info():
    """
    Base route for /api/knowledge
    
    Returns:
        dict: API status and available endpoints
    """
    return {
        "status": "Knowledge API is active",
        "endpoints": [
            "POST   /upload           - Upload document (text, URL, or file)",
            "POST   /upload/file      - Upload file document",
            "POST   /upload/url       - Upload document from URL",
            "POST   /upload/text      - Upload text content",
            "POST   /query            - Search knowledge base with semantic search",
            "GET    /search           - Search knowledge base (GET endpoint)",
            "GET    /stats            - Get knowledge base statistics",
            "GET    /sessions         - List all sessions",
            "GET    /sessions/{id}    - Get documents in a session",
            "DELETE /documents        - Delete documents/sessions"
        ]
    }

@router.post("/upload/file", summary="Upload File", description="Upload and process a document file (PDF, TXT, DOCX, etc)")
async def upload_file(
    file: UploadFile = File(..., description="File to upload"),
    session_id: str = Form(..., description="Session ID to group documents"),
    user_id: Optional[str] = Form("anonymous", description="User identifier"),
    metadata: Optional[str] = Form("{}", description="JSON metadata string")
):
    """Upload and process a document file (PDF, TXT, DOCX, etc)"""
    try:
        # Validate file type
        allowed_types = [
            "text/plain", "text/csv", "text/markdown", 
            "application/pdf", "application/json"
        ]
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file.content_type}. Supported types: {', '.join(allowed_types)}"
            )
        
        # Read file content
        file_content = await file.read()
        
        # Process the upload
        result = await KnowledgeService.upload_document({
            "type": "file",
            "file": {
                "filename": file.filename,
                "content": file_content,
                "content_type": file.content_type
            },
            "session_id": session_id,
            "user_id": user_id,
            "metadata": json.loads(metadata) if metadata else {}
        })
        
        return {
            "success": True,
            "message": f"File '{file.filename}' uploaded and processed successfully",
            **result
        }

    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to upload file: {str(error)}"
        )

@router.post("/upload/url", summary="Upload URL", description="Upload and process documents from URLs (Notion, Confluence, Wikipedia, etc.)")
async def upload_url(
    url: str = Form(..., description="URL to fetch content from"),
    session_id: str = Form(..., description="Session ID to group documents"),
    user_id: Optional[str] = Form("anonymous", description="User identifier"),
    metadata: Optional[str] = Form("{}", description="JSON metadata string")
):
    """Upload and process documents from URLs (Notion, Confluence, Wikipedia, etc.)"""
    try:
        # Validate URL format
        from urllib.parse import urlparse
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise HTTPException(status_code=400, detail="Invalid URL format")

        # Parse metadata
        import json
        try:
            metadata_dict = json.loads(metadata) if metadata else {}
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid metadata JSON format")

        # Process the URL upload
        result = await KnowledgeService.upload_document({
            "type": "url",
            "url": url,
            "session_id": session_id,
            "user_id": user_id,
            "metadata": metadata_dict
        })
        
        return {
            "success": True,
            "message": "URL content uploaded and processed successfully",
            **result
        }

    except Exception as error:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to upload URL content: {str(error)}"
        )

@router.post("/upload", summary="Upload Document", description="Upload and process documents from various sources")
async def upload_document(
    type: str = Form(..., description="Type of upload: 'text', 'url', or 'file'"),
    session_id: str = Form(..., description="Session ID to group documents"),
    user_id: Optional[str] = Form("anonymous", description="User identifier"),
    metadata: Optional[str] = Form("{}", description="JSON metadata string"),
    content: Optional[str] = Form(None, description="Text content (for type='text')"),
    url: Optional[str] = Form(None, description="URL to fetch content from (for type='url')"),
    file: Optional[UploadFile] = File(None, description="File to upload (for type='file')")
):
    """Upload and process documents from various sources"""
    try:
        # Validate upload type
        if type not in ["text", "url", "file"]:
            raise HTTPException(
                status_code=400, 
                detail="Invalid upload type. Must be text, url, or file"
            )

        # Parse metadata
        import json
        try:
            metadata_dict = json.loads(metadata) if metadata else {}
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid metadata JSON format")

        # Prepare upload data based on type
        upload_data = {
            "type": type,
            "session_id": session_id,
            "user_id": user_id,
            "metadata": metadata_dict
        }

        if type == "text":
            if not content:
                raise HTTPException(status_code=400, detail="Content is required for text upload")
            upload_data["content"] = content

        elif type == "url":
            if not url:
                raise HTTPException(status_code=400, detail="URL is required for URL upload")
            upload_data["url"] = url

        elif type == "file":
            if not file:
                raise HTTPException(status_code=400, detail="File is required for file upload")
            
            # Validate file type
            allowed_types = [
                "text/plain", "text/csv", "text/markdown", 
                "application/pdf", "application/json"
            ]
            if file.content_type not in allowed_types:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Unsupported file type: {file.content_type}"
                )
            
            # Read file content
            file_content = await file.read()
            upload_data["file"] = {
                "filename": file.filename,
                "content": file_content,
                "content_type": file.content_type
            }

        # Process the upload
        result = await KnowledgeService.upload_document(upload_data)
        
        return {
            "success": True,
            "message": "Document uploaded and processed successfully",
            **result
        }

    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to upload document: {str(error)}"
        )

@router.get("/search")
async def search_knowledge(
    q: str = Query(..., description="Search query"),
    session_id: Optional[str] = Query(None),
    user_id: Optional[str] = Query("anonymous"),
    limit: Optional[int] = Query(10, ge=1, le=50),
    similarity_threshold: Optional[float] = Query(0.7, ge=0.0, le=1.0)
):
    """Search the knowledge base using similarity search"""
    try:
        # Use session_id from query or generate one
        search_session_id = session_id or f"search_{int(datetime.now().timestamp() * 1000)}"

        # Perform the query - pass parameters directly
        result = await KnowledgeService.query_knowledge_base(
            query=q,
            session_id=search_session_id,
            user_id=user_id,
            top_k=limit,
            min_score=similarity_threshold
        )

        if result and isinstance(result, dict):
            return {
                "success": True,
                "message": "Knowledge base search completed",
                **result
            }
        else:
            # Handle case where result is None or not a dict
            return {
                "success": False,
                "message": "Knowledge base search completed",
                "error": "No results found or search error occurred",
                "results": [],
                "count": 0
            }

    except Exception as error:
        # Don't raise an exception, return a response with error details
        logger.error(f"Knowledge search error: {error}", exc_info=True)
        return {
            "success": False,
            "message": "Knowledge base search completed",
            "error": f"Search failed: {str(error)}",
            "results": [],
            "count": 0
        }

@router.post("/query")
async def query_knowledge(
    query: SearchQuery,
    current_user: dict = Depends(get_current_user)
):
    """
    Query the knowledge base using semantic search
    
    Args:
        query: Search query parameters
        current_user: Authenticated user
        
    Returns:
        dict: Search results with relevance scores
    """
    try:
        # Add user ID if not provided
        if not query.user_id and current_user:
            query.user_id = str(current_user.get('id', 'anonymous'))
            
        # Perform the search
        result = await KnowledgeService.query_knowledge_base(
            query=query.query,
            session_id=query.session_id,
            user_id=query.user_id,
            top_k=query.top_k,
            min_score=query.min_score
        )
        
        return {
            "success": True,
            "query": query.query,
            "results": result.get('results', []),
            "count": result.get('count', 0)
        }

    except Exception as error:
        logger.error(f"Query failed: {str(error)}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to query knowledge base: {str(error)}"
        )

@router.get("/stats")
async def get_knowledge_stats(
    user_id: Optional[str] = Query(None, description="Filter stats by user ID"),
    session_id: Optional[str] = Query(None, description="Filter stats by session ID"),
    current_user: dict = Depends(get_current_user)
):
    """
    Get knowledge base statistics and metrics
    
    Args:
        user_id: Filter by user ID
        session_id: Filter by session ID
        current_user: Authenticated user
        
    Returns:
        dict: Statistics about the knowledge base
    """
    try:
        # If no user_id provided, use current user
        if not user_id and current_user:
            user_id = str(current_user.get('id'))
            
        # Get statistics from service
        stats = await KnowledgeService.get_knowledge_stats(
            session_id=session_id,
            user_id=user_id
        )
        
        return {
            "success": True,
            "message": "Knowledge base statistics retrieved",
            "data": stats
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as error:
        logger.error(f"Failed to get knowledge stats: {str(error)}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail="Failed to retrieve knowledge base statistics"
        )

@router.delete("/documents")
async def delete_documents(
    session_id: Optional[str] = Query(None, description="Delete all documents in a session"),
    user_id: Optional[str] = Query(None, description="Delete all documents for a user"),
    document_ids: Optional[str] = Query(None, description="Comma-separated list of document IDs to delete"),
    filter_metadata: Optional[str] = Query(None, description="JSON string of metadata filters")
):
    """
    Delete documents from the knowledge base
    
    At least one of the following must be provided:
    - session_id: Delete all documents in a session
    - user_id: Delete all documents for a user
    - document_ids: Comma-separated list of specific document IDs to delete
    - filter_metadata: JSON string with metadata filters (e.g., {"source_type": "file"})
    """
    try:
        # Parse document IDs if provided
        doc_ids = [id.strip() for id in document_ids.split(',')] if document_ids else []
        
        # Parse metadata filter if provided
        metadata_filter = {}
        if filter_metadata:
            try:
                metadata_filter = json.loads(filter_metadata)
                if not isinstance(metadata_filter, dict):
                    raise ValueError("filter_metadata must be a JSON object")
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON in filter_metadata")
        
        # Prepare delete data
        delete_data = {
            "session_id": session_id,
            "user_id": user_id,
            "document_ids": doc_ids,
            "filter_metadata": metadata_filter
        }
        
        # Validate that at least one deletion criteria is provided
        if not any([session_id, user_id, doc_ids, metadata_filter]):
            raise HTTPException(
                status_code=400,
                detail="At least one of session_id, user_id, document_ids, or filter_metadata must be provided"
            )
        
        # Execute deletion
        result = await KnowledgeService.delete_documents(delete_data)
        
        return {
            "success": True,
            "message": f"Successfully deleted {result['deleted_count']} documents",
            "deleted_count": result['deleted_count'],
            "criteria": result.get('criteria', [])
        }
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except HTTPException:
        raise
    except Exception as error:
        logger.error(f"Failed to delete documents: {str(error)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete documents: {str(error)}"
        )

@router.get("/sessions")
async def list_sessions(
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of sessions to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    current_user: dict = Depends(get_current_user)
):
    """
    List all knowledge base sessions with metadata
    
    Returns:
        - sessions: List of session objects with metadata
        - total: Total number of sessions matching filters
    """
    try:
        # If no user_id provided, use current user if available
        if not user_id and current_user:
            user_id = str(current_user.get('id'))
            
        # Get unique sessions with metadata
        result = await db_rpc('get_sessions', {
            'user_id': user_id,
            'limit': limit,
            'offset': offset
        })
        
        if 'error' in result:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to retrieve sessions: {result.get('error', 'Unknown error')}"
            )
            
        return {
            "success": True,
            "sessions": result.get('data', []),
            "total": result.get('count', 0)
        }
        
    except HTTPException:
        raise
    except Exception as error:
        logger.error(f"Failed to list sessions: {str(error)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve sessions: {str(error)}"
        )

@router.get("/sessions/{session_id}")
async def get_session_data(
    session_id: str,
    include_documents: bool = Query(True, description="Include document content in response"),
    current_user: dict = Depends(get_current_user)
):
    """
    Get all documents and metadata for a specific session
    
    Args:
        session_id: ID of the session to retrieve
        include_documents: Whether to include full document content
        current_user: Authenticated user (for access control)
        
    Returns:
        dict: Session data including metadata and documents
    """
    try:
        if not session_id:
            raise HTTPException(status_code=400, detail="Session ID is required")
            
        # Get session data from service
        result = await KnowledgeService.get_session_data(
            session_id=session_id,
            user_id=str(current_user.get('id')) if current_user else None,
            include_documents=include_documents
        )
        
        if not result:
            raise HTTPException(status_code=404, detail="Session not found")
            
        return {
            "success": True,
            "session": {
                "id": session_id,
                "created_at": result.get('created_at'),
                "updated_at": result.get('updated_at'),
                "document_count": result.get('document_count', 0),
                "user_id": result.get('user_id'),
                "metadata": result.get('metadata', {})
            },
            "documents": result.get('documents', []),
            "count": result.get('document_count', 0)
        }
        
    except HTTPException:
        raise
    except Exception as error:
        logger.error(f"Failed to get session data: {str(error)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve session data: {str(error)}"
        )
        
@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Delete a session and all its documents
    
    Args:
        session_id: ID of the session to delete
        current_user: Authenticated user (for authorization)
        
    Returns:
        dict: Deletion result with count of deleted documents
    """
    try:
        if not session_id:
            raise HTTPException(status_code=400, detail="Session ID is required")
            
        # Verify session exists and belongs to user
        session_data = await KnowledgeService.get_session_data(
            session_id=session_id,
            user_id=str(current_user.get('id')) if current_user else None,
            include_documents=False
        )
        
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found or access denied")
            
        # Delete session and all its documents
        result = await KnowledgeService.delete_documents({
            'session_id': session_id,
            'user_id': str(current_user.get('id')) if current_user else None
        })
        
        return {
            "success": True,
            "message": f"Session and {result.get('deleted_count', 0)} documents deleted",
            "deleted_count": result.get('deleted_count', 0)
        }
        
    except HTTPException:
        raise
    except Exception as error:
        logger.error(f"Failed to delete session: {str(error)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete session: {str(error)}"
        )

@router.post("/test")
async def test_knowledge_base():
    """
    Comprehensive test endpoint for knowledge base functionality
    
    Tests:
    1. Document upload (text, URL, file)
    2. Query functionality
    3. Stats retrieval
    4. Session data retrieval
    5. Document deletion
    """
    import uuid
    from fastapi import UploadFile
    from io import BytesIO
    
    test_session_id = f"test_session_{uuid.uuid4().hex[:8]}"
    test_user_id = f"test_user_{uuid.uuid4().hex[:4]}"
    test_results = {}
    
    try:
        # 1. Test text upload
        test_text = """
        Artificial intelligence (AI) is intelligence demonstrated by machines, 
        as opposed to natural intelligence displayed by animals including humans. 
        AI research has been defined as the field of study of intelligent agents, 
        which refers to any system that perceives its environment and takes actions 
        that maximize its chance of achieving its goals.
        """
        
        text_upload = await KnowledgeService.upload_document({
            "type": "text",
            "content": test_text,
            "session_id": test_session_id,
            "user_id": test_user_id,
            "metadata": {"test": "text_upload", "topic": "AI"}
        })
        
        test_results["text_upload"] = {
            "success": True,
            "document_id": text_upload.get("document_ids", [None])[0],
            "chunk_count": text_upload.get("chunk_count", 0)
        }
        
        # 2. Test URL upload (simulated)
        test_url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
        try:
            url_upload = await KnowledgeService.upload_document({
                "type": "url",
                "url": test_url,
                "session_id": test_session_id,
                "user_id": test_user_id,
                "metadata": {"test": "url_upload", "source": "wikipedia"}
            })
            
            test_results["url_upload"] = {
                "success": True,
                "document_id": url_upload.get("document_ids", [None])[0],
                "chunk_count": url_upload.get("chunk_count", 0)
            }
        except Exception as e:
            test_results["url_upload"] = {
                "success": False,
                "error": str(e),
                "warning": "URL upload test skipped - may be due to network issues"
            }
        
        # 3. Test file upload (simulated)
        test_file_content = b"""
        Machine Learning (ML) is the study of computer algorithms that can improve 
        automatically through experience and by the use of data. It is seen as a part 
        of artificial intelligence.
        """
        
        file_upload = await KnowledgeService.upload_document({
            "type": "file",
            "file": {
                "filename": "ml_concepts.txt",
                "content": test_file_content,
                "content_type": "text/plain"
            },
            "session_id": test_session_id,
            "user_id": test_user_id,
            "metadata": {"test": "file_upload", "format": "text"}
        })
        
        test_results["file_upload"] = {
            "success": True,
            "document_id": file_upload.get("document_ids", [None])[0],
            "chunk_count": file_upload.get("chunk_count", 0)
        }
        
        # 4. Test query
        query_result = await KnowledgeService.query_knowledge_base({
            "query": "What is artificial intelligence?",
            "session_id": test_session_id,
            "user_id": test_user_id,
            "limit": 2,
            "similarity_threshold": 0.3
        })
        
        test_results["query"] = {
            "success": True,
            "result_count": len(query_result.get("results", [])),
            "processing_time_ms": query_result.get("processing_time_ms", 0)
        }
        
        # 5. Test stats
        stats = await KnowledgeService.get_knowledge_stats(
            user_id=test_user_id,
            session_id=test_session_id
        )
        
        test_results["stats"] = {
            "success": True,
            **stats
        }
        
        # 6. Test session data
        session_data = await KnowledgeService.get_session_data(
            session_id=test_session_id,
            user_id=test_user_id
        )
        
        test_results["session_data"] = {
            "success": True,
            "document_count": session_data.get("count", 0),
            "session_id": session_data.get("session", {}).get("session_id")
        }
        
        # 7. Cleanup - delete test data
        await KnowledgeService.delete_documents({
            "session_id": test_session_id,
            "user_id": test_user_id
        })
        
        # Verify cleanup
        stats_after = await KnowledgeService.get_knowledge_stats(
            user_id=test_user_id,
            session_id=test_session_id
        )
        
        test_results["cleanup"] = {
            "success": stats_after.get("chunks", 0) == 0,
            "documents_remaining": stats_after.get("chunks", 0)
        }
        
        # Calculate overall test status
        all_tests = [v for k, v in test_results.items() if "success" in v]
        successful_tests = sum(1 for t in all_tests if t["success"])
        
        return {
            "success": successful_tests == len(all_tests),
            "test_session_id": test_session_id,
            "test_user_id": test_user_id,
            "test_results": test_results,
            "summary": {
                "total_tests": len(all_tests),
                "successful_tests": successful_tests,
                "success_rate": f"{(successful_tests / len(all_tests) * 100):.1f}%"
            },
            "tested_at": datetime.utcnow().isoformat()
        }
        
    except Exception as error:
        logger.error(f"Knowledge base test failed: {str(error)}", exc_info=True)
        
        # Include partial results if available
        test_results["error"] = str(error)
        
        return {
            "success": False,
            "test_session_id": test_session_id,
            "test_user_id": test_user_id,
            "error": f"Test failed: {str(error)}",
            "test_results": test_results,
            "tested_at": datetime.utcnow().isoformat()
        }
