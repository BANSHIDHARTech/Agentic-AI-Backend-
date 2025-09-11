from typing import Optional
from fastapi import APIRouter, HTTPException, Query, UploadFile, File, Form
from fastapi.responses import JSONResponse
from datetime import datetime

from ..core.models import (
    DocumentCreateRequest, DocumentResponse, KnowledgeQueryRequest, 
    KnowledgeQueryResponse, APIResponse
)
from ..services.knowledge_service import KnowledgeService
from ..core.database import supabase

router = APIRouter()

@router.get("/")
async def knowledge_info():
    """Base route for /api/knowledge"""
    return {
        "status": "Knowledge API is active",
        "endpoints": [
            "POST   /upload",
            "POST   /upload/url",
            "GET    /search",
            "POST   /query",
            "GET    /stats",
            "DELETE /documents",
            "GET    /sessions/:sessionId",
            "POST   /test"
        ]
    }

@router.post("/upload/url")
async def upload_url(
    url: str = Form(...),
    session_id: str = Form(...),
    user_id: Optional[str] = Form("anonymous"),
    metadata: Optional[str] = Form("{}")
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

@router.post("/upload")
async def upload_document(
    type: str = Form(...),
    session_id: str = Form(...),
    user_id: Optional[str] = Form("anonymous"),
    metadata: Optional[str] = Form("{}"),
    content: Optional[str] = Form(None),
    url: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
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

        # Perform the query
        result = await KnowledgeService.query_knowledge_base({
            "query": q,
            "session_id": search_session_id,
            "user_id": user_id,
            "limit": limit,
            "similarity_threshold": similarity_threshold
        })

        return {
            "success": True,
            "message": "Knowledge base search completed",
            **result
        }

    except Exception as error:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to search knowledge base: {str(error)}"
        )

@router.post("/query", response_model=KnowledgeQueryResponse)
async def query_knowledge(request: KnowledgeQueryRequest):
    """Query the knowledge base using similarity search"""
    try:
        # Perform the query
        result = await KnowledgeService.query_knowledge_base(request.dict())
        return result

    except Exception as error:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to query knowledge base: {str(error)}"
        )

@router.get("/stats")
async def get_knowledge_stats():
    """Get knowledge base statistics and metrics"""
    try:
        stats = await KnowledgeService.get_knowledge_stats()
        
        return {
            "success": True,
            "message": "Knowledge base statistics retrieved",
            **stats
        }

    except Exception as error:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to retrieve knowledge base statistics: {str(error)}"
        )

@router.delete("/documents")
async def delete_documents(
    session_id: Optional[str] = None,
    user_id: Optional[str] = None
):
    """Delete documents by session or user"""
    try:
        if not session_id and not user_id:
            raise HTTPException(
                status_code=400, 
                detail="Either session_id or user_id is required"
            )

        result = await KnowledgeService.delete_documents({
            "session_id": session_id,
            "user_id": user_id
        })

        return {
            "success": True,
            "message": "Documents deleted successfully",
            **result
        }

    except Exception as error:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to delete documents: {str(error)}"
        )

@router.get("/sessions/{session_id}")
async def get_session_data(session_id: str):
    """Get knowledge base activity for a specific session"""
    try:
        # Get session queries
        queries_result = supabase.from_('knowledge_sessions').select('*').eq('session_id', session_id).order('created_at', desc=True).execute()
        
        if queries_result.error:
            raise Exception(queries_result.error)

        # Get session documents
        docs_result = supabase.from_('documents').select('id, source_type, source_reference, chunk_index, total_chunks, created_at, metadata').eq('session_id', session_id).order('created_at', desc=True).execute()
        
        if docs_result.error:
            raise Exception(docs_result.error)

        queries = queries_result.data or []
        documents = docs_result.data or []

        return {
            "success": True,
            "session_id": session_id,
            "queries": queries,
            "documents": documents,
            "query_count": len(queries),
            "document_count": len(documents)
        }

    except Exception as error:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to retrieve session data: {str(error)}"
        )

@router.post("/test")
async def test_knowledge_base():
    """Test endpoint for knowledge base functionality"""
    try:
        test_queries = [
            'What is artificial intelligence?',
            'How does machine learning work?',
            'Explain neural networks',
            'What are the benefits of automation?'
        ]

        results = []
        
        for query in test_queries:
            try:
                result = await KnowledgeService.query_knowledge_base({
                    "query": query,
                    "session_id": f"test_{int(datetime.now().timestamp() * 1000)}",
                    "user_id": "test_user",
                    "limit": 3,
                    "similarity_threshold": 0.5
                })
                
                results.append({
                    "query": query,
                    "result_count": result.get("result_count", 0),
                    "processing_time_ms": result.get("processing_time_ms", 0),
                    "success": True
                })
            except Exception as error:
                results.append({
                    "query": query,
                    "success": False,
                    "error": str(error)
                })

        successful_queries = len([r for r in results if r["success"]])

        return {
            "success": True,
            "message": "Knowledge base test completed",
            "test_results": results,
            "total_queries": len(test_queries),
            "successful_queries": successful_queries,
            "tested_at": datetime.now().isoformat()
        }

    except Exception as error:
        raise HTTPException(
            status_code=500, 
            detail=f"Knowledge base test failed: {str(error)}"
        )
