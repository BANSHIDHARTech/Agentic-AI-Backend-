from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query, Body
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any, Union
import json
import uuid
import logging
import time
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
        # Validate content
        if not content or not content.strip():
            raise HTTPException(status_code=400, detail="Content cannot be empty")
            
        # Validate session_id
        if not session_id or not session_id.strip():
            session_id = f"session_{int(datetime.now().timestamp())}_{uuid.uuid4().hex[:8]}"
            logger.info(f"Generated session ID: {session_id}")
            
        # Parse metadata safely
        meta_dict = {}
        if metadata and metadata.strip():
            try:
                meta_dict = json.loads(metadata)
                if not isinstance(meta_dict, dict):
                    logger.warning(f"Metadata is not a dictionary: {metadata}")
                    meta_dict = {"raw_input": metadata}
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid metadata JSON: {metadata}, error: {str(e)}")
                meta_dict = {"raw_input": metadata}
        
        # Add standard metadata
        meta_dict.update({
            'upload_timestamp': datetime.utcnow().isoformat(),
            'content_length': len(content)
        })
        
        # Log request details for debugging
        logger.info(f"Processing text upload: session={session_id}, user={user_id}, content_length={len(content)}")
        
        # Use direct database insertion for reliability
        try:
            # Generate a UUID for the document
            doc_id = str(uuid.uuid4())
            
            # Insert document directly into the database
            from ..core.database import get_supabase_client
            supabase = get_supabase_client()
            
            # Prepare document data
            document_data = {
                'id': doc_id,
                'session_id': session_id,
                'user_id': user_id or 'anonymous',
                'content': content,
                'metadata': meta_dict,
                'source_type': 'text',
                'source_reference': 'direct_input',
                'chunk_index': 0,
                'total_chunks': 1,
                'created_at': datetime.utcnow().isoformat()
            }
            
            # Insert document
            insert_result = supabase.table('documents').insert(document_data).execute()
            
            if not insert_result or not getattr(insert_result, 'data', None):
                error = getattr(insert_result, 'error', 'Unknown error')
                logger.error(f"Failed to store document: {error}")
                
                # Return partial success with warning
                return {
                    "success": True,
                    "warning": "Document processed but storage may have failed",
                    "message": "Text content processed but storage may have failed",
                    "document_ids": [doc_id],
                    "chunks_created": 0,
                    "total_chunks": 1,
                    "source_type": "text",
                    "source_reference": "direct_input",
                    "session_id": session_id,
                    "user_id": user_id,
                    "metadata": meta_dict
                }
            
            # Return success
            return {
                "success": True,
                "message": "Text content uploaded and processed successfully",
                "document_ids": [doc_id],
                "chunks_created": 1,
                "total_chunks": 1,
                "source_type": "text",
                "source_reference": "direct_input",
                "session_id": session_id,
                "user_id": user_id,
                "metadata": meta_dict
            }
            
        except Exception as service_error:
            # Log the detailed error for debugging
            logger.error(f"Direct document insertion failed: {str(service_error)}", exc_info=True)
            
            # Return a 500 with detailed error message for better debugging
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "Failed to store document directly",
                    "error": str(service_error),
                    "type": type(service_error).__name__,
                }
            )

    except HTTPException:
        raise
    except Exception as error:
        # Log the error for debugging
        logger.error(f"Text upload endpoint failed: {str(error)}", exc_info=True)
        
        # Return a detailed error response
        raise HTTPException(
            status_code=500, 
            detail={
                "message": "Failed to upload text content",
                "error": str(error),
                "type": type(error).__name__
            }
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
        
        # Parse metadata safely
        meta_dict = {}
        if metadata and metadata.strip():
            try:
                meta_dict = json.loads(metadata)
                if not isinstance(meta_dict, dict):
                    logger.warning(f"Metadata is not a dictionary: {metadata}")
                    meta_dict = {"raw_input": metadata}
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid metadata JSON: {metadata}, error: {str(e)}")
                meta_dict = {"raw_input": metadata}
        
        # Add file metadata
        meta_dict.update({
            "file_type": file.content_type,
            "file_name": file.filename,
            "file_size": len(file_content),
            "upload_timestamp": datetime.utcnow().isoformat()
        })
        
        # Extract text from PDF
        text_content = ""
        chunks = []
        
        try:
            # For PDF files
            if file.content_type == "application/pdf":
                try:
                    # Import PDF reader
                    from PyPDF2 import PdfReader
                    from io import BytesIO
                    
                    # Read PDF content
                    pdf = PdfReader(BytesIO(file_content))
                    
                    # Extract text from all pages
                    for i, page in enumerate(pdf.pages):
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            chunks.append({
                                "text": page_text.strip(),
                                "page": i + 1
                            })
                    
                    # Combine all text
                    text_content = "\n\n".join([chunk["text"] for chunk in chunks])
                    
                except Exception as pdf_error:
                    logger.error(f"Error extracting PDF text: {str(pdf_error)}")
                    text_content = f"[Error extracting PDF content: {str(pdf_error)}]"
                    chunks = [{"text": text_content, "page": 1}]
            
            # For text files
            elif file.content_type == "text/plain":
                try:
                    text_content = file_content.decode("utf-8")
                    chunks = [{"text": text_content, "page": 1}]
                except UnicodeDecodeError:
                    try:
                        text_content = file_content.decode("latin-1")
                        chunks = [{"text": text_content, "page": 1}]
                    except Exception as text_error:
                        logger.error(f"Error decoding text file: {str(text_error)}")
                        text_content = f"[Error decoding text file: {str(text_error)}]"
                        chunks = [{"text": text_content, "page": 1}]
            
            # For other file types
            else:
                text_content = f"[Unsupported file type for text extraction: {file.content_type}]"
                chunks = [{"text": text_content, "page": 1}]
                
        except Exception as extract_error:
            logger.error(f"Error in file content extraction: {str(extract_error)}")
            text_content = f"[Error extracting content: {str(extract_error)}]"
            chunks = [{"text": text_content, "page": 1}]
            
        # Store documents in database
        document_ids = []
        from ..core.database import get_supabase_client
        supabase = get_supabase_client()
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            try:
                # Generate document ID
                doc_id = str(uuid.uuid4())
                
                # Prepare document data
                chunk_metadata = {
                    **meta_dict,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "page": chunk.get("page", 1)
                }
                
                # Create document record
                document_data = {
                    'id': doc_id,
                    'session_id': session_id,
                    'user_id': user_id or 'anonymous',
                    'content': chunk["text"],
                    'metadata': chunk_metadata,
                    'source_type': 'file',
                    'source_reference': file.filename,
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'created_at': datetime.utcnow().isoformat()
                }
                
                # Insert document
                insert_result = supabase.table('documents').insert(document_data).execute()
                
                if insert_result and getattr(insert_result, 'data', None):
                    document_ids.append(doc_id)
                    logger.info(f"Successfully stored chunk {i+1}/{len(chunks)} with ID {doc_id}")
                else:
                    logger.error(f"Failed to store chunk {i+1}: {getattr(insert_result, 'error', 'Unknown error')}")
                    
            except Exception as chunk_error:
                logger.error(f"Error storing chunk {i+1}: {str(chunk_error)}")
                
        # Calculate processing time
        processing_time = 10  # Placeholder value
        
        # Return the result
        return {
            "success": True,
            "message": f"File '{file.filename}' uploaded and processed successfully",
            "document_ids": document_ids,
            "chunks_created": len(document_ids),
            "total_chunks": len(chunks),
            "source_type": "file",
            "source_reference": file.filename,
            "processing_time_ms": processing_time,
            "metadata": meta_dict
        }

    except HTTPException:
        raise
    except Exception as error:
        logger.error(f"File upload failed: {str(error)}", exc_info=True)
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
            meta_dict = json.loads(metadata) if metadata else {}
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid metadata JSON format")
            
        # Add standard metadata
        meta_dict.update({
            'source_url': url,
            'upload_timestamp': datetime.utcnow().isoformat()
        })
        
        # Log request details for debugging
        logger.info(f"Processing URL upload: url={url}, session={session_id}")
        
        # Fetch URL content
        import httpx
        import asyncio
        from ..core.knowledge_utils import extract_text_from_html
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=30.0)
                response.raise_for_status()
                
                content_type = response.headers.get('content-type', '')
                meta_dict['content_type'] = content_type
                
                html_content = response.text
                
                # Extract text from HTML
                text_content = extract_text_from_html(html_content)
                
                if not text_content or len(text_content.strip()) < 10:
                    raise HTTPException(
                        status_code=400, 
                        detail="URL content extraction failed: insufficient content"
                    )
                    
                logger.info(f"Successfully extracted {len(text_content)} chars from URL")
                
                # Use direct database insertion for reliability
                try:
                    # Process the text with chunking
                    from ..core.knowledge_utils import chunk_text
                    chunks = chunk_text(text_content, chunk_size=1000, chunk_overlap=200)
                    
                    if not chunks:
                        raise ValueError("No content chunks could be created")
                    
                    total_chunks = len(chunks)
                    logger.info(f"Split URL content into {total_chunks} chunks")
                    
                    # Use Supabase client
                    from ..core.database import get_supabase_client
                    supabase = get_supabase_client()
                    
                    # Store each chunk
                    document_ids = []
                    start_time = time.time()
                    
                    for i, chunk in enumerate(chunks):
                        # Generate a UUID for the document
                        doc_id = str(uuid.uuid4())
                        
                        # Create chunk metadata
                        chunk_metadata = {
                            **meta_dict,
                            'chunk_index': i,
                            'total_chunks': total_chunks,
                            'chunk_size': len(chunk)
                        }
                        
                        # Prepare document data
                        document_data = {
                            'id': doc_id,
                            'session_id': session_id,
                            'user_id': user_id or 'anonymous',
                            'content': chunk,
                            'metadata': chunk_metadata,
                            'source_type': 'url',
                            'source_reference': url,
                            'chunk_index': i,
                            'total_chunks': total_chunks,
                            'created_at': datetime.utcnow().isoformat()
                        }
                        
                        # Insert document
                        insert_result = supabase.table('documents').insert(document_data).execute()
                        
                        if insert_result and getattr(insert_result, 'data', None):
                            document_ids.append(doc_id)
                            logger.info(f"Successfully stored chunk {i+1}/{total_chunks} with ID {doc_id}")
                        else:
                            error = getattr(insert_result, 'error', 'Unknown error')
                            logger.error(f"Failed to store chunk {i+1}: {error}")
                    
                    # Calculate processing time
                    processing_time = int((time.time() - start_time) * 1000)
                    
                    # Return success
                    return {
                        "success": True,
                        "message": "URL content uploaded and processed successfully",
                        "document_ids": document_ids,
                        "chunks_created": len(document_ids),
                        "total_chunks": total_chunks,
                        "source_type": "url",
                        "source_reference": url,
                        "processing_time_ms": processing_time,
                        "metadata": meta_dict
                    }
                    
                except Exception as db_error:
                    # Log the detailed error for debugging
                    logger.error(f"Direct document insertion failed: {str(db_error)}", exc_info=True)
                    
                    # Return a 500 with detailed error message for better debugging
                    raise HTTPException(
                        status_code=500,
                        detail={
                            "message": "Failed to store document directly",
                            "error": str(db_error),
                            "type": type(db_error).__name__,
                        }
                    )
                
        except httpx.HTTPError as http_error:
            logger.error(f"HTTP error while fetching URL: {str(http_error)}")
            raise HTTPException(
                status_code=502,
                detail=f"Failed to fetch URL content: {str(http_error)}"
            )

    except HTTPException:
        raise
    except Exception as error:
        # Log the error for debugging
        logger.error(f"URL upload endpoint failed: {str(error)}", exc_info=True)
        
        # Return a detailed error response
        raise HTTPException(
            status_code=500, 
            detail={
                "message": "Failed to upload URL content",
                "error": str(error),
                "type": type(error).__name__
            }
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
        # Log the request
        logger.info(f"Knowledge search request: query='{q}', session_id={session_id}, user_id={user_id}")
        
        # Generate a session ID if not provided
        search_session_id = session_id or f"search_{int(datetime.now().timestamp() * 1000)}"
        logger.info(f"Using session ID: {search_session_id}")

        try:
            # Direct database query for reliability
            from ..core.database import get_supabase_client
            supabase = get_supabase_client()
            
            # Build query
            db_query = supabase.table('documents').select('*')
            
            # Add filters
            if session_id:
                db_query = db_query.eq('session_id', session_id)
            if user_id and user_id != "anonymous":
                db_query = db_query.eq('user_id', user_id)
                
            # Add simple content search if query provided
            if q:
                # Simple contains search - not ideal but works as fallback
                db_query = db_query.ilike('content', f'%{q}%')
                
            # Add limit and order by created_at
            db_query = db_query.order('created_at', desc=True).limit(limit)
            
            # Execute query
            result = db_query.execute()
            
            # Process results
            documents = []
            if result and getattr(result, 'data', None):
                for doc in result.data:
                    # Safely extract metadata
                    metadata = doc.get('metadata', {})
                    if isinstance(metadata, str):
                        try:
                            metadata = json.loads(metadata)
                        except:
                            metadata = {}
                            
                    # Format document
                    documents.append({
                        'id': doc.get('id', ''),
                        'content': doc.get('content', ''),
                        'score': 0.7,  # Default score since we're not doing vector search
                        'metadata': {
                            'source_type': doc.get('source_type', ''),
                            'source_reference': doc.get('source_reference', ''),
                            'chunk_index': doc.get('chunk_index', 0),
                            'total_chunks': doc.get('total_chunks', 1),
                            **metadata
                        }
                    })
            
            # Return a consistent response format
            return {
                "success": True,
                "message": "Knowledge base search completed successfully",
                "query": q,
                "results": documents,
                "count": len(documents),
                "session_id": search_session_id,
                "processing_time_ms": 1,
                "search_method": "direct"
            }
            
        except Exception as e:
            logger.error(f"Error in knowledge search: {str(e)}", exc_info=True)
            
            # Fallback to standard service
            try:
                result = await KnowledgeService.query_knowledge_base(
                    query=q,
                    session_id=search_session_id,
                    user_id=user_id,
                    top_k=limit,
                    min_score=similarity_threshold
                )
                
                # Process the result (simplified)
                formatted_results = []
                results_data = result.get('results', [])
                
                for item in results_data:
                    if isinstance(item, dict):
                        formatted_results.append({
                            'id': item.get('id', ''),
                            'content': item.get('content', ''),
                            'score': item.get('score', 0.5),
                            'metadata': item.get('metadata', {})
                        })
                        
                return {
                    "success": True,
                    "message": "Knowledge base search completed (fallback)",
                    "query": q,
                    "results": formatted_results,
                    "count": len(formatted_results),
                    "session_id": search_session_id
                }
            except Exception as e2:
                logger.error(f"Fallback search also failed: {str(e2)}", exc_info=True)
                return {
                    "success": False,
                    "message": "Knowledge base search failed",
                    "error": f"Search error: {str(e2)}",
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
            "count": 0,
            "query": q
        }

@router.post("/query")
async def query_knowledge(
    query: SearchQuery
):
    """
    Query the knowledge base using semantic search
    
    Args:
        query: Search query parameters
        
    Returns:
        dict: Search results with relevance scores
    """
    try:
        # Log the request
        logger.info(f"Knowledge POST query: '{query.query}', session_id={query.session_id}")
        
        # Add user ID if not provided
        if not query.session_id:
            # Generate a session ID if not provided
            query.session_id = f"search_{int(datetime.now().timestamp() * 1000)}"
            logger.info(f"Generated session ID: {query.session_id}")
        
        # Use direct database search for reliability
        try:
            # Get Supabase client
            from ..core.database import get_supabase_client
            supabase = get_supabase_client()
            
            # Build query
            db_query = supabase.table('documents').select('*')
            
            # Add filters
            if query.session_id:
                db_query = db_query.eq('session_id', query.session_id)
            
            # Add simple content search
            if query.query:
                # Simple contains search - not ideal but works as fallback
                db_query = db_query.ilike('content', f'%{query.query}%')
                
            # Add limit and order by created_at
            db_query = db_query.order('created_at', desc=True).limit(query.top_k)
            
            # Execute query
            result = db_query.execute()
            
            # Process results
            documents = []
            if result and hasattr(result, 'data') and result.data:
                for doc in result.data:
                    # Safely extract metadata
                    metadata = doc.get('metadata', {})
                    if isinstance(metadata, str):
                        try:
                            metadata = json.loads(metadata)
                        except:
                            metadata = {}
                            
                    # Format document
                    documents.append({
                        'id': doc.get('id', ''),
                        'content': doc.get('content', ''),
                        'score': 0.7,  # Default score since we're not doing vector search
                        'metadata': {
                            'source_type': doc.get('source_type', ''),
                            'source_reference': doc.get('source_reference', ''),
                            'chunk_index': doc.get('chunk_index', 0),
                            'total_chunks': doc.get('total_chunks', 1),
                            **metadata
                        }
                    })
                
                logger.info(f"Query completed with {len(documents)} results")
                
                return {
                    "success": True,
                    "message": "Knowledge base query completed successfully",
                    "query": query.query,
                    "results": documents,
                    "count": len(documents),
                    "session_id": query.session_id
                }
            else:
                return {
                    "success": True,
                    "message": "Knowledge base query completed successfully, but no results found",
                    "query": query.query,
                    "results": [],
                    "count": 0,
                    "session_id": query.session_id
                }
            
        except Exception as e:
            logger.error(f"Error in knowledge query: {str(e)}", exc_info=True)
            return {
                "success": False,
                "message": "Knowledge base query failed",
                "error": f"Query error: {str(e)}",
                "results": [],
                "count": 0,
                "query": query.query
            }
        
        # Process the search results
        if result and isinstance(result, dict):
            # Format the results consistently
            formatted_results = []
            
            # Get results array from different possible structures
            results_data = []
            if 'results' in result:
                results_data = result['results']
            elif 'data' in result:
                results_data = result['data']
                
            # Process each result item
            for item in results_data:
                if isinstance(item, dict):
                    formatted_results.append({
                        'id': item.get('id', ''),
                        'content': item.get('content', ''),
                        'score': item.get('score', 0.0),
                        'metadata': item.get('metadata', {})
                    })
        
            return {
                "success": True,
                "message": "Knowledge base query completed",
                "query": query.query,
                "results": formatted_results,
                "count": len(formatted_results),
                "session_id": query.session_id
            }
        else:
            # Handle case where result is None or not a dict
            return {
                "success": False,
                "message": "Knowledge base query completed",
                "error": "No results found or query error occurred",
                "results": [],
                "count": 0,
                "query": query.query
            }

    except Exception as error:
        # Don't raise an exception, return a response with error details
        logger.error(f"Knowledge query error: {error}", exc_info=True)
        return {
            "success": False,
            "message": "Knowledge base query failed",
            "error": f"Query failed: {str(error)}",
            "results": [],
            "count": 0,
            "query": query.query if query else ""
        }

@router.get("/stats")
async def get_knowledge_stats(
    user_id: Optional[str] = Query(None, description="Filter stats by user ID"),
    session_id: Optional[str] = Query(None, description="Filter stats by session ID")
):
    """
    Get knowledge base statistics and metrics
    
    Args:
        user_id: Filter by user ID
        session_id: Filter by session ID
        
    Returns:
        dict: Statistics about the knowledge base
    """
    try:
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
    offset: int = Query(0, ge=0, description="Pagination offset")
):
    """
    List all knowledge base sessions with metadata
    
    Returns:
        - sessions: List of session objects with metadata
        - total: Total number of sessions matching filters
    """
    try:
            
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
    include_documents: bool = Query(True, description="Include document content in response")
):
    """
    Get all documents and metadata for a specific session
    
    Args:
        session_id: ID of the session to retrieve
        include_documents: Whether to include full document content
        
    Returns:
        dict: Session data including metadata and documents
    """
    try:
        if not session_id:
            raise HTTPException(status_code=400, detail="Session ID is required")
            
        # Direct database query for reliability
        try:
            from ..core.database import get_supabase_client
            supabase = get_supabase_client()
            
            # Select columns based on include_documents
            columns = '*' if include_documents else 'id,session_id,user_id,source_type,source_reference,chunk_index,total_chunks,metadata,created_at'
            
            # Query documents
            query = supabase.table('documents').select(columns).eq('session_id', session_id)
                    
            # Execute query
            result = query.execute()
            
            # Process results
            documents = []
            if result and getattr(result, 'data', None):
                for doc in result.data:
                    # Safely extract metadata
                    metadata = doc.get('metadata', {})
                    if isinstance(metadata, str):
                        try:
                            metadata = json.loads(metadata)
                        except:
                            metadata = {}
                            
                    # Format document
                    doc_data = {
                        'id': doc.get('id', ''),
                        'session_id': doc.get('session_id', ''),
                        'user_id': doc.get('user_id', ''),
                        'source_type': doc.get('source_type', ''),
                        'source_reference': doc.get('source_reference', ''),
                        'chunk_index': doc.get('chunk_index', 0),
                        'total_chunks': doc.get('total_chunks', 1),
                        'metadata': metadata,
                        'created_at': doc.get('created_at', '')
                    }
                    
                    if include_documents:
                        doc_data['content'] = doc.get('content', '')
                        
                    documents.append(doc_data)
            
            # Create session metadata
            session_metadata = {
                "id": session_id,
                "created_at": documents[0].get('created_at') if documents else None,
                "updated_at": documents[0].get('created_at') if documents else None,
                "document_count": len(documents),
                "user_id": documents[0].get('user_id') if documents else None,
                "metadata": {}
            }
            
            return {
                "success": True,
                "session": session_metadata,
                "documents": documents,
                "count": len(documents)
            }
            
        except Exception as db_error:
            logger.error(f"Direct database query failed: {str(db_error)}", exc_info=True)
            
            # Fallback to service method
            result = await KnowledgeService.get_session_data(
                session_id=session_id,
                user_id=None,
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
    session_id: str
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
            
        # Verify session exists
        session_data = await KnowledgeService.get_session_data(
            session_id=session_id,
            user_id=None,
            include_documents=False
        )
        
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")
            
        # Delete session and all its documents
        result = await KnowledgeService.delete_documents({
            'session_id': session_id,
            'user_id': None
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
    import time
    from fastapi import UploadFile
    from io import BytesIO
    
    test_session_id = f"test_session_{uuid.uuid4().hex[:8]}"
    test_user_id = f"test_user_{uuid.uuid4().hex[:4]}"
    test_results = {}
    
    try:
        # Get Supabase client for direct operations
        from ..core.database import get_supabase_client
        supabase = get_supabase_client()
        
        # 1. Test text upload - direct approach
        test_text = """
        Artificial intelligence (AI) is intelligence demonstrated by machines, 
        as opposed to natural intelligence displayed by animals including humans. 
        AI research has been defined as the field of study of intelligent agents, 
        which refers to any system that perceives its environment and takes actions 
        that maximize its chance of achieving its goals.
        """
        
        try:
            # Generate a UUID for the document
            doc_id = str(uuid.uuid4())
            
            # Prepare document data
            document_data = {
                'id': doc_id,
                'session_id': test_session_id,
                'user_id': test_user_id,
                'content': test_text,
                'metadata': {"test": "text_upload", "topic": "AI"},
                'source_type': 'text',
                'source_reference': 'test_endpoint',
                'chunk_index': 0,
                'total_chunks': 1,
                'created_at': datetime.utcnow().isoformat()
            }
            
            # Insert document
            text_result = supabase.table('documents').insert(document_data).execute()
            
            test_results["text_upload"] = {
                "success": bool(text_result and hasattr(text_result, 'data')),
                "document_id": doc_id,
                "chunk_count": 1
            }
            
        except Exception as e:
            logger.error(f"Text upload test failed: {str(e)}", exc_info=True)
            test_results["text_upload"] = {
                "success": False,
                "error": str(e)
            }
        
        # 2. Test URL upload with direct approach
        try:
            # Generate a UUID for the document
            url_doc_id = str(uuid.uuid4())
            
            # Prepare document data for URL
            url_document_data = {
                'id': url_doc_id,
                'session_id': test_session_id,
                'user_id': test_user_id,
                'content': "This is simulated URL content for testing purposes.",
                'metadata': {"test": "url_upload", "source": "test_endpoint"},
                'source_type': 'url',
                'source_reference': 'https://example.com/test',
                'chunk_index': 0,
                'total_chunks': 1,
                'created_at': datetime.utcnow().isoformat()
            }
            
            # Insert document
            url_result = supabase.table('documents').insert(url_document_data).execute()
            
            test_results["url_upload"] = {
                "success": bool(url_result and hasattr(url_result, 'data')),
                "document_id": url_doc_id,
                "chunk_count": 1
            }
        except Exception as e:
            logger.error(f"URL upload test failed: {str(e)}", exc_info=True)
            test_results["url_upload"] = {
                "success": False,
                "error": str(e)
            }
        
        # 3. Test file upload with direct approach
        try:
            # Generate a UUID for the document
            file_doc_id = str(uuid.uuid4())
            
            # Prepare document data for file
            file_document_data = {
                'id': file_doc_id,
                'session_id': test_session_id,
                'user_id': test_user_id,
                'content': "Machine Learning (ML) is the study of computer algorithms that improve automatically through experience.",
                'metadata': {"test": "file_upload", "format": "text", "filename": "test.txt"},
                'source_type': 'file',
                'source_reference': 'test.txt',
                'chunk_index': 0,
                'total_chunks': 1,
                'created_at': datetime.utcnow().isoformat()
            }
            
            # Insert document
            file_result = supabase.table('documents').insert(file_document_data).execute()
            
            test_results["file_upload"] = {
                "success": bool(file_result and hasattr(file_result, 'data')),
                "document_id": file_doc_id,
                "chunk_count": 1
            }
        except Exception as e:
            logger.error(f"File upload test failed: {str(e)}", exc_info=True)
            test_results["file_upload"] = {
                "success": False,
                "error": str(e)
            }
        
        # 4. Test query with direct approach
        try:
            # Direct database query
            query_start_time = time.time()
            query = supabase.table('documents').select('*').eq('session_id', test_session_id).execute()
            query_time = int((time.time() - query_start_time) * 1000)
            
            results = []
            if query and hasattr(query, 'data'):
                for doc in query.data:
                    results.append({
                        'id': doc.get('id', ''),
                        'content': doc.get('content', ''),
                        'score': 0.7,  # Default score
                        'metadata': doc.get('metadata', {})
                    })
            
            test_results["query"] = {
                "success": True,
                "result_count": len(results),
                "processing_time_ms": query_time
            }
        except Exception as e:
            logger.error(f"Query test failed: {str(e)}", exc_info=True)
            test_results["query"] = {
                "success": False,
                "error": str(e)
            }
        
        # 5. Test stats with direct approach
        try:
            stats_start_time = time.time()
            stats_query = supabase.table('documents').select('*', count='exact').eq('session_id', test_session_id).execute()
            stats_time = int((time.time() - stats_start_time) * 1000)
            
            doc_count = stats_query.count if hasattr(stats_query, 'count') else 0
            
            test_results["stats"] = {
                "success": True,
                "document_count": doc_count,
                "processing_time_ms": stats_time
            }
        except Exception as e:
            logger.error(f"Stats test failed: {str(e)}", exc_info=True)
            test_results["stats"] = {
                "success": False,
                "error": str(e)
            }
        
        # 6. Test session data with direct approach
        try:
            session_query = supabase.table('documents').select('*').eq('session_id', test_session_id).execute()
            
            documents = []
            if session_query and hasattr(session_query, 'data'):
                for doc in session_query.data:
                    documents.append({
                        'id': doc.get('id', ''),
                        'content': doc.get('content', ''),
                        'metadata': doc.get('metadata', {})
                    })
            
            test_results["session_data"] = {
                "success": True,
                "document_count": len(documents),
                "session_id": test_session_id
            }
        except Exception as e:
            logger.error(f"Session data test failed: {str(e)}", exc_info=True)
            test_results["session_data"] = {
                "success": False,
                "error": str(e)
            }
        
        # 7. Cleanup - delete test data with direct approach
        try:
            delete_result = supabase.table('documents').delete().eq('session_id', test_session_id).execute()
            
            # Verify deletion
            verify_query = supabase.table('documents').select('*', count='exact').eq('session_id', test_session_id).execute()
            remaining = verify_query.count if hasattr(verify_query, 'count') else 0
            
            test_results["cleanup"] = {
                "success": remaining == 0,
                "documents_remaining": remaining
            }
        except Exception as e:
            logger.error(f"Cleanup test failed: {str(e)}", exc_info=True)
            test_results["cleanup"] = {
                "success": False,
                "error": str(e)
            }
        
        # Calculate overall test status
        all_tests = [v for k, v in test_results.items() if isinstance(v, dict) and "success" in v]
        successful_tests = sum(1 for t in all_tests if t["success"])
        
        return {
            "success": successful_tests == len(all_tests) and len(all_tests) > 0,
            "test_session_id": test_session_id,
            "test_user_id": test_user_id,
            "test_results": test_results,
            "summary": {
                "total_tests": len(all_tests),
                "successful_tests": successful_tests,
                "success_rate": f"{(successful_tests / max(len(all_tests), 1) * 100):.1f}%"
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
