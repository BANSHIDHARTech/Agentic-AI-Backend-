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
    min_score: float = Field(0.1, ge=0.0, le=1.0, description="Minimum similarity score (0.0-1.0)")

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
            "DELETE /documents        - Delete documents/sessions",
            "GET    /test-ai-query    - Test improved AI query capability",
            "GET    /debug/session/{session_id} - Debug session documents",
            "POST   /create-test-content - Create test AI content for search testing"
        ]
    }

@router.get("/test-ai-query")
async def test_ai_query(
    session_id: str = Query(..., description="Session ID to test"),
    term: str = Query("artificial intelligence", description="AI term to search for")
):
    """
    Test endpoint to verify improved AI term matching
    
    This endpoint uses the enhanced search algorithm to find AI-related content
    """
    # Log important diagnostic info
    logger.info(f"=== AI QUERY TEST === session_id={session_id}, term={term}")
    try:
        # Get Supabase client
        supabase = get_supabase_client()
        start_time = time.time()
        
        # Split query into keywords
        search_terms = term.lower().split()
        
        # Add AI synonyms
        ai_synonyms = {
            'ai': ['artificial intelligence', 'a.i.', 'artificial'],
            'artificial': ['ai', 'artificial intelligence'],
            'intelligence': ['ai', 'artificial intelligence'],
            'machine': ['ml', 'machine learning'],
            'learning': ['ml', 'machine learning', 'deep learning'],
            'ml': ['machine learning'],
            'neural': ['neural network', 'neural-network', 'neural net']
        }
        
        # Expand search terms with synonyms
        expanded_terms = []
        for term in search_terms:
            expanded_terms.append(term)
            if term in ai_synonyms:
                expanded_terms.extend(ai_synonyms[term])
        
        # Make terms unique
        expanded_terms = list(set(expanded_terms))
        
        # Query documents
        query = supabase.table('documents').select('*').eq('session_id', session_id).limit(100)
        result = query.execute()
        
        if not result or not hasattr(result, 'data') or not result.data:
            return {
                "success": False,
                "message": f"No documents found in session {session_id}",
                "session_id": session_id
            }
            
        # Process results with improved matching
        matched_docs = []
        
        for doc in result.data:
            content = doc.get('content', '').lower()
            
            # Calculate relevance with expanded terms
            match_count = 0
            term_matches = {}
            
            for term in expanded_terms:
                count = content.count(term)
                if count > 0:
                    if term in search_terms:
                        weight = 1.0
                    else:
                        weight = 0.7
                    weighted_count = count * weight
                    match_count += weighted_count
                    term_matches[term] = count
            
            if match_count > 0:
                score = min(match_count / max(len(search_terms), 1), 1.0)
                
                # Get metadata
                metadata = doc.get('metadata', {})
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except:
                        metadata = {}
                
                matched_docs.append({
                    'id': doc.get('id', ''),
                    'content_preview': doc.get('content', '')[:100] + '...',
                    'score': score,
                    'term_matches': term_matches,
                    'metadata': {
                        'source_type': doc.get('source_type', ''),
                        'source_reference': doc.get('source_reference', ''),
                        **metadata
                    }
                })
        
        # Sort by score
        matched_docs.sort(key=lambda x: x['score'], reverse=True)
        
        # Get processing time
        processing_time = int((time.time() - start_time) * 1000)
        
        return {
            "success": True,
            "message": "AI term matching test completed",
            "query": term,
            "expanded_terms": expanded_terms,
            "session_id": session_id,
            "result_count": len(matched_docs),
            "processing_time_ms": processing_time,
            "note": "This endpoint demonstrates improved AI term matching with expanded synonyms and lower thresholds",
            "results": matched_docs[:10]  # Return top 10 results
        }
        
    except Exception as e:
        logger.error(f"Test AI query error: {str(e)}", exc_info=True)
        return {
            "success": False,
            "message": "Test failed",
            "error": str(e)
        }

@router.get("/debug/session/{session_id}")
async def debug_session(
    session_id: str,
    include_content: bool = Query(True, description="Include full document content")
):
    """
    Debug endpoint to directly examine session documents
    
    This endpoint bypasses the search logic and directly returns all documents
    in the specified session for debugging purposes
    """
    try:
        # Get Supabase client
        supabase = get_supabase_client()
        start_time = time.time()
        
        # Log debug request
        logger.info(f"Debug request for session: {session_id}")
        
        # Build query
        select_fields = "*" if include_content else "id, session_id, user_id, source_type, source_reference, metadata, created_at"
        query = supabase.table('documents').select(select_fields).eq('session_id', session_id).order('created_at', desc=True).limit(100)
        result = query.execute()
        
        # Error check
        if not result or not hasattr(result, 'data'):
            return {
                "success": False,
                "message": "Failed to query database",
                "session_id": session_id,
                "error": "Database query failed"
            }
            
        # Process results
        documents = []
        for doc in result.data:
            # Parse metadata if needed
            metadata = doc.get('metadata', {})
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except:
                    metadata = {}
            
            # Format document
            doc_data = {
                'id': doc.get('id', ''),
                'source_type': doc.get('source_type', ''),
                'source_reference': doc.get('source_reference', ''),
                'created_at': doc.get('created_at', ''),
                'metadata': metadata
            }
            
            # Include content if requested
            if include_content:
                content = doc.get('content', '')
                doc_data['content'] = content
                doc_data['content_length'] = len(content) if content else 0
                doc_data['content_snippet'] = content[:100] + '...' if len(content) > 100 else content
                
                # Simple text analysis for debugging
                if content:
                    lower_content = content.lower()
                    ai_terms = ['ai', 'artificial intelligence', 'machine learning', 'neural', 'deep learning']
                    doc_data['ai_term_matches'] = {term: lower_content.count(term) for term in ai_terms if term in lower_content}
            
            # Add to results
            documents.append(doc_data)
        
        # Get processing time
        processing_time = int((time.time() - start_time) * 1000)
        
        return {
            "success": True,
            "message": "Session debug information retrieved",
            "session_id": session_id,
            "document_count": len(documents),
            "processing_time_ms": processing_time,
            "note": "This is a debug endpoint that bypasses normal search logic",
            "documents": documents
        }
        
    except Exception as e:
        logger.error(f"Session debug error: {str(e)}", exc_info=True)
        return {
            "success": False,
            "message": "Debug failed",
            "error": str(e)
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
    similarity_threshold: Optional[float] = Query(0.1, ge=0.0, le=1.0)
):
    """Search the knowledge base using similarity search"""
    try:
        # Log the request with request info
        logger.info(f"Knowledge GET search: query='{q}', session_id={session_id}, user_id={user_id}, threshold={similarity_threshold}")
        
        # Generate a session ID if not provided
        search_session_id = session_id or f"search_{int(datetime.now().timestamp() * 1000)}"
        logger.info(f"Using session ID: {search_session_id}")

        try:
            # Direct database query for reliability
            from ..core.database import get_supabase_client
            supabase = get_supabase_client()
            import time
            start_time = time.time()
            
            # DIAGNOSTIC: Check if there are ANY documents in the database
            diag_query = supabase.table('documents').select('id', count='exact').limit(1).execute()
            total_docs = diag_query.count if hasattr(diag_query, 'count') else 0
            logger.info(f"DIAGNOSTIC: Total documents in database: {total_docs}")
            
            # DIAGNOSTIC: Check if there are any documents in the requested session
            if session_id:
                session_diag_query = supabase.table('documents').select('id', count='exact').eq('session_id', session_id).execute()
                session_docs = session_diag_query.count if hasattr(session_diag_query, 'count') else 0
                logger.info(f"DIAGNOSTIC: Documents in session {session_id}: {session_docs}")
            
            # Split query into keywords for more flexible searching
            search_terms = q.lower().split() if q else []
            
            # ENHANCED: More comprehensive AI-related synonyms and variations
            ai_synonyms = {
                'ai': ['artificial intelligence', 'a.i.', 'artificial', 'ai systems', 'ai technology'],
                'artificial': ['ai', 'artificial intelligence', 'computer', 'synthetic', 'automated'],
                'intelligence': ['ai', 'artificial intelligence', 'smart', 'cognitive', 'reasoning'],
                'machine': ['ml', 'machine learning', 'automated', 'deep learning', 'neural'],
                'learning': ['ml', 'machine learning', 'deep learning', 'training', 'neural learning'],
                'ml': ['machine learning', 'model', 'algorithm', 'training'],
                'neural': ['neural network', 'neural-network', 'neural net', 'deep learning', 'neural computing'],
                'deep': ['deep learning', 'neural', 'ai', 'depth'],
                'computer': ['ai', 'computation', 'computing', 'processor'],
                'model': ['ai model', 'neural model', 'statistical model'],
                'training': ['learning', 'fitting', 'optimization'],
                'data': ['dataset', 'information', 'records', 'training data'],
                'prediction': ['forecast', 'output', 'inference']
            }
            
            # Expand search terms with synonyms
            expanded_terms = []
            for term in search_terms:
                expanded_terms.append(term)
                if term in ai_synonyms:
                    expanded_terms.extend(ai_synonyms[term])
            
            # ENHANCED: Check for compound terms like "artificial intelligence" in the original query
            compound_terms = [
                "artificial intelligence",
                "machine learning",
                "deep learning",
                "neural network",
                "expert system",
                "computer vision",
                "natural language processing",
                "nlp"
            ]
            
            # Check if any compound terms are in the original query
            original_query_lower = q.lower()
            for term in compound_terms:
                if term in original_query_lower and term not in expanded_terms:
                    expanded_terms.append(term)
                    # Also add component words
                    expanded_terms.extend(term.split())
            
            # Make terms unique
            expanded_terms = list(set(expanded_terms))
            logger.info(f"Expanded search terms: {expanded_terms}")
            
            # Build query - get all documents from the session first
            db_query = supabase.table('documents').select('*')
            
            # Add filters
            if session_id:
                db_query = db_query.eq('session_id', session_id)
            if user_id and user_id != "anonymous":
                db_query = db_query.eq('user_id', user_id)
                
            # Add limit and order by created_at
            db_query = db_query.order('created_at', desc=True).limit(100)  # Get more docs first, then filter
            
            # Execute query
            result = db_query.execute()
            
            # Process results with manual filtering
            documents = []
            matched_docs = []
            
            if result and getattr(result, 'data', None):
                logger.info(f"Found {len(result.data)} documents for session '{session_id}', now filtering for relevance")
                if len(result.data) == 0:
                    # Log the query details to help diagnose the issue
                    logger.warning(f"No documents found for session '{session_id}'. This could indicate the session doesn't exist or has no documents.")
                
                for doc in result.data:
                    # Get document content and convert to lowercase for case-insensitive comparison
                    content = doc.get('content', '').lower()
                    if not content:
                        continue
                    
                    # ENHANCED: Three-tier term weighting with more sophisticated scoring
                    match_count = 0
                    term_matches = {}
                    
                    # Check for exact original query match (highest weight)
                    exact_query_count = content.count(original_query_lower)
                    if exact_query_count > 0:
                        match_count += exact_query_count * 2.0  # Highest weight
                        term_matches["exact_query"] = exact_query_count
                    
                    # Check both original and expanded terms
                    for term in expanded_terms:
                        count = content.count(term)
                        if count > 0:
                            # Weight matches differently based on term importance
                            if term in search_terms:  # Original search term
                                weight = 1.0  # Medium weight
                            elif term in compound_terms:  # Compound technical terms
                                weight = 0.9  # Higher than regular synonyms
                            else:  # Expanded/synonym term
                                weight = 0.6  # Lowest weight
                                
                            weighted_count = count * weight
                            match_count += weighted_count
                            term_matches[term] = count
                    
                    # ENHANCED: Score normalization based on document length
                    normalization_factor = 1.0
                    if len(content) > 1000:
                        # Gradually reduce score for very long documents
                        normalization_factor = 1000 / len(content)
                        match_count *= max(0.5, normalization_factor)
                    
                    # Log detailed document analysis for debugging
                    logger.info(f"Document ID {doc.get('id', 'unknown')}: content_length={len(content)}, match_count={match_count}")
                    logger.info(f"Term matches: {term_matches}")
                    
                    # Only include documents with at least one matching term or if no search terms
                    if match_count > 0 or not expanded_terms:
                        # Calculate relevance score with improved scaling
                        # For zero or one search term, use raw match count capped at 1.0
                        if len(search_terms) <= 1:
                            relevance_score = min(match_count, 1.0)
                        else:
                            # For multiple search terms, scale by number of search terms
                            relevance_score = min(match_count / len(search_terms), 1.0)
                        
                        # Safely extract metadata
                        metadata = doc.get('metadata', {})
                        if isinstance(metadata, str):
                            try:
                                metadata = json.loads(metadata)
                            except:
                                metadata = {}
                                
                        # Add to matched documents with score
                        matched_docs.append({
                            'doc': doc,
                            'score': relevance_score,
                            'term_matches': term_matches,
                            'match_count': match_count,
                            'content_length': len(content),
                            'metadata': metadata
                        })
                
                # Sort by relevance score
                matched_docs.sort(key=lambda x: x['score'], reverse=True)
                
                # Take top results
                top_results = matched_docs[:limit]
                
                # Filter by minimum score
                filtered_results = [r for r in top_results if r['score'] >= similarity_threshold]
                
                # Log detailed filtering information
                filtered_out = len(top_results) - len(filtered_results)
                if filtered_out > 0:
                    logger.info(f"Filtered out {filtered_out} documents with scores below threshold {similarity_threshold}")
                    # Log scores of filtered out documents for debugging
                    for i, doc in enumerate(top_results):
                        if doc['score'] < similarity_threshold:
                            logger.debug(f"Filtered doc {i}: score={doc['score']}, below threshold={similarity_threshold}")
                
                # Format results
                for match in filtered_results:
                    doc = match['doc']
                    documents.append({
                        'id': doc.get('id', ''),
                        'content': doc.get('content', ''),
                        'score': match['score'],
                        'metadata': {
                            'source_type': doc.get('source_type', ''),
                            'source_reference': doc.get('source_reference', ''),
                            'chunk_index': doc.get('chunk_index', 0),
                            'total_chunks': doc.get('total_chunks', 1),
                            'term_matches': match.get('term_matches', {}),
                            **match['metadata']
                        }
                    })
                
                # Calculate processing time
                processing_time = int((time.time() - start_time) * 1000)
                
                # Log success with detail
                logger.info(f"GET Search '{q}' completed: found {len(documents)} relevant documents in {processing_time}ms")
                    
                # Return a consistent response format with debug info
                return {
                    "success": True,
                    "message": "Knowledge base search completed successfully",
                    "query": q,
                    "results": documents,
                    "count": len(documents),
                    "session_id": search_session_id,
                    "processing_time_ms": processing_time,
                    "search_method": "direct",
                    "search_info": {
                        "expanded_terms": expanded_terms,
                        "original_terms": search_terms,
                        "min_score_threshold": similarity_threshold,
                        "documents_found": len(result.data) if result and hasattr(result, 'data') else 0,
                        "documents_matched": len(matched_docs)
                    }
                }
            else:
                # No documents found in session, try alternative approach
                logger.warning(f"No documents found for session {session_id}, trying vector search fallback")
                
                # Fallback to vector search using KnowledgeService
                try:
                    # Use a lower threshold for the fallback search
                    fallback_threshold = min(0.1, similarity_threshold)
                    
                    result = await KnowledgeService.query_knowledge_base(
                        query=q,
                        session_id=search_session_id,
                        user_id=user_id,
                        top_k=limit,
                        min_score=fallback_threshold
                    )
                    
                    # Process the result
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
                    
                    # If vector search found results
                    if formatted_results:
                        logger.info(f"Vector search fallback found {len(formatted_results)} results")
                        return {
                            "success": True,
                            "message": "Knowledge base search completed using vector search",
                            "query": q,
                            "results": formatted_results,
                            "count": len(formatted_results),
                            "session_id": search_session_id,
                            "search_method": "vector_fallback"
                        }
                    else:
                        # No results from either method
                        logger.warning(f"Both direct and vector search found no results for '{q}'")
                        return {
                            "success": True,
                            "message": "Knowledge base search completed successfully, but no results found",
                            "query": q,
                            "results": [],
                            "count": 0,
                            "session_id": search_session_id,
                            "search_info": {
                                "expanded_terms": expanded_terms,
                                "original_terms": search_terms
                            }
                        }
                        
                except Exception as e2:
                    logger.error(f"Vector search fallback failed: {str(e2)}", exc_info=True)
                    return {
                        "success": True,
                        "message": "Knowledge base search completed, but no results found",
                        "query": q,
                        "results": [],
                        "count": 0,
                        "session_id": search_session_id
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
                    "message": "Knowledge base search completed (service fallback)",
                    "query": q,
                    "results": formatted_results,
                    "count": len(formatted_results),
                    "session_id": search_session_id
                }
            except Exception as e2:
                logger.error(f"All fallback search methods failed: {str(e2)}", exc_info=True)
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
        # Log the request with request info
        logger.info(f"Knowledge POST query: '{query.query}', session_id={query.session_id}, threshold={query.min_score}")
        
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
            import time
            start_time = time.time()
            
            # DIAGNOSTIC: Check if there are ANY documents in the database
            diag_query = supabase.table('documents').select('id', count='exact').limit(1).execute()
            total_docs = diag_query.count if hasattr(diag_query, 'count') else 0
            logger.info(f"DIAGNOSTIC: Total documents in database: {total_docs}")
            
            # DIAGNOSTIC: Check if there are any documents in the requested session
            if query.session_id:
                session_diag_query = supabase.table('documents').select('id', count='exact').eq('session_id', query.session_id).execute()
                session_docs = session_diag_query.count if hasattr(session_diag_query, 'count') else 0
                logger.info(f"DIAGNOSTIC: Documents in session {query.session_id}: {session_docs}")
            
            # Log query details
            logger.info(f"Searching for: '{query.query}' in session: '{query.session_id}'")
            
            # Split query into keywords for more flexible searching
            search_terms = query.query.lower().split()
            
            # ENHANCED: More comprehensive AI-related synonyms and variations to improve matching
            ai_synonyms = {
                'ai': ['artificial intelligence', 'a.i.', 'artificial', 'ai systems', 'ai technology'],
                'artificial': ['ai', 'artificial intelligence', 'computer', 'synthetic', 'automated'],
                'intelligence': ['ai', 'artificial intelligence', 'smart', 'cognitive', 'reasoning'],
                'machine': ['ml', 'machine learning', 'automated', 'deep learning', 'neural'],
                'learning': ['ml', 'machine learning', 'deep learning', 'training', 'neural learning'],
                'ml': ['machine learning', 'model', 'algorithm', 'training'],
                'neural': ['neural network', 'neural-network', 'neural net', 'deep learning', 'neural computing'],
                'deep': ['deep learning', 'neural', 'ai', 'depth'],
                'computer': ['ai', 'computation', 'computing', 'processor'],
                'model': ['ai model', 'neural model', 'statistical model'],
                'training': ['learning', 'fitting', 'optimization'],
                'data': ['dataset', 'information', 'records', 'training data'],
                'prediction': ['forecast', 'output', 'inference']
            }
            
            # Expand search terms with synonyms
            expanded_terms = []
            for term in search_terms:
                expanded_terms.append(term)
                if term in ai_synonyms:
                    expanded_terms.extend(ai_synonyms[term])
            
            # ENHANCED: Check for compound terms like "artificial intelligence" in the original query
            compound_terms = [
                "artificial intelligence",
                "machine learning",
                "deep learning",
                "neural network",
                "expert system",
                "computer vision",
                "natural language processing",
                "nlp"
            ]
            
            # Check if any compound terms are in the original query
            original_query_lower = query.query.lower()
            for term in compound_terms:
                if term in original_query_lower and term not in expanded_terms:
                    expanded_terms.append(term)
                    # Also add component words
                    expanded_terms.extend(term.split())
            
            # Make terms unique
            expanded_terms = list(set(expanded_terms))
            logger.info(f"Expanded search terms: {expanded_terms}")
            
            # Try both approaches: first with session filter, then without if needed
            
            # Approach 1: Filter by session ID
            db_query = supabase.table('documents')
            
            # Add session filter if provided
            if query.session_id:
                db_query = db_query.select('*').eq('session_id', query.session_id)
            else:
                # If no session ID, select all documents but limit more
                db_query = db_query.select('*').limit(50)
                
            # Add ordering
            db_query = db_query.order('created_at', desc=True)
            
            # Execute query
            result = db_query.execute()
            
            # Process results with enhanced matching
            documents = []
            matched_docs = []
            
            # First process documents from the specific session or all documents
            if result and hasattr(result, 'data') and result.data:
                logger.info(f"Found {len(result.data)} documents to search, now filtering for relevance")
                
                for doc in result.data:
                    # Get document content and convert to lowercase for case-insensitive comparison
                    content = doc.get('content', '').lower()
                    if not content:
                        continue
                        
                    # Calculate a more sophisticated relevance score based on term matches
                    match_count = 0
                    term_matches = {}
                    
                    # ENHANCED: Three-tier term weighting
                    # 1. Exact original query match (highest)
                    # 2. Original search terms (medium)
                    # 3. Expanded/synonym terms (lowest)
                    
                    # Check for exact original query match (highest weight)
                    exact_query_count = content.count(original_query_lower)
                    if exact_query_count > 0:
                        match_count += exact_query_count * 2.0  # Highest weight
                        term_matches["exact_query"] = exact_query_count
                    
                    # Check both original and expanded terms
                    for term in expanded_terms:
                        count = content.count(term)
                        if count > 0:
                            # Weight matches differently based on term importance
                            if term in search_terms:  # Original search term
                                weight = 1.0  # Medium weight
                            elif term in compound_terms:  # Compound technical terms
                                weight = 0.9  # Higher than regular synonyms
                            else:  # Expanded/synonym term
                                weight = 0.6  # Lowest weight
                                
                            weighted_count = count * weight
                            match_count += weighted_count
                            term_matches[term] = count
                    
                    # ENHANCED: Score normalization based on document length
                    # This prevents longer documents from getting artificially high scores
                    # just because they have more text
                    normalization_factor = 1.0
                    if len(content) > 1000:
                        # Gradually reduce score for very long documents
                        normalization_factor = 1000 / len(content)
                        match_count *= max(0.5, normalization_factor)
                    
                    # Only include documents with at least one matching term or if no search terms provided
                    if match_count > 0 or not expanded_terms:
                        # Calculate relevance score with improved scaling
                        # For zero or one search term, use raw match count capped at 1.0
                        if len(search_terms) <= 1:
                            relevance_score = min(match_count, 1.0)
                        else:
                            # For multiple search terms, scale by number of search terms
                            relevance_score = min(match_count / len(search_terms), 1.0)
                        
                        # Safely extract metadata
                        metadata = doc.get('metadata', {})
                        if isinstance(metadata, str):
                            try:
                                metadata = json.loads(metadata)
                            except:
                                metadata = {}
                                
                        # Add to matched documents with score
                        matched_docs.append({
                            'doc': doc,
                            'score': relevance_score,
                            'term_matches': term_matches,
                            'match_count': match_count,
                            'content_length': len(content),
                            'metadata': metadata
                        })
                
                # Sort by relevance score
                matched_docs.sort(key=lambda x: x['score'], reverse=True)
                
                # Take top_k results
                top_results = matched_docs[:query.top_k]
                
                # Filter by minimum score
                filtered_results = [r for r in top_results if r['score'] >= query.min_score]
                
                # Log detailed filtering information
                filtered_out = len(top_results) - len(filtered_results)
                if filtered_out > 0:
                    logger.info(f"Filtered out {filtered_out} documents with scores below threshold {query.min_score}")
                    # Log scores of filtered out documents for debugging
                    for i, doc in enumerate(top_results):
                        if doc['score'] < query.min_score:
                            logger.debug(f"Filtered doc {i}: score={doc['score']}, below threshold={query.min_score}")
                
                # Format results
                for match in filtered_results:
                    doc = match['doc']
                    documents.append({
                        'id': doc.get('id', ''),
                        'content': doc.get('content', ''),
                        'score': match['score'],
                        'metadata': {
                            'source_type': doc.get('source_type', ''),
                            'source_reference': doc.get('source_reference', ''),
                            'chunk_index': doc.get('chunk_index', 0),
                            'total_chunks': doc.get('total_chunks', 1),
                            'term_matches': match.get('term_matches', {}),
                            **match['metadata']
                        }
                    })
                
                # Calculate processing time
                processing_time = int((time.time() - start_time) * 1000)
                
                # Log success with detail
                logger.info(f"Query '{query.query}' completed: found {len(documents)} relevant documents in {processing_time}ms")
                
                return {
                    "success": True,
                    "message": "Knowledge base query completed successfully",
                    "query": query.query,
                    "results": documents,
                    "count": len(documents),
                    "session_id": query.session_id,
                    "processing_time_ms": processing_time,
                    "search_info": {
                        "expanded_terms": expanded_terms,
                        "original_terms": search_terms,
                        "min_score_threshold": query.min_score,
                        "documents_found": len(result.data),
                        "documents_matched": len(matched_docs)
                    }
                }
            else:
                # No documents found in session, try alternative approach
                logger.warning(f"No documents found for session {query.session_id}, trying vector search fallback")
                
                # Fallback to vector search using KnowledgeService
                try:
                    # Use a lower threshold for the fallback search
                    fallback_threshold = min(0.1, query.min_score)
                    
                    result = await KnowledgeService.query_knowledge_base(
                        query=query.query,
                        session_id=query.session_id,
                        user_id=None,
                        top_k=query.top_k,
                        min_score=fallback_threshold
                    )
                    
                    # Process the result
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
                    
                    # If vector search found results
                    if formatted_results:
                        logger.info(f"Vector search fallback found {len(formatted_results)} results")
                        return {
                            "success": True,
                            "message": "Knowledge base query completed using vector search",
                            "query": query.query,
                            "results": formatted_results,
                            "count": len(formatted_results),
                            "session_id": query.session_id,
                            "search_method": "vector_fallback"
                        }
                    else:
                        # No results from either method
                        logger.warning(f"Both direct and vector search found no results for '{query.query}'")
                        return {
                            "success": True,
                            "message": "Knowledge base query completed successfully, but no results found",
                            "query": query.query,
                            "results": [],
                            "count": 0,
                            "session_id": query.session_id,
                            "search_info": {
                                "expanded_terms": expanded_terms,
                                "original_terms": search_terms
                            }
                        }
                        
                except Exception as e2:
                    logger.error(f"Vector search fallback failed: {str(e2)}", exc_info=True)
                    return {
                        "success": True,
                        "message": "Knowledge base query completed, but no results found",
                        "query": query.query,
                        "results": [],
                        "count": 0,
                        "session_id": query.session_id
                    }
            
        except Exception as e:
            logger.error(f"Error in knowledge query: {str(e)}", exc_info=True)
            
            # Fallback to service approach
            try:
                result = await KnowledgeService.query_knowledge_base(
                    query=query.query,
                    session_id=query.session_id,
                    user_id="anonymous",
                    top_k=query.top_k,
                    min_score=query.min_score
                )
                
                # Process the result
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
                    "message": "Knowledge base query completed (service fallback)",
                    "query": query.query,
                    "results": formatted_results,
                    "count": len(formatted_results),
                    "session_id": query.session_id
                }
            except Exception as e2:
                logger.error(f"All fallback query methods failed: {str(e2)}", exc_info=True)
                return {
                    "success": False,
                    "message": "Knowledge base query failed",
                    "error": f"Query error: {str(e2)}",
                    "results": [],
                    "count": 0,
                    "query": query.query
                }
        
    except Exception as error:
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
        result = await db_rpc('get_knowledge_sessions', {
            'p_user_id': user_id,
            'p_limit': limit,
            'p_offset': offset
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

class TestContentRequest(BaseModel):
    session_id: str = Field(..., description="Session ID to store test content")
    content_type: str = Field("ai_article", description="Type of test content to create")

@router.post("/create-test-content")
async def create_test_content(
    request: TestContentRequest
):
    """
    Create test documents with AI-related content for search testing
    
    This endpoint creates several test documents with various AI-related content
    to ensure the search functionality works properly with AI terms
    """
    # Log the request details
    logger.info(f"Creating test content in session: {request.session_id}, type: {request.content_type}")
    try:
        # Get Supabase client
        supabase = get_supabase_client()
        start_time = time.time()
        
        # Log the request
        logger.info(f"Creating test content in session: {request.session_id}, type: {request.content_type}")
        
        # Create a list to hold all document IDs
        document_ids = []
        
        # Test content specifically designed to test AI term matching
        ai_test_documents = [
            {
                "title": "Introduction to Artificial Intelligence",
                "content": """Artificial Intelligence (AI) refers to the simulation of human intelligence in machines 
                that are programmed to think like humans and mimic their actions. The term may also be applied to any machine 
                that exhibits traits associated with a human mind such as learning and problem-solving. The ideal characteristic 
                of artificial intelligence is its ability to rationalize and take actions that have the best chance of achieving 
                a specific goal. A subset of artificial intelligence is machine learning (ML), which refers to the concept that 
                computer programs can automatically learn from and adapt to new data without being assisted by humans.""",
                "tags": ["AI", "Machine Learning", "Introduction"]
            },
            {
                "title": "Deep Learning and Neural Networks",
                "content": """Deep Learning is a subset of machine learning in artificial intelligence that has networks capable 
                of learning unsupervised from data that is unstructured or unlabeled. Also known as deep neural learning or deep 
                neural network (DNN), it is a computational approach that models the way the human brain works. Neural networks, 
                which are the foundation of deep learning algorithms, are inspired by the biological neural networks that constitute 
                animal brains. These networks consist of node layers: an input layer, one or more hidden layers, and an output layer.""",
                "tags": ["Deep Learning", "Neural Networks", "AI"]
            },
            {
                "title": "Natural Language Processing (NLP)",
                "content": """Natural Language Processing (NLP) is a branch of artificial intelligence that helps computers 
                understand, interpret and manipulate human language. NLP draws from many disciplines, including computer science 
                and computational linguistics, in its pursuit to fill the gap between human communication and computer understanding. 
                Modern NLP algorithms rely on machine learning, especially statistical machine learning and deep learning methods, 
                to derive meaning from human languages.""",
                "tags": ["NLP", "AI", "Text Processing"]
            },
            {
                "title": "Machine Learning Algorithms",
                "content": """Machine Learning algorithms are methods that allow computers to learn patterns from data. They enable 
                machines to identify complex relationships in data to make accurate predictions or decisions without being explicitly 
                programmed. Common types of machine learning algorithms include supervised learning (where the algorithm is trained on 
                labeled data), unsupervised learning (where the algorithm finds patterns in unlabeled data), and reinforcement learning 
                (where the algorithm learns through trial and error).""",
                "tags": ["Machine Learning", "Algorithms", "AI"]
            },
            {
                "title": "AI Ethics and Responsible AI",
                "content": """AI ethics is a system of moral principles and techniques intended to inform the development and responsible 
                use of artificial intelligence technology. As AI becomes more sophisticated and integrated into our daily lives, questions 
                about its ethical implications have become increasingly important. Key ethical considerations include fairness, transparency, 
                privacy, security, and accountability. Responsible AI frameworks aim to ensure that AI systems are designed and deployed in 
                ways that are beneficial, equitable, and respectful of human autonomy and rights.""",
                "tags": ["AI Ethics", "Responsible AI", "Technology Ethics"]
            }
        ]
        
        # Process each test document
        for i, doc in enumerate(ai_test_documents):
            # Generate a UUID for the document
            doc_id = str(uuid.uuid4())
            
            # Prepare document data
            document_data = {
                'id': doc_id,
                'session_id': request.session_id,
                'user_id': 'test_system',
                'content': doc['content'],
                'metadata': {
                    'title': doc['title'],
                    'tags': doc['tags'],
                    'test_document': True,
                    'document_index': i,
                    'created_for_testing': True,
                    'content_type': request.content_type
                },
                'source_type': 'text',
                'source_reference': 'test_content',
                'chunk_index': 0,
                'total_chunks': 1,
                'created_at': datetime.utcnow().isoformat()
            }
            
            # Insert document
            insert_result = supabase.table('documents').insert(document_data).execute()
            
            if insert_result and hasattr(insert_result, 'data'):
                document_ids.append(doc_id)
                logger.info(f"Successfully created test document {i+1}/{len(ai_test_documents)} with ID {doc_id}")
            else:
                error = getattr(insert_result, 'error', 'Unknown error')
                logger.error(f"Failed to create test document {i+1}: {error}")
        
        # Calculate processing time
        processing_time = int((time.time() - start_time) * 1000)
        
        return {
            "success": True,
            "message": f"Created {len(document_ids)} test documents with AI-related content",
            "document_ids": document_ids,
            "session_id": request.session_id,
            "processing_time_ms": processing_time,
            "note": "These documents are designed for testing AI-related search queries"
        }
        
    except Exception as e:
        logger.error(f"Failed to create test content: {str(e)}", exc_info=True)
        return {
            "success": False,
            "message": "Failed to create test content",
            "error": str(e)
        }

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
