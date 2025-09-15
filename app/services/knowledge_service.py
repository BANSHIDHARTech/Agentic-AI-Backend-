"""
KnowledgeService - Comprehensive Knowledge Base Management

Provides advanced knowledge base functionality including:
- Document ingestion from multiple sources (text, URL, PDF, files)
- Intelligent text chunking
- Vector embedding generation using OpenAI
- Similarity search with configurable thresholds
- Session tracking and analytics
- Comprehensive error handling and logging
"""

import os
import time
import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
from uuid import uuid4
from pathlib import Path

import aiohttp
import aiofiles
import httpx
from fastapi import UploadFile, HTTPException, status
from pydantic import BaseModel, Field, validator
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Document processing
from PyPDF2 import PdfReader
from docx import Document
from bs4 import BeautifulSoup
import markdown
from io import BytesIO
import re

# Core imports
from ..core.database import supabase, db_insert, db_update, db_select, db_delete, db_rpc
from ..core.document_processor import DocumentProcessor
from ..core.knowledge_utils import safe_data, is_valid_url, clean_html, extract_text_from_html, chunk_text
from .embeddings_service import embedding_service

# Configure logging
logger = logging.getLogger(__name__)

class KnowledgeService:
    """Knowledge base management service with advanced document processing and search capabilities"""
    
    # Embedding model configuration
    EMBEDDING_MODEL = "text-embedding-3-small"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    @classmethod
    async def get_embedding(cls, text: str) -> List[float]:
        """Generate embedding for the given text using OpenAI"""
        try:
            if not text or not text.strip():
                raise ValueError("Cannot generate embedding for empty text")
                
            # Use OpenAI's embedding API
            import openai
            response = await openai.Embedding.create(
                model=cls.EMBEDDING_MODEL,
                input=text
            )
            return response['data'][0]['embedding']
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise ValueError(f"Failed to generate embedding: {str(e)}")
    
    @classmethod
    async def upload_document(cls, upload_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Upload and process document from various sources
        
        Args:
            upload_data: Dictionary containing:
                - type: 'text', 'url', or 'file'
                - content/url/file: Content based on type
                - session_id: Session identifier
                - user_id: User identifier
                - metadata: Optional metadata
                
        Returns:
            Dictionary with upload results
        """
        start_time = time.time()
        
        try:
            # Validate required fields
            upload_type = upload_data.get('type')
            if not upload_type:
                raise ValueError("Missing required field 'type' in upload data")
            
            # Ensure session_id is present
            if not upload_data.get('session_id'):
                upload_data['session_id'] = f"session_{int(time.time())}_{uuid4().hex[:8]}"
                
            # Log the upload attempt with detailed info
            logger.info(f"Starting document upload of type: {upload_type}")
            logger.info(f"Session ID: {upload_data.get('session_id')}, User ID: {upload_data.get('user_id')}")
            
            if upload_type == 'text':
                content = upload_data.get('content', '')
                logger.info(f"Text upload content length: {len(content)} chars")
                if not content or not content.strip():
                    raise ValueError("Empty content provided for text upload")
            
            # Process based on upload type with enhanced error catching
            try:
                if upload_type == 'text':
                    return await cls._process_text_upload(upload_data)
                elif upload_type == 'url':
                    return await cls._process_url_upload(upload_data)
                elif upload_type == 'file':
                    return await cls._process_file_upload(upload_data)
                else:
                    raise ValueError(f"Unsupported upload type: {upload_type}")
            except Exception as process_error:
                # Log detailed error information
                logger.error(f"Document processing error for {upload_type} upload: {str(process_error)}", exc_info=True)
                
                # Re-raise with more context
                raise ValueError(f"Document processing failed for {upload_type}: {str(process_error)}")
                
        except Exception as e:
            logger.error(f"Document upload failed: {str(e)}", exc_info=True)
            # Include timing info in the error
            processing_time = int((time.time() - start_time) * 1000)
            error_details = {
                "error": str(e),
                "error_type": type(e).__name__,
                "processing_time_ms": processing_time,
                "upload_type": upload_data.get('type'),
                "session_id": upload_data.get('session_id')
            }
            logger.error(f"Upload failed with details: {error_details}")
            raise
    
    @classmethod
    async def _process_text_upload(cls, upload_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process direct text upload"""
        # Extract and validate text content
        text_content = upload_data.get('content', '')
        if not text_content or not text_content.strip():
            raise ValueError("No text content provided")
            
        # Extract other parameters with defaults
        metadata = upload_data.get('metadata', {})
        session_id = upload_data.get('session_id')
        user_id = upload_data.get('user_id', 'anonymous')
        
        # Add debug metadata
        debug_metadata = {
            **metadata,
            'upload_time': datetime.utcnow().isoformat(),
            'content_length': len(text_content),
            'upload_method': 'api_text_upload'
        }
        
        logger.info(f"Processing text upload: {len(text_content)} chars, session={session_id}")
        
        # Process the content with enhanced error tracking
        try:
            result = await cls._process_content(
                content=text_content,
                source_type='text',
                source_reference='direct_input',
                metadata=debug_metadata,
                session_id=session_id,
                user_id=user_id
            )
            
            # Log success details
            doc_count = len(result.get('document_ids', []))
            logger.info(f"Text upload processed successfully: {doc_count} documents created")
            
            return result
            
        except Exception as e:
            # Log detailed error and re-raise
            logger.error(f"Text upload processing failed: {str(e)}", exc_info=True)
            raise ValueError(f"Failed to process text content: {str(e)}")
        
    @classmethod
    async def _process_url_upload(cls, upload_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process URL upload by fetching and extracting content"""
        url = upload_data.get('url')
        if not url:
            raise ValueError("No URL provided")
            
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        raise ValueError(f"Failed to fetch URL: {response.status}")
                        
                    content_type = response.headers.get('content-type', '')
                    if 'text/html' in content_type:
                        text = await response.text()
                        # Clean HTML content
                        text = await clean_html(text)
                    else:
                        text = await response.text()
                        
            return await cls._process_content(
                content=text,
                source_type='url',
                source_reference=url,
                metadata=upload_data.get('metadata', {}),
                session_id=upload_data.get('session_id'),
                user_id=upload_data.get('user_id')
            )
            
        except Exception as e:
            logger.error(f"URL processing failed: {str(e)}")
            raise ValueError(f"Failed to process URL: {str(e)}")
    
    @classmethod
    async def _process_file_upload(cls, upload_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process file upload with support for multiple file types"""
        file = upload_data.get('file')
        if not file:
            raise ValueError("No file provided")
            
        try:
            # Read file content
            file_content = await file.read()
            
            # Process with document processor
            processor = DocumentProcessor()
            file_type = processor.detect_file_type(file_content, file.filename)
            text_content = processor.extract_text(file_content, file_type)
            
            return await cls._process_content(
                content=text_content,
                source_type='file',
                source_reference=file.filename,
                metadata={
                    **upload_data.get('metadata', {}),
                    'file_type': file_type,
                    'file_name': file.filename,
                    'file_size': len(file_content)
                },
                session_id=upload_data.get('session_id'),
                user_id=upload_data.get('user_id')
            )
            
        except Exception as e:
            logger.error(f"File processing failed: {str(e)}")
            raise ValueError(f"Failed to process file: {str(e)}")
    
    @classmethod
    async def _process_content(
        cls,
        content: str,
        source_type: str,
        source_reference: str,
        metadata: Dict[str, Any],
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process and store document content with chunking and embeddings"""
        try:
            # Generate session ID if not provided
            if not session_id:
                session_id = f"sess_{int(time.time())}_{uuid4().hex[:8]}"
                
            # Process content with chunking
            processor = DocumentProcessor()
            chunks = processor.chunk_text(
                content,
                chunk_size=cls.CHUNK_SIZE,
                chunk_overlap=cls.CHUNK_OVERLAP
            )
            
            if not chunks:
                raise ValueError("No valid content could be extracted")
            
            # Prepare documents for storage
            documents = []
            chunk_embeddings = []
            
            for i, chunk in enumerate(chunks):
                doc_id = str(uuid4())
                document = {
                    'id': doc_id,
                    'session_id': session_id,
                    'user_id': user_id,
                    'content': chunk['text'],
                    'source_type': source_type,
                    'source_reference': source_reference,
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'metadata': {
                        **metadata,
                        'chunk_size': len(chunk['text']),
                        'created_at': datetime.utcnow().isoformat()
                    },
                    'created_at': datetime.utcnow().isoformat()
                }
                documents.append(document)
                chunk_embeddings.append(chunk['text'])
            
            # Generate embeddings for all chunks
            embeddings = await embedding_service.get_embeddings(chunk_embeddings)
            
            # Store documents and embeddings
            for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
                doc['embedding'] = embedding
                
                # Insert into database
                result = await db_insert('documents', doc)
                if not result or 'error' in result:
                    logger.error(f"Failed to store chunk {i}: {result.get('error')}")
            
            # Log the upload
            await cls._log_knowledge_activity(
                'document_uploaded',
                {
                    'session_id': session_id,
                    'user_id': user_id,
                    'source_type': source_type,
                    'num_chunks': len(chunks),
                    'total_size': sum(len(c['text']) for c in chunks)
                }
            )
            
            return {
                'success': True,
                'session_id': session_id,
                'num_chunks': len(chunks),
                'total_size': sum(len(c['text']) for c in chunks)
            }
            
        except Exception as e:
            logger.error(f"Content processing failed: {str(e)}", exc_info=True)
            raise
    
    @classmethod
    async def query_knowledge_base(
        cls,
        query: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        top_k: int = 5,
        min_score: float = 0.5
    ) -> Dict[str, Any]:
        """Query the knowledge base using semantic search"""
        try:
            if not query or not query.strip():
                raise ValueError("Query cannot be empty")
            
            # Try to get all documents first as a fallback
            filter_conditions = {}
            if session_id:
                filter_conditions['session_id'] = session_id
            if user_id:
                filter_conditions['user_id'] = user_id
                
            try:
                # Get query embedding - first try embedding-based search
                query_embedding = await embedding_service.get_embedding(query)
                
                # Perform vector similarity search using RPC
                result = await db_rpc('match_documents', {
                    'query_embedding': query_embedding,
                    'match_threshold': min_score,
                    'match_count': top_k,
                    'session_id': session_id,
                    'user_id': user_id
                })
                
            except Exception as e:
                logger.warning(f"Vector search failed: {e}, falling back to basic search")
                return await cls._fallback_text_search(query, session_id, user_id, top_k)
            
            if not result or 'error' in result:
                logger.error(f"Vector search failed: {result.get('error')}")
                # Fallback to text search
                return await cls._fallback_text_search(query, session_id, user_id, top_k)
            
            # Format results
            results = []
            for doc in result.get('data', [])[:top_k]:
                results.append({
                    'id': doc.get('id'),
                    'content': doc.get('content'),
                    'score': doc.get('similarity', 0.0),
                    'metadata': {
                        'source_type': doc.get('source_type'),
                        'source_reference': doc.get('source_reference'),
                        'chunk_index': doc.get('chunk_index'),
                        'total_chunks': doc.get('total_chunks')
                    }
                })
            
            return {
                'success': True,
                'query': query,
                'results': results,
                'count': len(results)
            }
            
        except Exception as e:
            logger.error(f"Knowledge base query failed: {str(e)}", exc_info=True)
            # Try fallback search
            return await cls._fallback_text_search(query, session_id, user_id, top_k)
    
    @classmethod
    async def _fallback_text_search(
        cls,
        query: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 5
    ) -> Dict[str, Any]:
        """Fallback text search when vector search fails"""
        try:
            # Use db_select directly with filters
            filters = {}
            if session_id:
                filters['session_id'] = session_id
            if user_id:
                filters['user_id'] = user_id
                
            # Execute query using db_select
            result = await db_select(
                'documents',  # Correct table name from the database
                columns='*',
                filters=filters,
                limit=limit,
                order_by={'created_at': 'desc'}
            )
            
            # Format results
            results = []
            for doc in result.get('data', [])[:limit]:
                results.append({
                    'id': doc.get('id'),
                    'content': doc.get('content'),
                    'score': 0.5,  # Default score for text search
                    'metadata': {
                        'source_type': doc.get('source_type'),
                        'source_reference': doc.get('source_reference'),
                        'chunk_index': doc.get('chunk_index'),
                        'total_chunks': doc.get('total_chunks')
                    }
                })
            
            return {
                'success': True,
                'query': query,
                'results': results,
                'count': len(results),
                'fallback': True
            }
            
        except Exception as e:
            logger.error(f"Fallback text search failed: {str(e)}")
            return {
                'success': False,
                'error': f"Search failed: {str(e)}",
                'results': [],
                'count': 0
            }
    
    @classmethod
    async def get_knowledge_stats(
        cls,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get statistics about the knowledge base
        
        Args:
            session_id: Optional session ID to filter by
            user_id: Optional user ID to filter by
            
        Returns:
            dict: Statistics including document counts, sessions, and sources
        """
        try:
            # Get stats from database
            result = await db_rpc('get_knowledge_stats', {
                'p_session_id': session_id,
                'p_user_id': user_id
            })
            
            if 'error' in result:
                logger.error(f"Failed to get knowledge stats: {result.get('error')}")
                raise Exception(result.get('error', 'Failed to retrieve knowledge stats'))
                
            return {
                'success': True,
                'stats': result.get('data', {})
            }
            
        except Exception as e:
            logger.error(f"Failed to get knowledge stats: {str(e)}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }
    
    @classmethod
    async def get_session_data(
        cls,
        session_id: str,
        user_id: Optional[str] = None,
        include_documents: bool = True
    ) -> Dict[str, Any]:
        """
        Get all data for a specific session
        
        Args:
            session_id: Session ID to retrieve
            user_id: Optional user ID for access control
            include_documents: Whether to include document content
            
        Returns:
            dict: Session data including metadata and documents
        """
        try:
            if not session_id:
                raise ValueError("Session ID is required")
                
            # Get session metadata
            result = await db_rpc('get_session_data', {
                'p_session_id': session_id,
                'p_user_id': user_id,
                'p_include_documents': include_documents
            })
            
            if 'error' in result:
                logger.error(f"Failed to get session data: {result.get('error')}")
                return None
                
            return result.get('data', {})
            
        except Exception as e:
            logger.error(f"Failed to get session data: {str(e)}", exc_info=True)
            return None
    
    @classmethod
    async def delete_documents(
        cls,
        criteria: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Delete documents based on criteria
        
        Args:
            criteria: Dictionary with one or more of:
                - session_id: Delete all documents in a session
                - user_id: Delete all documents for a user
                - document_ids: List of specific document IDs to delete
                - filter_metadata: Dictionary of metadata filters
                
        Returns:
            dict: Result with count of deleted documents
        """
        try:
            # Call database function to handle deletion
            result = await db_rpc('delete_documents', {
                'p_session_id': criteria.get('session_id'),
                'p_user_id': criteria.get('user_id'),
                'p_document_ids': criteria.get('document_ids', []),
                'p_filter_metadata': json.dumps(criteria.get('filter_metadata', {}))
            })
            
            if 'error' in result:
                logger.error(f"Document deletion failed: {result.get('error')}")
                raise Exception(result.get('error', 'Document deletion failed'))
                
            return {
                'success': True,
                'deleted_count': result.get('data', {}).get('deleted_count', 0),
                'criteria': criteria
            }
            
        except Exception as e:
            logger.error(f"Document deletion failed: {str(e)}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'deleted_count': 0
            }
    
    @classmethod
    async def _log_knowledge_activity(
        cls,
        activity_type: str,
        data: Dict[str, Any]
    ) -> None:
        """
        Log knowledge base activity
        
        Args:
            activity_type: Type of activity (e.g., 'document_uploaded', 'document_deleted')
            data: Activity data to log
        """
        try:
            log_entry = {
                'id': str(uuid4()),
                'activity_type': activity_type,
                'data': data,
                'created_at': datetime.utcnow().isoformat()
            }
            
            await db_insert('knowledge_activity_logs', log_entry)
            
        except Exception as e:
            logger.error(f"Failed to log activity: {str(e)}", exc_info=True)
    
    @classmethod
    async def _process_url_upload(cls, upload_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process URL upload"""
        url = upload_data.get('url')
        if not url:
            raise ValueError("No URL provided")
            
        if not is_valid_url(url):
            raise ValueError(f"Invalid URL: {url}")
            
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=30.0)
                response.raise_for_status()
                html_content = response.text
                
            # Extract text from HTML
            text_content = extract_text_from_html(html_content)
            
            metadata = upload_data.get('metadata', {})
            metadata.update({
                'source_url': url,
                'content_type': response.headers.get('content-type', '')
            })
            
            return await cls._process_content(
                content=text_content,
                source_type='url',
                source_reference=url,
                metadata=metadata,
                session_id=upload_data.get('session_id'),
                user_id=upload_data.get('user_id')
            )
            
        except Exception as e:
            logger.error(f"Failed to process URL {url}: {str(e)}")
            raise ValueError(f"Failed to process URL: {str(e)}")
    
    @classmethod
    async def _process_file_upload(cls, upload_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process file upload"""
        file_data = upload_data.get('file')
        if not file_data or 'content' not in file_data:
            raise ValueError("No file content provided")
            
        filename = file_data.get('filename', 'unknown')
        file_content = file_data['content']
        
        if not isinstance(file_content, bytes):
            if isinstance(file_content, str):
                file_content = file_content.encode('utf-8')
            else:
                raise ValueError("Invalid file content type")
        
        # Process the file based on its extension
        try:
            # Use DocumentProcessor to handle different file types
            processor = DocumentProcessor()
            file_type = processor.detect_file_type(file_content, filename)
            text_content = processor.extract_text(file_content, file_type)
            
            # Create a single document with the extracted content
            documents = [
                {
                    'content': text_content,
                    'metadata': {
                        **upload_data.get('metadata', {}),
                        'file_type': file_type,
                        'file_name': filename,
                        'file_size': len(file_content)
                    }
                }
            ]
            
            if not documents:
                raise ValueError("No content could be extracted from the file")
                
            # Process each document (for files that might have multiple pages/sections)
            results = []
            for doc in documents:
                result = await cls._process_content(
                    content=doc['content'],
                    source_type='file',
                    source_reference=filename,
                    metadata=doc.get('metadata', {}),
                    session_id=upload_data.get('session_id'),
                    user_id=upload_data.get('user_id')
                )
                results.append(result)
                
            # Combine results
            if len(results) == 1:
                return results[0]
                
            # For multiple documents, combine the results
            return {
                'success': all(r.get('success', False) for r in results),
                'document_ids': [doc_id for r in results for doc_id in r.get('document_ids', [])],
                'chunks_created': sum(r.get('chunks_created', 0) for r in results),
                'total_chunks': sum(r.get('total_chunks', 0) for r in results),
                'source_type': 'file',
                'source_reference': filename,
                'processing_time_ms': sum(r.get('processing_time_ms', 0) for r in results),
                'metadata': {k: v for r in results for k, v in r.get('metadata', {}).items()}
            }
            
        except Exception as e:
            logger.error(f"Failed to process file {filename}: {str(e)}", exc_info=True)
            raise ValueError(f"Failed to process file: {str(e)}")
    
    @classmethod
    async def _process_content(
        cls,
        content: str,
        source_type: str,
        source_reference: str,
        metadata: Dict[str, Any],
        session_id: str,
        user_id: str
    ) -> Dict[str, Any]:
        """Process content into chunks and store in the database"""
        start_time = time.time()
        
        try:
            # Split content into chunks
            chunks = chunk_text(
                content,
                chunk_size=cls.CHUNK_SIZE,
                chunk_overlap=cls.CHUNK_OVERLAP
            )
            
            if not chunks:
                raise ValueError("No content chunks could be created")
                
            logger.info(f"Split content into {len(chunks)} chunks")
            
            # Process each chunk
            document_ids = []
            
            for i, chunk in enumerate(chunks):
                try:
                    # Generate embedding for the chunk
                    embedding = await cls.get_embedding(chunk)
                    
                    # Prepare document data
                    doc_data = {
                        'content': chunk,
                        'embedding': embedding,
                        'metadata': {
                            **metadata,
                            'chunk_index': i,
                            'total_chunks': len(chunks),
                            'source_type': source_type,
                            'source_reference': source_reference,
                            'chunk_size': len(chunk),
                            'original_length': len(content)
                        },
                        'session_id': session_id,
                        'user_id': user_id,
                        'created_at': datetime.utcnow().isoformat()
                    }
                    
                    # Store in database
                    result = await db_insert('documents', doc_data)
                    if result and 'id' in result:
                        document_ids.append(result['id'])
                    
                except Exception as chunk_error:
                    logger.error(f"Failed to process chunk {i + 1}: {str(chunk_error)}")
                    # Continue with other chunks
            
            processing_time = int((time.time() - start_time) * 1000)
            
            # Log the successful upload
            await cls._log_activity(
                activity_type='document_upload',
                details={
                    'source_type': source_type,
                    'source_reference': source_reference,
                    'chunks_created': len(document_ids),
                    'total_chunks': len(chunks),
                    'processing_time_ms': processing_time,
                    'session_id': session_id,
                    'user_id': user_id,
                    'content_length': len(content),
                    'metadata': metadata
                }
            )
            
            logger.info(f"Successfully stored {len(document_ids)} chunks in {processing_time}ms")
            
            return {
                'success': True,
                'document_ids': document_ids,
                'chunks_created': len(document_ids),
                'total_chunks': len(chunks),
                'source_type': source_type,
                'source_reference': source_reference,
                'processing_time_ms': processing_time,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Content processing failed: {str(e)}", exc_info=True)
            raise
    
    @classmethod
    async def get_knowledge_stats(cls, user_id: Optional[str] = None, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get knowledge base statistics and metrics
        
        Args:
            user_id: Optional user ID to filter stats by user
            session_id: Optional session ID to filter stats by session
            
        Returns:
            Dictionary containing statistics about the knowledge base
        """
        start_time = time.time()
        
        try:
            # Get Supabase client
            from ..core.database import get_supabase_client
            supabase = get_supabase_client()
            
            stats = {
                'documents': 0,
                'chunks': 0,
                'sessions': 0,
                'users': 0,
                'by_source_type': {}
            }
            
            # Build base query
            query = supabase.table('documents').select('*', count='exact')
            
            # Add filters if provided
            if user_id:
                query = query.eq('user_id', user_id)
            if session_id:
                query = query.eq('session_id', session_id)
            
            # Get total document count
            result = query.execute()
            stats['chunks'] = result.count or 0
            
            # Get unique document count (group by content hash or other unique identifier)
            # This is a simplified approach - you might need to adjust based on your schema
            stats['documents'] = stats['chunks']  # Default to chunks count if no better way
            
            # Get unique sessions count
            sessions_query = supabase.table('documents').select('session_id', count='exact')
            if user_id:
                sessions_query = sessions_query.eq('user_id', user_id)
            sessions_result = sessions_query.execute()
            stats['sessions'] = sessions_result.count or 0
            
            # Get unique users count
            users_query = supabase.table('documents').select('user_id', count='exact')
            if session_id:
                users_query = users_query.eq('session_id', session_id)
            users_result = users_query.execute()
            stats['users'] = users_result.count or 0
            
            # Get stats by source type
            source_types_query = supabase.table('documents').select('source_type', count='exact')
            if user_id:
                source_types_query = source_types_query.eq('user_id', user_id)
            if session_id:
                source_types_query = source_types_query.eq('session_id', session_id)
                
            source_types_result = source_types_query.execute()
            
            if source_types_result.data:
                for item in source_types_result.data:
                    source_type = item.get('source_type', 'unknown')
                    stats['by_source_type'][source_type] = item.get('count', 0)
            
            # Add processing time
            stats['processing_time_ms'] = int((time.time() - start_time) * 1000)
            stats['generated_at'] = datetime.utcnow().isoformat()
            
            # Log the stats request
            await cls._log_activity(
                activity_type='stats_retrieval',
                details={
                    'user_id': user_id,
                    'session_id': session_id,
                    'processing_time_ms': stats['processing_time_ms']
                }
            )
            
            return stats
            
        except Exception as error:
            print(f"❌ [KnowledgeService] Stats retrieval failed: {error}")
            raise error
    
    @classmethod
    async def get_session_data(cls, session_id: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get all documents and metadata for a specific session
        
        Args:
            session_id: The session ID to retrieve data for
            user_id: Optional user ID for access control
            
        Returns:
            Dictionary with session data and documents
        """
        start_time = time.time()
        
        try:
            if not session_id:
                raise ValueError("Session ID is required")
                
            logger.info(f"Retrieving data for session: {session_id}")
            
            # Get Supabase client
            from ..core.database import get_supabase_client
            supabase = get_supabase_client()
            
            # Build the query
            query = supabase.table('documents').select('*').eq('session_id', session_id)
            
            if user_id:
                query = query.eq('user_id', user_id)
                
            # Order by creation date
            query = query.order('created_at', desc=True)
            
            # Execute the query
            result = query.execute()
            
            if result.error:
                raise Exception(result.error)
                
            documents = result.data or []
            
            # Process documents to parse metadata if needed
            processed_docs = []
            for doc in documents:
                metadata = doc.get('metadata', {})
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except json.JSONDecodeError:
                        metadata = {}
                
                processed_docs.append({
                    'id': doc.get('id'),
                    'content': doc.get('content'),
                    'source_type': doc.get('source_type'),
                    'source_reference': doc.get('source_reference'),
                    'metadata': metadata,
                    'created_at': doc.get('created_at'),
                    'chunk_index': doc.get('chunk_index'),
                    'total_chunks': doc.get('total_chunks')
                })
            
            # Get session metadata if available
            session_metadata = {}
            if documents:
                # Try to extract common metadata from the first document
                first_doc = documents[0]
                session_metadata = {
                    'session_id': session_id,
                    'user_id': first_doc.get('user_id'),
                    'document_count': len(documents),
                    'source_types': list(set(doc.get('source_type') for doc in documents if doc.get('source_type'))),
                    'created_at': first_doc.get('created_at'),
                    'updated_at': max(doc.get('updated_at', '') for doc in documents) if documents else None
                }
            
            processing_time = int((time.time() - start_time) * 1000)
            
            # Log the session data retrieval
            await cls._log_activity(
                activity_type='session_data_retrieval',
                details={
                    'session_id': session_id,
                    'user_id': user_id,
                    'document_count': len(processed_docs),
                    'processing_time_ms': processing_time
                }
            )
            
            return {
                'success': True,
                'session': session_metadata,
                'documents': processed_docs,
                'count': len(processed_docs),
                'processing_time_ms': processing_time
            }
            
        except Exception as error:
            logger.error(f"Failed to retrieve session data: {str(error)}", exc_info=True)
            raise
            
    @classmethod
    async def delete_documents(cls, delete_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Delete documents from the knowledge base
        
        Args:
            delete_data: Dictionary containing criteria for deletion
                - session_id: Delete all documents for a session
                - user_id: Delete all documents for a user
                - document_ids: List of specific document IDs to delete
                - filter_metadata: Metadata filters for documents to delete
                
        Returns:
            Dictionary with deletion results
        """
        start_time = time.time()
        
        try:
            session_id = delete_data.get('session_id')
            user_id = delete_data.get('user_id')
            document_ids = delete_data.get('document_ids', [])
            filter_metadata = delete_data.get('filter_metadata', {})
            
            if not any([session_id, user_id, document_ids, filter_metadata]):
                raise ValueError("No deletion criteria provided. Must specify session_id, user_id, document_ids, or filter_metadata")
            
            # Track what's being deleted for logging
            deleted_info = {
                'session_id': session_id,
                'user_id': user_id,
                'document_ids': document_ids,
                'filter_metadata': filter_metadata,
                'deleted_count': 0
            }
            
            # Get Supabase client
            from ..core.database import get_supabase_client
            supabase = get_supabase_client()
            
            # Build the delete query
            query = supabase.table('documents').delete()
            
            # Add conditions based on provided criteria
            conditions = []
            
            if session_id:
                query = query.eq('session_id', session_id)
                conditions.append(f'session_id={session_id}')
                
            if user_id:
                query = query.eq('user_id', user_id)
                conditions.append(f'user_id={user_id}')
                
            if document_ids:
                query = query.in_('id', document_ids)
                conditions.append(f'document_ids={len(document_ids)}')
                
            # Add metadata filters
            if filter_metadata and isinstance(filter_metadata, dict):
                for key, value in filter_metadata.items():
                    query = query.contains(f'metadata->{key}', f'"{value}"')
                    conditions.append(f'{key}={value}')
            
            # Execute the delete
            result = query.execute()
            
            # Get the count of deleted rows if available
            if hasattr(result, 'count') and result.count is not None:
                deleted_count = result.count
            else:
                # If count isn't available, we'll have to query first to get the count
                count_query = supabase.table('documents').select('*', count='exact')
                
                if session_id:
                    count_query = count_query.eq('session_id', session_id)
                if user_id:
                    count_query = count_query.eq('user_id', user_id)
                if document_ids:
                    count_query = count_query.in_('id', document_ids)
                    
                count_result = count_query.execute()
                deleted_count = count_result.count or 0
            
            deleted_info['deleted_count'] = deleted_count
            
            processing_time = int((time.time() - start_time) * 1000)
            
            # Log the deletion
            await cls._log_activity(
                activity_type='documents_deleted',
                details={
                    **deleted_info,
                    'processing_time_ms': processing_time
                }
            )
            
            logger.info(f"Deleted {deleted_count} documents in {processing_time}ms")
            
            return {
                'success': True,
                'deleted_count': deleted_count,
                'processing_time_ms': processing_time,
                'criteria': conditions
            }
            
        except Exception as error:
            logger.error(f"Document deletion failed: {str(error)}", exc_info=True)
            raise
    
    @classmethod
    async def log_knowledge_activity(cls, event_type: str, details: Dict[str, Any]):
        """Log knowledge base activity"""
        try:
            log_data = {
                'event_type': event_type,
                'details': {
                    **details,
                    'timestamp': datetime.now().isoformat(),
                    'service': 'KnowledgeService'
                }
            }
            
            await db_insert('logs', log_data)
            
        except Exception as error:
            print(f"❌ [KnowledgeService] Logging failed: {error}")
            # Don't throw - logging failures shouldn't break the main flow
            
    @classmethod
    async def _log_activity(cls, activity_type: str, details: Dict[str, Any]) -> None:
        """
        Internal method for logging knowledge base activity
        
        Args:
            activity_type: Type of activity (e.g., 'document_upload', 'document_deleted')
            details: Activity data to log
        """
        try:
            log_data = {
                'activity_type': activity_type,
                'details': details,
                'timestamp': datetime.now().isoformat(),
                'service': 'KnowledgeService'
            }
            
            await db_insert('logs', log_data)
            
        except Exception as e:
            logger.error(f"Failed to log activity: {str(e)}", exc_info=True)
            # Don't throw - logging failures shouldn't break the main flow
    
    @classmethod
    async def knowledge_search_tool(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tool function for agents to search knowledge base
        
        Args:
            params: Search parameters with query, limit, similarity_threshold, session_id
            
        Returns:
            Top relevant chunks for agent use
        """
        try:
            query_result = await cls.query_knowledge_base({
                'query': params['query'],
                'session_id': params.get('session_id', f"tool_{int(time.time() * 1000)}"),
                'user_id': 'agent_system',
                'limit': params.get('limit', 3),
                'similarity_threshold': params.get('similarity_threshold', 0.7)
            })
            
            # Format results for agent consumption
            formatted_results = []
            for index, result in enumerate(query_result['results']):
                formatted_results.append({
                    'rank': index + 1,
                    'content': result['content'],
                    'source': result['source_reference'],
                    'similarity': result['similarity'],
                    'metadata': result['metadata']
                })
            
            return {
                'query': params['query'],
                'results': formatted_results,
                'result_count': len(formatted_results),
                'search_successful': True,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as error:
            print(f"❌ [KnowledgeService] Tool search failed: {error}")
            
            return {
                'query': params['query'],
                'results': [],
                'result_count': 0,
                'search_successful': False,
                'error': str(error),
                'timestamp': datetime.now().isoformat()
            }