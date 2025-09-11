"""
KnowledgeService - Comprehensive Knowledge Base Management

Provides advanced knowledge base functionality including:
- Document ingestion from multiple sources (text, URL, PDF, files)
- Intelligent text chunking using Langchain
- Vector embedding generation using OpenAI
- Similarity search with configurable thresholds
- Session tracking and analytics
- Comprehensive error handling and logging
"""

import os
import time
import asyncio
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import aiohttp
import aiofiles
from pathlib import Path

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

# PDF processing
import PyPDF2
from io import BytesIO

# HTML processing
from bs4 import BeautifulSoup

# Core imports
from ..core.database import supabase, db_insert, db_update, db_select, db_delete
from openai import AsyncOpenAI

class KnowledgeService:
    """Knowledge base management service with advanced document processing and search capabilities"""
    
    # Initialize OpenAI client
    openai_client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=['\n\n', '\n', '. ', ' ', '']
    )
    
    @classmethod
    async def upload_document(cls, upload_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Upload and process document from various sources
        
        Args:
            upload_data: Upload configuration containing type, content/url/file, session_id, user_id, metadata
            
        Returns:
            Upload result with document IDs and statistics
        """
        start_time = time.time()
        
        try:
            print(f"🔄 [KnowledgeService] Starting document upload: {upload_data['type']}")
            
            # Extract text content based on source type
            text_content = ''
            source_reference = ''
            extracted_metadata = upload_data.get('metadata', {})
            
            upload_type = upload_data['type']
            
            if upload_type == 'text':
                text_content = upload_data['content']
                source_reference = 'direct_input'
                
            elif upload_type == 'url':
                url_result = await cls.extract_from_url(upload_data['url'])
                text_content = url_result['content']
                source_reference = upload_data['url']
                extracted_metadata.update(url_result['metadata'])
                
            elif upload_type == 'file':
                file_result = await cls.extract_from_file(upload_data['file'])
                text_content = file_result['content']
                source_reference = upload_data['file'].get('filename', 'unknown_file')
                extracted_metadata.update(file_result['metadata'])
                
            else:
                raise ValueError(f"Unsupported upload type: {upload_type}")
            
            if not text_content or len(text_content.strip()) == 0:
                raise ValueError('No text content could be extracted from the source')
            
            print(f"📝 [KnowledgeService] Extracted {len(text_content)} characters")
            
            # Chunk the text using Langchain
            chunks = await asyncio.to_thread(cls.text_splitter.split_text, text_content)
            print(f"🔪 [KnowledgeService] Split into {len(chunks)} chunks")
            
            # Generate embeddings and store chunks
            document_ids = []
            
            for i, chunk in enumerate(chunks):
                try:
                    # Generate embedding for this chunk
                    embedding = await cls.generate_embedding(chunk)
                    
                    # Store chunk in database
                    chunk_data = {
                        'content': chunk,
                        'metadata': {
                            **extracted_metadata,
                            'chunk_size': len(chunk),
                            'original_length': len(text_content),
                            'extraction_method': upload_type
                        },
                        'embedding': embedding,
                        'source_type': upload_type,
                        'source_reference': source_reference,
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'session_id': upload_data['session_id'],
                        'user_id': upload_data['user_id']
                    }
                    
                    result = await db_insert('documents', chunk_data)
                    if result:
                        document_ids.append(result[0]['id'])
                    
                    print(f"✅ [KnowledgeService] Stored chunk {i + 1}/{len(chunks)}")
                    
                except Exception as chunk_error:
                    print(f"❌ [KnowledgeService] Failed to process chunk {i + 1}: {chunk_error}")
                    # Continue with other chunks rather than failing completely
            
            processing_time = int((time.time() - start_time) * 1000)
            
            # Log the upload
            await cls.log_knowledge_activity('document_upload', {
                'source_type': upload_type,
                'source_reference': source_reference,
                'chunks_created': len(document_ids),
                'total_chunks': len(chunks),
                'processing_time_ms': processing_time,
                'session_id': upload_data['session_id'],
                'user_id': upload_data['user_id'],
                'content_length': len(text_content),
                'metadata': extracted_metadata
            })
            
            print(f"🎉 [KnowledgeService] Upload completed: {len(document_ids)} chunks stored")
            
            return {
                'success': True,
                'document_ids': document_ids,
                'chunks_created': len(document_ids),
                'total_chunks': len(chunks),
                'source_type': upload_type,
                'source_reference': source_reference,
                'processing_time_ms': processing_time,
                'metadata': extracted_metadata
            }
            
        except Exception as error:
            processing_time = int((time.time() - start_time) * 1000)
            
            print(f"❌ [KnowledgeService] Upload failed: {error}")
            
            # Log the error
            await cls.log_knowledge_activity('document_upload_error', {
                'source_type': upload_data.get('type'),
                'error': str(error),
                'processing_time_ms': processing_time,
                'session_id': upload_data.get('session_id'),
                'user_id': upload_data.get('user_id')
            })
            
            raise error
    
    @classmethod
    async def query_knowledge_base(cls, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Query the knowledge base using similarity search
        
        Args:
            query_data: Query configuration with query, session_id, user_id, limit, similarity_threshold
            
        Returns:
            Query results with relevant chunks and metadata
        """
        start_time = time.time()
        
        try:
            print(f"🔍 [KnowledgeService] Querying knowledge base: \"{query_data['query']}\"")
            
            # Generate embedding for the query
            query_embedding = await cls.generate_embedding(query_data['query'])
            
            # Perform similarity search using Supabase RPC
            result = supabase.rpc('search_documents', {
                'query_embedding': query_embedding,
                'similarity_threshold': query_data.get('similarity_threshold', 0.7),
                'match_count': query_data.get('limit', 10)
            }).execute()
            
            if result.error:
                raise Exception(result.error)
            
            results = result.data or []
            processing_time = int((time.time() - start_time) * 1000)
            
            # Log the query in knowledge_sessions
            session_data = {
                'session_id': query_data['session_id'],
                'user_id': query_data['user_id'],
                'query': query_data['query'],
                'results': results,
                'result_count': len(results),
                'similarity_threshold': query_data.get('similarity_threshold', 0.7),
                'processing_time_ms': processing_time
            }
            
            await db_insert('knowledge_sessions', session_data)
            
            await cls.log_knowledge_activity('knowledge_query', {
                'query': query_data['query'],
                'result_count': len(results),
                'similarity_threshold': query_data.get('similarity_threshold', 0.7),
                'processing_time_ms': processing_time,
                'session_id': query_data['session_id'],
                'user_id': query_data['user_id']
            })
            
            print(f"✅ [KnowledgeService] Query completed: {len(results)} results found")
            
            return {
                'success': True,
                'query': query_data['query'],
                'results': results,
                'result_count': len(results),
                'similarity_threshold': query_data.get('similarity_threshold', 0.7),
                'processing_time_ms': processing_time,
                'session_id': query_data['session_id']
            }
            
        except Exception as error:
            processing_time = int((time.time() - start_time) * 1000)
            
            print(f"❌ [KnowledgeService] Query failed: {error}")
            
            await cls.log_knowledge_activity('knowledge_query_error', {
                'query': query_data['query'],
                'error': str(error),
                'processing_time_ms': processing_time,
                'session_id': query_data['session_id'],
                'user_id': query_data['user_id']
            })
            
            raise error
    
    @classmethod
    async def extract_from_url(cls, url: str) -> Dict[str, Any]:
        """
        Extract text content from URL
        
        Args:
            url: URL to fetch and extract content from
            
        Returns:
            Extracted content and metadata
        """
        try:
            print(f"🌐 [KnowledgeService] Fetching content from URL: {url}")
            
            async with aiohttp.ClientSession() as session:
                headers = {
                    'User-Agent': 'AgentFlow Knowledge Base Bot 1.0',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
                }
                
                async with session.get(url, headers=headers, timeout=30) as response:
                    response_text = await response.text()
                    content_type = response.headers.get('content-type', '')
                    
                    metadata = {
                        'url': url,
                        'content_type': content_type,
                        'status_code': response.status,
                        'fetched_at': datetime.now().isoformat()
                    }
                    
                    content = ''
                    
                    # Enhanced HTML text extraction using BeautifulSoup
                    if 'text/html' in content_type:
                        soup = BeautifulSoup(response_text, 'html.parser')
                        
                        # Remove script and style elements
                        for script in soup(['script', 'style', 'nav', 'footer', 'header']):
                            script.decompose()
                        
                        # Extract main content areas
                        content_selectors = [
                            'main', 'article', '.content', '.main-content',
                            '#content', '.post-content', '.entry-content'
                        ]
                        
                        main_content = None
                        for selector in content_selectors:
                            main_content = soup.select_one(selector)
                            if main_content:
                                break
                        
                        # Use main content or fallback to body
                        content_element = main_content or soup.body
                        content = content_element.get_text() if content_element else ''
                        
                        # Clean up whitespace
                        content = ' '.join(content.split())
                        
                        # Extract title
                        title_element = soup.find('title')
                        if title_element:
                            metadata['title'] = title_element.get_text().strip()
                        
                        # Extract meta description
                        description_element = soup.find('meta', attrs={'name': 'description'})
                        if description_element:
                            metadata['description'] = description_element.get('content')
                        
                        # Detect content type (Notion, Confluence, etc.)
                        if 'notion.so' in url or 'notion.site' in url:
                            metadata['source_platform'] = 'Notion'
                        elif 'confluence' in url:
                            metadata['source_platform'] = 'Confluence'
                        elif 'wikipedia.org' in url:
                            metadata['source_platform'] = 'Wikipedia'
                        elif 'github.com' in url:
                            metadata['source_platform'] = 'GitHub'
                    else:
                        content = response_text
                    
                    # Validate content length
                    if not content or len(content) < 50:
                        raise ValueError('Insufficient content extracted from URL')
                    
                    return {'content': content, 'metadata': metadata}
                    
        except Exception as error:
            print(f"❌ [KnowledgeService] URL extraction failed: {error}")
            raise Exception(f"Failed to extract content from URL: {error}")
    
    @classmethod
    async def extract_from_file(cls, file_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract text content from uploaded file
        
        Args:
            file_data: File data containing filename, mimetype, content
            
        Returns:
            Extracted content and metadata
        """
        try:
            filename = file_data.get('filename', 'unknown')
            print(f"📄 [KnowledgeService] Processing file: {filename}")
            
            content = ''
            metadata = {
                'filename': filename,
                'mimetype': file_data.get('mimetype'),
                'size': len(file_data.get('content', b'')),
                'processed_at': datetime.now().isoformat()
            }
            
            file_content = file_data.get('content', b'')
            mimetype = file_data.get('mimetype', '')
            
            # Handle different file types
            if mimetype == 'application/pdf':
                # Process PDF
                pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
                content_parts = []
                
                for page in pdf_reader.pages:
                    content_parts.append(page.extract_text())
                
                content = '\n'.join(content_parts)
                metadata['pages'] = len(pdf_reader.pages)
                
            elif mimetype and mimetype.startswith('text/'):
                content = file_content.decode('utf-8')
                
            else:
                raise ValueError(f"Unsupported file type: {mimetype}")
            
            return {'content': content, 'metadata': metadata}
            
        except Exception as error:
            print(f"❌ [KnowledgeService] File extraction failed: {error}")
            raise Exception(f"Failed to extract content from file: {error}")
    
    @classmethod
    async def generate_embedding(cls, text: str) -> List[float]:
        """
        Generate embedding using OpenAI
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            1536-dimensional embedding vector
        """
        try:
            # Limit text to avoid token limits
            limited_text = text[:8000]
            
            response = await cls.openai_client.embeddings.create(
                model='text-embedding-ada-002',
                input=limited_text
            )
            
            return response.data[0].embedding
            
        except Exception as error:
            print(f"❌ [KnowledgeService] Embedding generation failed: {error}")
            raise Exception(f"Failed to generate embedding: {error}")
    
    @classmethod
    async def get_knowledge_stats(cls) -> Dict[str, Any]:
        """Get knowledge base statistics and metrics"""
        try:
            # Get statistics using RPC
            stats_result = supabase.rpc('get_knowledge_stats').execute()
            if stats_result.error:
                raise Exception(stats_result.error)
            
            # Get recent activity
            activity_result = supabase.from_('recent_knowledge_activity').select('*').limit(10).execute()
            
            return {
                'statistics': stats_result.data[0] if stats_result.data else {},
                'recent_activity': activity_result.data or [],
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as error:
            print(f"❌ [KnowledgeService] Stats retrieval failed: {error}")
            raise error
    
    @classmethod
    async def delete_documents(cls, delete_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Delete documents by session or user
        
        Args:
            delete_data: Deletion criteria with session_id or user_id
            
        Returns:
            Deletion result
        """
        try:
            if delete_data.get('session_id'):
                result = supabase.from_('documents').delete().eq('session_id', delete_data['session_id']).execute()
            elif delete_data.get('user_id'):
                result = supabase.from_('documents').delete().eq('user_id', delete_data['user_id']).execute()
            else:
                raise ValueError('Either session_id or user_id must be provided')
            
            if result.error:
                raise Exception(result.error)
            
            deleted_count = len(result.data) if result.data else 0
            
            await cls.log_knowledge_activity('documents_deleted', {
                'deleted_count': deleted_count,
                'session_id': delete_data.get('session_id'),
                'user_id': delete_data.get('user_id')
            })
            
            return {
                'success': True,
                'deleted_count': deleted_count
            }
            
        except Exception as error:
            print(f"❌ [KnowledgeService] Document deletion failed: {error}")
            raise error
    
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
