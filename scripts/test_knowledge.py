#!/usr/bin/env python3

"""
Knowledge Base Complete Test Script

Tests the complete Knowledge Base functionality including:
1. Document upload from various sources (text, URL, file)
2. Text chunking and embedding generation
3. Vector similarity search
4. Agent integration with knowledge search tool
5. Session tracking and analytics
6. Error handling and edge cases
"""

import asyncio
import aiohttp
import json
import sys
import base64
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

BASE_URL = 'http://localhost:8000'

# Test configuration
TEST_SESSION_ID = f"knowledge_test_{int(datetime.now().timestamp() * 1000)}"
TEST_USER_ID = 'knowledge_test_user'

async def api_request(session: aiohttp.ClientSession, endpoint: str, method: str = 'GET', data: dict = None, files: dict = None):
    """Helper function to make API requests"""
    url = f"{BASE_URL}{endpoint}"
    
    # Ensure required fields are present for knowledge endpoints
    if endpoint.startswith('/api/knowledge'):
        if method == 'POST' and 'type' not in (data or {}) and 'type' not in (files or {}):
            data = data or {}
            data['type'] = 'text'  # Default to text type if not specified
            
    try:
        if method == 'GET':
            async with session.get(url) as response:
                result = await response.json()
                if response.status >= 400:
                    raise Exception(f"API Error: {result.get('detail', 'Unknown error')}")
                return result
                
        elif files:
            # Handle file upload
            form_data = aiohttp.FormData()
            for key, value in (data or {}).items():
                form_data.add_field(key, str(value))
            for key, file_data in files.items():
                form_data.add_field(key, file_data['content'], filename=file_data['filename'])
            
            async with session.post(url, data=form_data) as response:
                result = await response.json()
                if response.status >= 400:
                    raise Exception(f"API Error: {result.get('detail', 'Unknown error')}")
                return result
                
        else:
            # For JSON requests, ensure we have the right content type
            headers = {'Content-Type': 'application/json'}
            async with session.request(method, url, json=data, headers=headers) as response:
                result = await response.json()
                if response.status >= 400:
                    raise Exception(f"API Error: {result.get('detail', 'Unknown error')}")
                return result
                
    except Exception as error:
        print(f"Failed to {method} {endpoint}: {error}")
        raise error

async def test_text_upload(session: aiohttp.ClientSession):
    """Test direct text upload"""
    print("📝 Testing direct text upload...")
    
    test_content = """
    This is a comprehensive test document for the knowledge base system.
    It contains information about artificial intelligence, machine learning,
    and natural language processing. The system should be able to chunk
    this text appropriately and generate embeddings for similarity search.
    
    Key topics covered:
    - AI and ML fundamentals
    - NLP techniques
    - Vector embeddings
    - Similarity search algorithms
    """
    
    try:
        # Create form data with proper content type
        form_data = aiohttp.FormData()
        form_data.add_field('type', 'text')
        form_data.add_field('content', test_content)
        form_data.add_field('session_id', TEST_SESSION_ID)
        form_data.add_field('user_id', TEST_USER_ID)
        form_data.add_field('metadata', json.dumps({
            'title': 'Test Document',
            'category': 'AI/ML',
            'test_type': 'direct_text'
        }))
        
        # Make the request - aiohttp will set the correct Content-Type with boundary
        async with session.post(
            f"{BASE_URL}/api/knowledge/upload", 
            data=form_data
        ) as response:
            result = await response.json()
            if response.status >= 400:
                raise Exception(f"API Error: {result.get('detail', 'Unknown error')}")
            
            print(f"✅ Text upload successful:")
            print(f"   Chunks created: {result.get('chunks_created', 0)}")
            print(f"   Processing time: {result.get('processing_time_ms', 0)}ms")
            print(f"   Document IDs: {len(result.get('document_ids', []))}")
            
            return result
            
    except Exception as error:
        print(f"❌ Text upload failed: {error}")
        return None

async def test_url_upload(session: aiohttp.ClientSession):
    """Test URL content extraction and upload"""
    print("\n🌐 Testing URL upload...")
    
    # Use a reliable test URL
    test_url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
    
    try:
        # Create form data with proper content type
        form_data = aiohttp.FormData()
        form_data.add_field('type', 'url')
        form_data.add_field('url', test_url)
        form_data.add_field('session_id', TEST_SESSION_ID)
        form_data.add_field('user_id', TEST_USER_ID)
        form_data.add_field('metadata', json.dumps({
            'title': 'AI Wikipedia Page',
            'source': 'wikipedia',
            'test_type': 'url_upload'
        }))
        
        # Make the request - aiohttp will set the correct Content-Type with boundary
        async with session.post(
            f"{BASE_URL}/api/knowledge/upload", 
            data=form_data
        ) as response:
            result = await response.json()
            if response.status >= 400:
                raise Exception(f"API Error: {result.get('detail', 'Unknown error')}")
            
            print(f"✅ URL upload successful:")
            print(f"   URL: {test_url}")
            print(f"   Chunks created: {result.get('chunks_created', 0)}")
            print(f"   Processing time: {result.get('processing_time_ms', 0)}ms")
            
            return result
            
    except Exception as error:
        print(f"❌ URL upload failed: {error}")
        return None

async def test_file_upload(session: aiohttp.ClientSession):
    """Test file upload (simulate text file)"""
    print("\n📄 Testing file upload...")
    
    # Create a mock text file for testing
    test_file_content = """
    Machine Learning Test Document
    
    This document contains information about machine learning algorithms,
    including supervised learning, unsupervised learning, and reinforcement learning.
    
    Supervised Learning:
    - Classification algorithms
    - Regression algorithms
    - Decision trees and random forests
    
    Unsupervised Learning:
    - Clustering algorithms
    - Dimensionality reduction
    - Association rules
    
    Reinforcement Learning:
    - Q-learning
    - Policy gradients
    - Actor-critic methods
    """
    
    try:
        # Create a temporary file for testing
        import tempfile
        import os
        
        # Create a temporary file with .txt extension
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp_file:
            temp_file.write(test_file_content.encode('utf-8'))
            temp_file_path = temp_file.name
        
        try:
            # Read the file content as binary
            with open(temp_file_path, 'rb') as f:
                file_content = f.read()
            
            # Create form data
            form_data = aiohttp.FormData()
            form_data.add_field('type', 'file')
            form_data.add_field('session_id', TEST_SESSION_ID)
            form_data.add_field('user_id', TEST_USER_ID)
            form_data.add_field('metadata', json.dumps({
                'title': 'ML Test Document',
                'category': 'Machine Learning',
                'test_type': 'file_upload'
            }))
            
            # Add file to form data
            form_data.add_field(
                'file',
                file_content,
                filename='ml_test_document.txt',
                content_type='text/plain'
            )
            
            # Make the request
            async with session.post(
                f"{BASE_URL}/api/knowledge/upload",
                data=form_data
            ) as response:
                result = await response.json()
                
                if response.status >= 400:
                    error_detail = result.get('detail', {}) or {}
                    error_msg = error_detail.get('message', 'Unknown error') if isinstance(error_detail, dict) else str(error_detail)
                    raise Exception(f"API Error: {error_msg}")
                
                print(f"✅ File upload successful:")
                print(f"   Filename: ml_test_document.txt")
                print(f"   Chunks created: {result.get('chunks_created', 0)}")
                print(f"   Processing time: {result.get('processing_time_ms', 0)}ms")
                
                return result
                
        finally:
            # Clean up the temporary file
            try:
                os.unlink(temp_file_path)
            except:
                pass
            
    except Exception as error:
        print(f"❌ File upload failed: {error}")
        return None

async def test_knowledge_search(session: aiohttp.ClientSession):
    """Test knowledge base search functionality"""
    print("\n🔍 Testing knowledge base search...")
    
    test_queries = [
        "What is artificial intelligence?",
        "Tell me about machine learning algorithms",
        "How does natural language processing work?",
        "Explain vector embeddings"
    ]
    
    search_results = []
    
    for query in test_queries:
        try:
            # Make the search request with proper headers
            headers = {'Content-Type': 'application/json'}
            
            # First try the query endpoint (POST /api/knowledge/query)
            payload = {
                'query': query,
                'session_id': TEST_SESSION_ID,
                'user_id': TEST_USER_ID,
                'limit': 3,
                'similarity_threshold': 0.3  # Lower threshold to get more results
            }
            
            async with session.post(
                f"{BASE_URL}/api/knowledge/query",
                json=payload,
                headers=headers
            ) as response:
                result = await response.json()
                
                if response.status >= 400:
                    # If query endpoint fails, try the search endpoint (GET /api/knowledge/search)
                    params = {
                        'q': query,
                        'session_id': TEST_SESSION_ID,
                        'user_id': TEST_USER_ID,
                        'limit': 3,
                        'similarity_threshold': 0.3
                    }
                    
                    async with session.get(
                        f"{BASE_URL}/api/knowledge/search",
                        params=params,
                        headers=headers
                    ) as search_response:
                        result = await search_response.json()
                        if search_response.status >= 400:
                            raise Exception("Both query and search endpoints failed")
                
                print(f"\n📋 Query: \"{query}\"")
                
                # Handle different response formats
                if isinstance(result, dict):
                    # Handle dict response (from /query)
                    if 'results' in result:
                        results = result['results']
                        print(f"   Results found: {len(results)}")
                        
                        for i, doc in enumerate(results[:2]):  # Show top 2 results
                            content = doc.get('content', doc.get('text', ''))
                            if len(content) > 100:
                                content = content[:97] + '...'
                            print(f"   Result {i+1}: {content}")
                    else:
                        # Handle single result format
                        print(f"   Result: {result.get('content', 'No content')}")
                elif isinstance(result, list):
                    # Handle list response (from /search)
                    print(f"   Results found: {len(result)}")
                    for i, doc in enumerate(result[:2]):  # Show top 2 results
                        content = doc.get('content', doc.get('text', ''))
                        if len(content) > 100:
                            content = content[:97] + '...'
                        print(f"   Result {i+1}: {content}")
                
                success = True
                result_count = (
                    len(result) if isinstance(result, list) 
                    else len(result.get('results', [])) if isinstance(result, dict) 
                    else 1 if result else 0
                )
                
                search_results.append({
                    'query': query,
                    'result_count': result_count,
                    'success': success
                })
                
        except Exception as error:
            print(f"❌ Search failed for \"{query}\": {error}")
            search_results.append({
                'query': query,
                'result_count': 0,
                'success': False,
                'error': str(error)
            })
    
    return search_results

async def test_knowledge_stats(session: aiohttp.ClientSession):
    """Test knowledge base statistics"""
    print("\n📊 Testing knowledge base statistics...")
    
    try:
        # First try the direct stats endpoint
        headers = {'Accept': 'application/json'}
        
        # Try the direct stats endpoint first
        stats_urls = [
            f"{BASE_URL}/api/knowledge/stats",
            f"{BASE_URL}/api/knowledge/stats/{TEST_SESSION_ID}"
        ]
        
        result = None
        last_error = None
        
        for url in stats_urls:
            try:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        break
                    elif response.status >= 400:
                        error_data = await response.json()
                        last_error = error_data.get('detail', f"Status {response.status}")
            except Exception as e:
                last_error = str(e)
                continue
        
        if result is None:
            # If direct endpoints failed, try the RPC endpoint
            try:
                rpc_payload = {
                    'session_id': TEST_SESSION_ID,
                    'user_id': TEST_USER_ID
                }
                
                async with session.post(
                    f"{BASE_URL}/api/knowledge/rpc/get_stats",
                    json=rpc_payload,
                    headers=headers
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                    else:
                        error_data = await response.json()
                        last_error = error_data.get('detail', f"Status {response.status}")
            except Exception as e:
                last_error = str(e)
        
        if result is None:
            raise Exception(f"All stats endpoints failed. Last error: {last_error}")
        
        print("✅ Knowledge base statistics:")
        
        # Handle different response formats
        if isinstance(result, dict):
            # Handle standard stats format
            if 'statistics' in result:
                stats = result['statistics']
                print(f"   Total documents: {stats.get('total_documents', 0)}")
                print(f"   Total sessions: {stats.get('total_sessions', 0)}")
                print(f"   Recent activity: {len(result.get('recent_activity', []))}")
            else:
                # Handle flat response
                print(f"   Total documents: {result.get('total_documents', 0)}")
                print(f"   Total sessions: {result.get('total_sessions', result.get('session_count', 0))}")
                print(f"   Documents in session: {result.get('documents_in_session', 0)}")
        else:
            print(f"   Raw response: {result}")
        
        return result
            
    except Exception as error:
        print(f"❌ Stats retrieval failed: {error}")
        return None

async def test_session_data(session: aiohttp.ClientSession):
    """Test session data retrieval"""
    print("\n📋 Testing session data retrieval...")
    
    try:
        headers = {'Accept': 'application/json'}
        
        # Try different session data endpoints
        session_urls = [
            f"{BASE_URL}/api/knowledge/sessions/{TEST_SESSION_ID}",
            f"{BASE_URL}/api/knowledge/session/{TEST_SESSION_ID}",
            f"{BASE_URL}/api/knowledge/session?session_id={TEST_SESSION_ID}"
        ]
        
        result = None
        last_error = None
        
        for url in session_urls:
            try:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        break
                    elif response.status >= 400:
                        error_data = await response.json()
                        last_error = error_data.get('detail', f"Status {response.status}")
            except Exception as e:
                last_error = str(e)
                continue
        
        if result is None:
            # If direct endpoints failed, try the RPC endpoint
            try:
                rpc_payload = {
                    'session_id': TEST_SESSION_ID,
                    'user_id': TEST_USER_ID
                }
                
                async with session.post(
                    f"{BASE_URL}/api/knowledge/rpc/get_session_data",
                    json=rpc_payload,
                    headers=headers
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                    else:
                        error_data = await response.json()
                        last_error = error_data.get('detail', f"Status {response.status}")
            except Exception as e:
                last_error = str(e)
        
        if result is None:
            raise Exception(f"All session data endpoints failed. Last error: {last_error}")
        
        print("✅ Session data retrieved:")
        
        # Handle different response formats
        if isinstance(result, dict):
            print(f"   Session ID: {result.get('session_id', TEST_SESSION_ID)}")
            
            # Check for different possible response structures
            if 'queries' in result:
                print(f"   Total queries: {len(result.get('queries', []))}")
            elif 'query_count' in result:
                print(f"   Total queries: {result.get('query_count', 0)}")
                
            if 'documents' in result:
                print(f"   Documents uploaded: {len(result.get('documents', []))}")
            elif 'document_count' in result:
                print(f"   Documents uploaded: {result.get('document_count', 0)}")
                
            if 'created_at' in result:
                print(f"   Session created: {result.get('created_at')}")
                
        else:
            print(f"   Raw response: {result}")
        
        return result
            
    except Exception as error:
        print(f"❌ Session data retrieval failed: {error}")
        return None

async def test_knowledge_cleanup(session: aiohttp.ClientSession):
    """Test knowledge base cleanup with comprehensive test cases"""
    print("\n🧹 Testing knowledge base cleanup...")
    
    try:
        headers = {'Content-Type': 'application/json'}
        
        # Test 1: Delete by document ID
        print("🔍 Testing document deletion by ID...")
        
        # First upload a test document
        upload_result = await api_request(
            session=session,
            endpoint='/api/knowledge/upload',
            method='POST',
            data={
                'type': 'text',
                'content': 'Test document for deletion by ID',
                'session_id': f'cleanup_test_{int(time.time())}',
                'user_id': TEST_USER_ID,
                'metadata': json.dumps({'test': 'delete_by_id'})
            }
        )
        
        document_id = upload_result.get('document_ids', [None])[0]
        if not document_id:
            raise Exception("Failed to upload test document for deletion")
        
        # Delete by document ID
        delete_result = await api_request(
            session=session,
            endpoint=f'/api/knowledge/documents?document_ids={document_id}',
            method='DELETE'
        )
        
        assert delete_result.get('deleted_count', 0) > 0, "Document deletion by ID failed"
        print("✅ Document deletion by ID successful")
        
        # Test 2: Delete by session ID
        print("\n🔍 Testing document deletion by session ID...")
        test_session_id = f'cleanup_session_{int(time.time())}'
        
        # Upload multiple documents to the same session
        for i in range(3):
            await api_request(
                session=session,
                endpoint='/api/knowledge/upload',
                method='POST',
                data={
                    'type': 'text',
                    'content': f'Test document {i} for session deletion',
                    'session_id': test_session_id,
                    'user_id': TEST_USER_ID,
                    'metadata': json.dumps({'test': 'delete_by_session', 'doc_num': i})
                }
            )
        
        # Verify documents were uploaded
        session_data = await api_request(
            session=session,
            endpoint=f'/api/knowledge/sessions/{test_session_id}?include_documents=true'
        )
        assert len(session_data.get('documents', [])) == 3, "Failed to upload test documents"
        
        # Delete by session ID
        delete_result = await api_request(
            session=session,
            endpoint=f'/api/knowledge/documents?session_id={test_session_id}',
            method='DELETE'
        )
        
        assert delete_result.get('deleted_count', 0) == 3, "Session deletion failed"
        print("✅ Session deletion successful")
        
        # Test 3: Delete by user ID
        print("\n🔍 Testing document deletion by user ID...")
        test_user_id = f'test_user_cleanup_{int(time.time())}'
        
        # Upload documents for test user
        for i in range(2):
            await api_request(
                session=session,
                endpoint='/api/knowledge/upload',
                method='POST',
                data={
                    'type': 'text',
                    'content': f'Test document {i} for user {test_user_id}',
                    'session_id': f'user_cleanup_{i}',
                    'user_id': test_user_id,
                    'metadata': json.dumps({'test': 'delete_by_user'})
                }
            )
        
        # Verify uploads
        user_stats = await api_request(
            session=session,
            endpoint=f'/api/knowledge/stats?user_id={test_user_id}'
        )
        assert user_stats.get('stats', {}).get('documents', 0) == 2, "Failed to upload test user documents"
        
        # Delete by user ID
        delete_result = await api_request(
            session=session,
            endpoint=f'/api/knowledge/documents?user_id={test_user_id}',
            method='DELETE'
        )
        
        assert delete_result.get('deleted_count', 0) == 2, "User document deletion failed"
        print("✅ User document deletion successful")
        
        # Test 4: Delete with metadata filter
        print("\n🔍 Testing document deletion with metadata filter...")
        metadata_filter = {'department': 'engineering', 'project': 'test_cleanup'}
        
        # Upload test documents with specific metadata
        for i in range(2):
            await api_request(
                session=session,
                endpoint='/api/knowledge/upload',
                method='POST',
                data={
                    'type': 'text',
                    'content': f'Engineering test document {i}',
                    'session_id': f'metadata_cleanup_{i}',
                    'user_id': TEST_USER_ID,
                    'metadata': json.dumps(metadata_filter)
                }
            )
        
        # Delete with metadata filter
        delete_result = await api_request(
            session=session,
            endpoint=f'/api/knowledge/documents',
            method='DELETE',
            data={'filter_metadata': metadata_filter}
        )
        
        assert delete_result.get('deleted_count', 0) == 2, "Metadata filter deletion failed"
        print("✅ Metadata filter deletion successful")
        
        # Test 5: Cleanup of non-existent documents (should not fail)
        print("\n🔍 Testing cleanup of non-existent documents...")
        delete_result = await api_request(
            session=session,
            endpoint='/api/knowledge/documents?document_ids=nonexistent123',
            method='DELETE'
        )
        
        assert delete_result.get('deleted_count', 0) == 0, "Should return 0 for non-existent document"
        print("✅ Cleanup of non-existent documents handled correctly")
        
        return {
            'success': True,
            'message': 'All cleanup tests passed',
            'deleted_count': 0  # We don't track total across all tests
        }
        
    except Exception as error:
        print(f"❌ Cleanup test failed: {error}")
        raise error

async def test_knowledge_base():
    """Main test function for knowledge base"""
    print('🧪 Knowledge Base Complete Test')
    print('Testing document upload, search, and management functionality...\n')
    
    async with aiohttp.ClientSession() as session:
        try:
            # 1. Health Check
            print('1. Testing system health...')
            health = await api_request(session, '/health')
            print(f"✅ System Status: {health.get('status')}\n")
            
            # 2. Test Document Uploads
            print('2. Testing document upload methods...')
            text_result = await test_text_upload(session)
            url_result = await test_url_upload(session)
            file_result = await test_file_upload(session)
            
            # 3. Test Knowledge Search
            search_results = await test_knowledge_search(session)
            
            # 4. Test Statistics
            stats_result = await test_knowledge_stats(session)
            
            # 5. Test Session Data
            session_result = await test_session_data(session)
            
            # 6. Generate Test Report
            print('\n6. Test Results Summary')
            print('=' * 50)
            
            # Upload results
            upload_success = sum([
                1 if text_result else 0,
                1 if url_result else 0,
                1 if file_result else 0
            ])
            print(f"Document Uploads: {upload_success}/3 successful")
            
            # Search results
            successful_searches = sum(1 for r in search_results if r['success'])
            total_searches = len(search_results)
            print(f"Knowledge Searches: {successful_searches}/{total_searches} successful")
            
            # Other tests
            other_tests = sum([
                1 if stats_result else 0,
                1 if session_result else 0
            ])
            print(f"Additional Tests: {other_tests}/2 successful")
            
            print(f"\nTest Session ID: {TEST_SESSION_ID}")
            
            # 7. Cleanup (optional)
            cleanup_result = await test_knowledge_cleanup(session)
            
            print("🎉 Knowledge base test completed!")
            
            # Overall success
            total_success = upload_success + successful_searches + other_tests
            total_tests = 3 + total_searches + 2
            success_rate = total_success / total_tests if total_tests > 0 else 0
            
            if success_rate >= 0.7:  # 70% success rate
                print("✅ Overall test: PASSED")
                return True
            else:
                print("❌ Overall test: FAILED")
                return False
                
        except Exception as error:
            print(f"❌ Test execution failed: {error}")
            return False

async def main():
    """Main entry point"""
    try:
        success = await test_knowledge_base()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as error:
        print(f"❌ Unexpected error: {error}")
        sys.exit(1)

if __name__ == '__main__':
    asyncio.run(main())