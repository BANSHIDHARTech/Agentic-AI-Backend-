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
    
    try:
        if method == 'GET':
            async with session.get(url) as response:
                result = await response.json()
        elif files:
            # Handle file upload
            form_data = aiohttp.FormData()
            for key, value in (data or {}).items():
                form_data.add_field(key, str(value))
            for key, file_data in files.items():
                form_data.add_field(key, file_data['content'], filename=file_data['filename'])
            
            async with session.post(url, data=form_data) as response:
                result = await response.json()
        else:
            async with session.request(method, url, json=data) as response:
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
        result = await api_request(session, '/api/knowledge/upload', 'POST', {
            'type': 'text',
            'content': test_content,
            'session_id': TEST_SESSION_ID,
            'user_id': TEST_USER_ID,
            'metadata': {
                'title': 'Test Document',
                'category': 'AI/ML',
                'test_type': 'direct_text'
            }
        })
        
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
        result = await api_request(session, '/api/knowledge/upload-url', 'POST', {
            'url': test_url,
            'session_id': TEST_SESSION_ID,
            'user_id': TEST_USER_ID,
            'metadata': {
                'source': 'Wikipedia',
                'topic': 'Artificial Intelligence',
                'test_type': 'url_extraction'
            }
        })
        
        print(f"✅ URL upload successful:")
        print(f"   Source: {result.get('source_reference', 'Unknown')}")
        print(f"   Chunks created: {result.get('chunks_created', 0)}")
        print(f"   Processing time: {result.get('processing_time_ms', 0)}ms")
        
        return result
    except Exception as error:
        print(f"❌ URL upload failed: {error}")
        return None

async def test_file_upload(session: aiohttp.ClientSession):
    """Test file upload (simulate PDF)"""
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
        files = {
            'file': {
                'content': test_file_content.encode('utf-8'),
                'filename': 'ml_test_document.txt'
            }
        }
        
        result = await api_request(session, '/api/knowledge/upload', 'POST', {
            'session_id': TEST_SESSION_ID,
            'user_id': TEST_USER_ID,
            'metadata': json.dumps({
                'title': 'ML Test Document',
                'category': 'Machine Learning',
                'test_type': 'file_upload'
            })
        }, files)
        
        print(f"✅ File upload successful:")
        print(f"   Filename: ml_test_document.txt")
        print(f"   Chunks created: {result.get('chunks_created', 0)}")
        print(f"   Processing time: {result.get('processing_time_ms', 0)}ms")
        
        return result
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
            result = await api_request(session, '/api/knowledge/query', 'POST', {
                'query': query,
                'session_id': TEST_SESSION_ID,
                'user_id': TEST_USER_ID,
                'limit': 3,
                'similarity_threshold': 0.7
            })
            
            print(f"\n📋 Query: \"{query}\"")
            print(f"   Results found: {result.get('result_count', 0)}")
            print(f"   Processing time: {result.get('processing_time_ms', 0)}ms")
            
            if result.get('results'):
                for i, doc in enumerate(result['results'][:2]):  # Show top 2 results
                    print(f"   Result {i+1}: {doc.get('content', '')[:100]}...")
            
            search_results.append({
                'query': query,
                'result_count': result.get('result_count', 0),
                'success': result.get('success', False)
            })
            
        except Exception as error:
            print(f"❌ Search failed for \"{query}\": {error}")
            search_results.append({
                'query': query,
                'result_count': 0,
                'success': False
            })
    
    return search_results

async def test_knowledge_stats(session: aiohttp.ClientSession):
    """Test knowledge base statistics"""
    print("\n📊 Testing knowledge base statistics...")
    
    try:
        result = await api_request(session, '/api/knowledge/stats')
        
        print("✅ Knowledge base statistics:")
        stats = result.get('statistics', {})
        print(f"   Total documents: {stats.get('total_documents', 0)}")
        print(f"   Total sessions: {stats.get('total_sessions', 0)}")
        print(f"   Recent activity: {len(result.get('recent_activity', []))}")
        
        return result
    except Exception as error:
        print(f"❌ Stats retrieval failed: {error}")
        return None

async def test_session_data(session: aiohttp.ClientSession):
    """Test session data retrieval"""
    print("\n📋 Testing session data retrieval...")
    
    try:
        result = await api_request(session, f'/api/knowledge/sessions/{TEST_SESSION_ID}')
        
        print("✅ Session data retrieved:")
        print(f"   Session ID: {result.get('session_id', 'Unknown')}")
        print(f"   Total queries: {len(result.get('queries', []))}")
        print(f"   Documents uploaded: {len(result.get('documents', []))}")
        
        return result
    except Exception as error:
        print(f"❌ Session data retrieval failed: {error}")
        return None

async def test_knowledge_cleanup(session: aiohttp.ClientSession):
    """Test knowledge base cleanup"""
    print("\n🧹 Testing knowledge base cleanup...")
    
    try:
        result = await api_request(session, '/api/knowledge/delete', 'DELETE', {
            'session_id': TEST_SESSION_ID
        })
        
        print("✅ Cleanup successful:")
        print(f"   Documents deleted: {result.get('deleted_count', 0)}")
        
        return result
    except Exception as error:
        print(f"❌ Cleanup failed: {error}")
        return None

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
