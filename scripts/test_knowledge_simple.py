"""
Simple Knowledge Base Testing Script

This script provides a simpler test for the knowledge base search functionality:
1. Uploads a text document
2. Searches the knowledge base
3. Cleanup

This is designed to verify that the search functionality works correctly.
"""

import asyncio
import sys
import os
from datetime import datetime
import httpx

# Add parent directory to path to import from app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

async def main():
    base_url = "http://localhost:8000"
    session_id = f"test_session_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    # Test data
    test_text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, 
    as opposed to natural intelligence displayed by animals including humans. 
    AI research has been defined as the field of study of intelligent agents, 
    which refers to any system that perceives its environment and takes actions 
    that maximize its chance of achieving its goals.
    """
    
    print("\n===== Simple Knowledge Base Test =====\n")
    
    # Test API server connection
    print("1. Testing API server connection...")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{base_url}/health")
            if response.status_code == 200:
                print("✅ API server is running")
            else:
                print(f"❌ API server returned status {response.status_code}")
                return
    except Exception as e:
        print(f"❌ Failed to connect to API server: {e}")
        print("   Is the FastAPI server running on port 8000?")
        return
    
    # Upload text document
    print("\n2. Uploading test document...")
    try:
        async with httpx.AsyncClient() as client:
            data = {
                "type": "text",
                "content": test_text,
                "session_id": session_id,
                "user_id": "test_user",
                "metadata": "{\"test\": true}"
            }
            response = await client.post(
                f"{base_url}/api/knowledge/upload/text",
                data=data
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Document uploaded successfully")
                print(f"   Session ID: {session_id}")
                print(f"   Success: {result.get('success', False)}")
                print(f"   Chunks: {result.get('num_chunks', 0)}")
            else:
                print(f"❌ Failed to upload document: {response.status_code}")
                print(f"   Error: {response.text}")
                return
    except Exception as e:
        print(f"❌ Error during upload: {e}")
        return
    
    # Search the knowledge base
    print("\n3. Searching knowledge base...")
    search_query = "What is artificial intelligence?"
    
    try:
        async with httpx.AsyncClient() as client:
            params = {
                "q": search_query,
                "session_id": session_id,
                "user_id": "test_user",
                "limit": 5,
                "similarity_threshold": 0.5
            }
            response = await client.get(
                f"{base_url}/api/knowledge/search",
                params=params
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Search completed successfully")
                print(f"   Query: '{search_query}'")
                print(f"   Success: {result.get('success', False)}")
                print(f"   Results: {len(result.get('results', []))}")
                
                if result.get('results'):
                    print("\n   Top result:")
                    top_result = result['results'][0]
                    print(f"   - Score: {top_result.get('score', 0)}")
                    content = top_result.get('content', '')
                    print(f"   - Content: {content[:100]}..." if len(content) > 100 else content)
                elif result.get('error'):
                    print(f"❌ Search error: {result.get('error')}")
            else:
                print(f"❌ Failed to search: {response.status_code}")
                print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Error during search: {e}")
    
    # Cleanup - delete test data
    print("\n4. Cleaning up test data...")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.delete(
                f"{base_url}/api/knowledge/documents",
                params={"session_id": session_id}
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Deleted {result.get('deleted_count', 0)} document chunks")
            else:
                print(f"⚠️ Cleanup returned status {response.status_code}")
    except Exception as e:
        print(f"⚠️ Error during cleanup: {e}")
    
    print("\n===== Test Complete =====")

if __name__ == "__main__":
    asyncio.run(main())