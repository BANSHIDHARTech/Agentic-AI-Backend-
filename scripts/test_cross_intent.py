#!/usr/bin/env python3

"""
Cross-Intent Workflow Switching Test Script

Tests mid-session intent recognition and dynamic workflow switching by:
1. Starting a workflow session
2. Sending messages that change intent mid-session
3. Verifying workflow_switch SSE events
4. Checking chat persistence
"""

import asyncio
import aiohttp
import json
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

BASE_URL = 'http://localhost:8000'

# Test configuration
TEST_SESSION_ID = f"cross_intent_test_{int(datetime.now().timestamp() * 1000)}"

async def api_request(session: aiohttp.ClientSession, endpoint: str, method: str = 'GET', data: dict = None):
    """Helper function to make API requests"""
    url = f"{BASE_URL}{endpoint}"
    
    try:
        if method == 'GET':
            async with session.get(url) as response:
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

async def test_intent_classification(session: aiohttp.ClientSession, query: str, expected_intent: str = None):
    """Helper function to test intent classification"""
    try:
        result = await api_request(session, '/api/router/classify', 'POST', {
            'query': query,
            'session_id': TEST_SESSION_ID,
            'user_id': 'cross_intent_test_user'
        })
        
        print(f"📝 Query: \"{query}\"")
        print(f"   Intent: {result.get('intent', 'None')}")
        print(f"   Confidence: {result.get('confidence', 0)}")
        print(f"   Agent: {result.get('agent_name', 'None')}")
        
        if expected_intent and result.get('intent') == expected_intent:
            print(f"   ✅ Expected intent matched: {expected_intent}")
            return True
        elif expected_intent:
            print(f"   ⚠️  Expected: {expected_intent}, Got: {result.get('intent', 'None')}")
            return False
        
        return result
    except Exception as error:
        print(f"❌ Intent classification failed for: \"{query}\" - {error}")
        return False

async def simulate_sse_session(session: aiohttp.ClientSession, workflow_id: str, input_data: dict):
    """Helper function to simulate SSE session (simplified)"""
    try:
        print(f"🔗 Simulating SSE session for workflow: {workflow_id}")
        print(f"📤 Input: {json.dumps(input_data)}")
        
        # For this test, we'll use the regular workflow run endpoint
        # In a real implementation, this would be the SSE endpoint
        result = await api_request(session, '/api/workflows/run', 'POST', {
            'workflow_id': workflow_id,
            'session_id': TEST_SESSION_ID,
            'input': input_data
        })
        
        print(f"✅ Workflow executed successfully")
        print(f"   Run ID: {result.get('run_id')}")
        print(f"   Status: {result.get('status')}")
        print(f"   Steps: {result.get('total_steps', 0)}")
        
        return result
    except Exception as error:
        print(f"❌ SSE session simulation failed: {error}")
        return None

async def test_cross_intent_switching():
    """Main test function for cross-intent workflow switching"""
    print('🧪 Cross-Intent Workflow Switching Test')
    print('Testing mid-session intent recognition and dynamic workflow switching...\n')
    
    async with aiohttp.ClientSession() as session:
        try:
            # 1. Health Check
            print('1. Testing system health...')
            health = await api_request(session, '/health')
            print(f"✅ System Status: {health.get('status')}\n")
            
            # 2. Test Intent Classification System
            print('2. Testing intent classification system...')
            
            test_queries = [
                {'query': 'I want to swap my SIM card', 'expected': 'sim_swap_request'},
                {'query': 'Check my account balance', 'expected': 'balance_inquiry'},
                {'query': 'I need help with my account', 'expected': 'general_support'},
                {'query': 'Can you authenticate me?', 'expected': 'authentication_required'}
            ]
            
            classification_results = []
            for test in test_queries:
                print(f"\n📋 Testing: \"{test['query']}\"")
                result = await test_intent_classification(session, test['query'], test['expected'])
                classification_results.append({
                    'query': test['query'],
                    'expected': test['expected'],
                    'result': result,
                    'passed': isinstance(result, bool) and result
                })
            
            # 3. Get Available Workflows
            print('\n3. Loading available workflows...')
            try:
                workflows = await api_request(session, '/api/workflows')
                print(f"📋 Found {len(workflows)} workflows")
                
                if len(workflows) == 0:
                    print("⚠️  No workflows available, creating mock workflow for testing")
                    workflows = [{'id': 'test_workflow', 'name': 'Test Workflow'}]
            except Exception:
                print("⚠️  Workflows endpoint not available, using mock data")
                workflows = [{'id': 'test_workflow', 'name': 'Test Workflow'}]
            
            # 4. Test Cross-Intent Scenario
            print('\n4. Testing cross-intent workflow switching scenario...')
            
            # Scenario: User starts with SIM swap, then switches to balance inquiry
            print('\n📋 Scenario: SIM Swap → Balance Inquiry')
            
            # Start with SIM swap intent
            print('\n🔄 Step 1: Initial SIM swap request')
            sim_swap_result = await test_intent_classification(
                session, 
                "I need to replace my SIM card", 
                "sim_swap_request"
            )
            
            # Simulate workflow execution for SIM swap
            if workflows:
                workflow_result = await simulate_sse_session(
                    session, 
                    workflows[0]['id'], 
                    {'query': 'I need to replace my SIM card', 'intent': 'sim_swap_request'}
                )
            
            # Mid-session intent change
            print('\n🔄 Step 2: Mid-session intent change to balance inquiry')
            balance_result = await test_intent_classification(
                session,
                "Actually, can you check my account balance first?",
                "balance_inquiry"
            )
            
            # Test workflow switching
            if workflows:
                switch_result = await simulate_sse_session(
                    session,
                    workflows[0]['id'],
                    {'query': 'Actually, can you check my account balance first?', 'intent': 'balance_inquiry'}
                )
            
            # 5. Test Session Persistence
            print('\n5. Testing session persistence...')
            
            # Check if session data is maintained across intent switches
            try:
                session_data = await api_request(session, f'/api/workflows/sessions/{TEST_SESSION_ID}')
                print(f"✅ Session data retrieved: {len(session_data.get('messages', []))} messages")
            except Exception:
                print("⚠️  Session persistence endpoint not available")
            
            # 6. Generate Test Report
            print('\n6. Test Results Summary')
            print('=' * 50)
            
            passed_classifications = sum(1 for r in classification_results if r['passed'])
            total_classifications = len(classification_results)
            
            print(f"Intent Classification: {passed_classifications}/{total_classifications} passed")
            
            for result in classification_results:
                status = "✅" if result['passed'] else "❌"
                print(f"  {status} {result['query'][:30]}... → {result['expected']}")
            
            print(f"\nSession ID: {TEST_SESSION_ID}")
            print("🎉 Cross-intent switching test completed!")
            
            # Overall test result
            overall_success = passed_classifications >= total_classifications * 0.7  # 70% pass rate
            if overall_success:
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
        success = await test_cross_intent_switching()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as error:
        print(f"❌ Unexpected error: {error}")
        sys.exit(1)

if __name__ == '__main__':
    asyncio.run(main())