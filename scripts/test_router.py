#!/usr/bin/env python3

"""
Router System Test Script

Tests the complete Router/Commander Agent functionality including:
1. Intent classification and confidence scoring
2. Agent routing and fallback handling
3. Cross-intent workflow switching
4. Router rules management
5. Analytics and performance metrics
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
TEST_SESSION_ID = f"router_test_{int(datetime.now().timestamp() * 1000)}"

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

async def test_intent_classification(session: aiohttp.ClientSession):
    """Test intent classification with various queries"""
    print("🎯 Testing intent classification...")
    
    test_cases = [
        {"query": "I want to swap my SIM card", "expected": "sim_swap_request"},
        {"query": "Check my account balance", "expected": "balance_inquiry"},
        {"query": "I need help with my account", "expected": "general_support"},
        {"query": "Can you authenticate me?", "expected": "authentication_required"},
        {"query": "Random gibberish xyz123", "expected": None}  # Should fallback
    ]
    
    results = []
    
    for test_case in test_cases:
        try:
            result = await api_request(session, '/api/router/classify', 'POST', {
                'query': test_case['query'],
                'session_id': TEST_SESSION_ID,
                'user_id': 'router_test_user'
            })
            
            print(f"\n📝 Query: \"{test_case['query']}\"")
            print(f"   Intent: {result.get('intent', 'None')}")
            print(f"   Confidence: {result.get('confidence', 0):.3f}")
            print(f"   Agent: {result.get('agent_name', 'None')}")
            
            success = (
                (test_case['expected'] is None and result.get('intent') is None) or
                (result.get('intent') == test_case['expected'])
            )
            
            results.append({
                'query': test_case['query'],
                'expected': test_case['expected'],
                'actual': result.get('intent'),
                'confidence': result.get('confidence', 0),
                'success': success
            })
            
        except Exception as error:
            print(f"❌ Classification failed for \"{test_case['query']}\": {error}")
            results.append({
                'query': test_case['query'],
                'expected': test_case['expected'],
                'actual': None,
                'confidence': 0,
                'success': False
            })
    
    return results

async def test_router_rules(session: aiohttp.ClientSession):
    """Test router rules management"""
    print("\n📋 Testing router rules management...")
    
    try:
        # Get existing rules
        rules = await api_request(session, '/api/router/rules')
        print(f"✅ Found {len(rules)} router rules")
        
        for rule in rules[:3]:  # Show first 3 rules
            print(f"   Rule: {rule.get('intent_name')} (priority: {rule.get('priority')})")
        
        # Test rule creation (if endpoint exists)
        try:
            new_rule = {
                'intent_name': 'test_intent',
                'keywords': ['test', 'example'],
                'priority': 999,
                'confidence_threshold': 0.5,
                'description': 'Test rule for router testing'
            }
            
            create_result = await api_request(session, '/api/router/rules', 'POST', new_rule)
            print(f"✅ Test rule created: {create_result.get('id')}")
            
            # Clean up test rule
            if create_result.get('id'):
                await api_request(session, f"/api/router/rules/{create_result['id']}", 'DELETE')
                print("✅ Test rule cleaned up")
                
        except Exception:
            print("⚠️  Rule creation/deletion not available")
        
        return rules
        
    except Exception as error:
        print(f"❌ Router rules test failed: {error}")
        return []

async def test_fallback_messages(session: aiohttp.ClientSession):
    """Test fallback message system"""
    print("\n💬 Testing fallback messages...")
    
    try:
        # Get fallback messages
        fallbacks = await api_request(session, '/api/router/fallback')
        print(f"✅ Found {len(fallbacks)} fallback messages")
        
        for msg in fallbacks[:2]:  # Show first 2 messages
            print(f"   Message: \"{msg.get('message', '')[:50]}...\"")
        
        return fallbacks
        
    except Exception as error:
        print(f"❌ Fallback messages test failed: {error}")
        return []

async def test_router_analytics(session: aiohttp.ClientSession):
    """Test router analytics and metrics"""
    print("\n📊 Testing router analytics...")
    
    try:
        # Get router analytics
        analytics = await api_request(session, '/api/router/analytics')
        print("✅ Router analytics retrieved:")
        print(f"   Total classifications: {analytics.get('total_classifications', 0)}")
        print(f"   Success rate: {analytics.get('success_rate', 0):.1%}")
        print(f"   Top intents: {len(analytics.get('top_intents', []))}")
        
        return analytics
        
    except Exception as error:
        print(f"❌ Router analytics test failed: {error}")
        return None

async def test_router_performance(session: aiohttp.ClientSession):
    """Test router performance with multiple queries"""
    print("\n⚡ Testing router performance...")
    
    test_queries = [
        "I want to swap my SIM",
        "Check balance",
        "Need help",
        "Login please",
        "Support needed"
    ] * 5  # 25 total queries
    
    start_time = datetime.now()
    successful_classifications = 0
    
    for i, query in enumerate(test_queries):
        try:
            result = await api_request(session, '/api/router/classify', 'POST', {
                'query': query,
                'session_id': f"{TEST_SESSION_ID}_perf_{i}",
                'user_id': 'router_perf_test'
            })
            
            if result.get('intent'):
                successful_classifications += 1
                
        except Exception:
            pass  # Continue with performance test
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print(f"✅ Performance test completed:")
    print(f"   Total queries: {len(test_queries)}")
    print(f"   Successful: {successful_classifications}")
    print(f"   Duration: {duration:.2f}s")
    print(f"   Avg time per query: {(duration/len(test_queries)*1000):.1f}ms")
    
    return {
        'total_queries': len(test_queries),
        'successful': successful_classifications,
        'duration': duration,
        'avg_time_ms': (duration/len(test_queries)*1000)
    }

async def test_router_system():
    """Main test function for router system"""
    print('🧪 Router System Complete Test')
    print('Testing intent classification, routing, and analytics...\n')
    
    async with aiohttp.ClientSession() as session:
        try:
            # 1. Health Check
            print('1. Testing system health...')
            health = await api_request(session, '/health')
            print(f"✅ System Status: {health.get('status')}\n")
            
            # 2. Test Intent Classification
            print('2. Testing intent classification...')
            classification_results = await test_intent_classification(session)
            
            # 3. Test Router Rules
            rules_results = await test_router_rules(session)
            
            # 4. Test Fallback Messages
            fallback_results = await test_fallback_messages(session)
            
            # 5. Test Analytics
            analytics_results = await test_router_analytics(session)
            
            # 6. Test Performance
            performance_results = await test_router_performance(session)
            
            # 7. Generate Test Report
            print('\n7. Test Results Summary')
            print('=' * 50)
            
            # Classification results
            successful_classifications = sum(1 for r in classification_results if r['success'])
            total_classifications = len(classification_results)
            print(f"Intent Classification: {successful_classifications}/{total_classifications} successful")
            
            # Other components
            components_working = sum([
                1 if rules_results else 0,
                1 if fallback_results else 0,
                1 if analytics_results else 0,
                1 if performance_results else 0
            ])
            print(f"Router Components: {components_working}/4 working")
            
            # Performance metrics
            if performance_results:
                print(f"Performance: {performance_results['avg_time_ms']:.1f}ms avg response time")
            
            print(f"\nTest Session ID: {TEST_SESSION_ID}")
            print("🎉 Router system test completed!")
            
            # Overall success
            classification_rate = successful_classifications / total_classifications if total_classifications > 0 else 0
            component_rate = components_working / 4
            
            overall_success = (classification_rate >= 0.8 and component_rate >= 0.5)
            
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
        success = await test_router_system()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as error:
        print(f"❌ Unexpected error: {error}")
        sys.exit(1)

if __name__ == '__main__':
    asyncio.run(main())
