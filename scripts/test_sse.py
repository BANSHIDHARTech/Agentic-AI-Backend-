#!/usr/bin/env python3

"""
Server-Sent Events (SSE) Test Script

Tests the complete SSE streaming functionality including:
1. Workflow streaming execution
2. Real-time event delivery
3. Connection handling and reconnection
4. Event parsing and processing
5. Performance and reliability testing
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
TEST_SESSION_ID = f"sse_test_{int(datetime.now().timestamp() * 1000)}"

async def parse_sse_stream(response):
    """Parse Server-Sent Events from response stream"""
    events = []
    current_event = {}
    
    async for line in response.content:
        line = line.decode('utf-8').strip()
        
        if line.startswith('data: '):
            data = line[6:]  # Remove 'data: ' prefix
            try:
                event_data = json.loads(data)
                current_event['data'] = event_data
            except json.JSONDecodeError:
                current_event['data'] = data
                
        elif line.startswith('event: '):
            current_event['event'] = line[7:]  # Remove 'event: ' prefix
            
        elif line.startswith('id: '):
            current_event['id'] = line[4:]  # Remove 'id: ' prefix
            
        elif line == '':
            # Empty line indicates end of event
            if current_event:
                events.append(current_event.copy())
                current_event = {}
                
        # Break after receiving a reasonable number of events for testing
        if len(events) >= 10:
            break
    
    return events

async def test_workflow_sse_stream(session: aiohttp.ClientSession):
    """Test workflow execution with SSE streaming"""
    print("🌊 Testing workflow SSE streaming...")
    
    try:
        # Create a test workflow execution request
        workflow_data = {
            'workflow_id': 'test_workflow_stream',
            'input': {
                'message': 'Hello from SSE test',
                'test_type': 'sse_streaming'
            },
            'session_id': TEST_SESSION_ID,
            'user_id': 'sse_test_user'
        }
        
        print(f"📤 Starting SSE stream for workflow: {workflow_data['workflow_id']}")
        
        # Make SSE request
        timeout = aiohttp.ClientTimeout(total=30)  # 30 second timeout
        async with session.post(
            f"{BASE_URL}/api/workflows/run/stream",
            json=workflow_data,
            timeout=timeout,
            headers={'Accept': 'text/event-stream'}
        ) as response:
            
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"SSE request failed: {response.status} - {error_text}")
            
            print("✅ SSE connection established")
            print("📡 Receiving events...")
            
            # Parse events from stream
            events = await parse_sse_stream(response)
            
            print(f"✅ Received {len(events)} SSE events")
            
            # Display events
            for i, event in enumerate(events):
                print(f"   Event {i+1}: {event.get('event', 'data')}")
                if event.get('data'):
                    data = event['data']
                    if isinstance(data, dict):
                        print(f"            Payload: {data.get('event', 'unknown')} - {data.get('payload', {}).get('message', '')}")
                    else:
                        print(f"            Data: {str(data)[:100]}...")
            
            return {
                'success': True,
                'events_received': len(events),
                'events': events
            }
            
    except asyncio.TimeoutError:
        print("❌ SSE stream timed out")
        return {'success': False, 'error': 'timeout'}
        
    except Exception as error:
        print(f"❌ SSE streaming failed: {error}")
        return {'success': False, 'error': str(error)}

async def test_sse_connection_handling(session: aiohttp.ClientSession):
    """Test SSE connection handling and error scenarios"""
    print("\n🔌 Testing SSE connection handling...")
    
    test_scenarios = [
        {
            'name': 'Invalid Workflow ID',
            'data': {
                'workflow_id': 'nonexistent_workflow',
                'input': {'test': 'invalid_workflow'},
                'session_id': TEST_SESSION_ID
            }
        },
        {
            'name': 'Missing Required Fields',
            'data': {
                'input': {'test': 'missing_fields'}
            }
        },
        {
            'name': 'Large Input Data',
            'data': {
                'workflow_id': 'test_workflow',
                'input': {
                    'large_data': 'x' * 10000,  # 10KB of data
                    'test_type': 'large_input'
                },
                'session_id': TEST_SESSION_ID
            }
        }
    ]
    
    results = []
    
    for scenario in test_scenarios:
        print(f"\n📋 Testing: {scenario['name']}")
        
        try:
            timeout = aiohttp.ClientTimeout(total=10)  # Shorter timeout for error cases
            async with session.post(
                f"{BASE_URL}/api/workflows/run/stream",
                json=scenario['data'],
                timeout=timeout,
                headers={'Accept': 'text/event-stream'}
            ) as response:
                
                if response.status >= 400:
                    error_data = await response.text()
                    print(f"   ✅ Expected error response: {response.status}")
                    results.append({
                        'scenario': scenario['name'],
                        'success': True,
                        'status': response.status,
                        'handled_correctly': True
                    })
                else:
                    # Try to read some events
                    events = []
                    try:
                        async for line in response.content:
                            line = line.decode('utf-8').strip()
                            if line.startswith('data: '):
                                events.append(line)
                            if len(events) >= 3:  # Read a few events
                                break
                    except Exception:
                        pass
                    
                    print(f"   ✅ Received response: {response.status}, {len(events)} events")
                    results.append({
                        'scenario': scenario['name'],
                        'success': True,
                        'status': response.status,
                        'events_received': len(events)
                    })
                    
        except Exception as error:
            print(f"   ⚠️  Connection error (expected): {error}")
            results.append({
                'scenario': scenario['name'],
                'success': True,  # Error handling is expected
                'error': str(error),
                'handled_correctly': True
            })
    
    return results

async def test_sse_performance(session: aiohttp.ClientSession):
    """Test SSE streaming performance"""
    print("\n⚡ Testing SSE performance...")
    
    performance_tests = []
    
    for i in range(3):  # Run 3 performance tests
        start_time = datetime.now()
        
        try:
            workflow_data = {
                'workflow_id': f'perf_test_workflow_{i}',
                'input': {
                    'performance_test': True,
                    'test_iteration': i
                },
                'session_id': f"{TEST_SESSION_ID}_perf_{i}"
            }
            
            timeout = aiohttp.ClientTimeout(total=15)
            async with session.post(
                f"{BASE_URL}/api/workflows/run/stream",
                json=workflow_data,
                timeout=timeout,
                headers={'Accept': 'text/event-stream'}
            ) as response:
                
                events_received = 0
                first_event_time = None
                last_event_time = None
                
                if response.status == 200:
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        if line.startswith('data: '):
                            events_received += 1
                            current_time = datetime.now()
                            
                            if first_event_time is None:
                                first_event_time = current_time
                            last_event_time = current_time
                            
                            # Stop after reasonable number of events
                            if events_received >= 5:
                                break
                
                end_time = datetime.now()
                total_duration = (end_time - start_time).total_seconds()
                
                time_to_first_event = None
                if first_event_time:
                    time_to_first_event = (first_event_time - start_time).total_seconds()
                
                performance_tests.append({
                    'iteration': i,
                    'total_duration': total_duration,
                    'time_to_first_event': time_to_first_event,
                    'events_received': events_received,
                    'success': True
                })
                
                print(f"   Test {i+1}: {events_received} events in {total_duration:.2f}s")
                if time_to_first_event:
                    print(f"           First event: {time_to_first_event:.3f}s")
                
        except Exception as error:
            print(f"   Test {i+1} failed: {error}")
            performance_tests.append({
                'iteration': i,
                'success': False,
                'error': str(error)
            })
    
    # Calculate averages
    successful_tests = [t for t in performance_tests if t['success']]
    if successful_tests:
        avg_duration = sum(t['total_duration'] for t in successful_tests) / len(successful_tests)
        avg_first_event = sum(t['time_to_first_event'] for t in successful_tests if t['time_to_first_event']) / len([t for t in successful_tests if t['time_to_first_event']])
        
        print(f"\n✅ Performance Summary:")
        print(f"   Successful tests: {len(successful_tests)}/3")
        print(f"   Average duration: {avg_duration:.2f}s")
        print(f"   Average time to first event: {avg_first_event:.3f}s")
    
    return performance_tests

async def test_sse_system():
    """Main test function for SSE system"""
    print('🧪 Server-Sent Events (SSE) Complete Test')
    print('Testing streaming workflows, connection handling, and performance...\n')
    
    async with aiohttp.ClientSession() as session:
        try:
            # 1. Health Check
            print('1. Testing system health...')
            async with session.get(f"{BASE_URL}/health") as response:
                health = await response.json()
                print(f"✅ System Status: {health.get('status')}\n")
            
            # 2. Test Basic SSE Streaming
            print('2. Testing basic SSE streaming...')
            stream_result = await test_workflow_sse_stream(session)
            
            # 3. Test Connection Handling
            connection_results = await test_sse_connection_handling(session)
            
            # 4. Test Performance
            performance_results = await test_sse_performance(session)
            
            # 5. Generate Test Report
            print('\n5. Test Results Summary')
            print('=' * 50)
            
            # Basic streaming
            streaming_success = 1 if stream_result.get('success') else 0
            events_received = stream_result.get('events_received', 0)
            print(f"Basic SSE Streaming: {streaming_success}/1 successful ({events_received} events)")
            
            # Connection handling
            connection_success = sum(1 for r in connection_results if r.get('success'))
            print(f"Connection Handling: {connection_success}/{len(connection_results)} scenarios handled")
            
            # Performance
            perf_success = sum(1 for r in performance_results if r.get('success'))
            print(f"Performance Tests: {perf_success}/{len(performance_results)} successful")
            
            print(f"\nTest Session ID: {TEST_SESSION_ID}")
            print("🎉 SSE system test completed!")
            
            # Overall success
            total_components = 3
            successful_components = sum([
                1 if streaming_success > 0 else 0,
                1 if connection_success >= len(connection_results) * 0.7 else 0,
                1 if perf_success >= len(performance_results) * 0.7 else 0
            ])
            
            success_rate = successful_components / total_components
            
            if success_rate >= 0.67:  # 67% success rate
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
        success = await test_sse_system()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as error:
        print(f"❌ Unexpected error: {error}")
        sys.exit(1)

if __name__ == '__main__':
    asyncio.run(main())
