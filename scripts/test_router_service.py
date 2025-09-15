#!/usr/bin/env python3
"""
Test script for the Router Service

This script tests the RouterService functionality including:
- Intent classification
- Agent selection
- Error handling
- Database operations
"""

import asyncio
import json
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.services.router_service import RouterService
from app.core.database import get_supabase_client, init_supabase
from app.core.models import RouterClassifyRequest

# Test data
TEST_RULES = [
    {
        "intent_name": "greeting",
        "keywords": ["hello", "hi", "hey", "greetings"],
        "agent_id": "00000000-0000-0000-0000-000000000001",
        "priority": 10,
        "confidence_threshold": 0.5,
        "description": "Handles greeting messages"
    },
    {
        "intent_name": "goodbye",
        "keywords": ["bye", "goodbye", "see you", "farewell"],
        "agent_id": "00000000-0000-0000-0000-000000000002",
        "priority": 20,
        "confidence_threshold": 0.6,
        "description": "Handles goodbye messages"
    }
]

TEST_AGENTS = [
    {
        "id": "00000000-0000-0000-0000-000000000001",
        "name": "Greeting Agent",
        "description": "Handles greeting messages",
        "is_active": True
    },
    {
        "id": "00000000-0000-0000-0000-000000000002",
        "name": "Farewell Agent",
        "description": "Handles farewell messages",
        "is_active": True
    }
]

async def setup_test_data():
    """Set up test data in the database"""
    supabase = get_supabase_client()
    
    # Clear existing test data
    await supabase.table('router_rules').delete().neq('id', '').execute()
    
    # Insert test agents if they don't exist
    for agent in TEST_AGENTS:
        await supabase.table('agents').upsert(agent).execute()
    
    # Insert test rules
    for rule in TEST_RULES:
        await supabase.table('router_rules').insert(rule).execute()
    
    print("✅ Test data set up complete")

async def test_intent_classification():
    """Test the intent classification functionality"""
    print("\n🧪 Testing intent classification...")
    
    test_cases = [
        ("Hello there!", "greeting"),
        ("Hi, how are you?", "greeting"),
        ("Goodbye for now", "goodbye"),
        ("See you later!", "goodbye"),
        ("This should not match anything", None)
    ]
    
    for query, expected_intent in test_cases:
        print(f"\n🔍 Testing query: {query}")
        try:
            result = await RouterService.classify_intent(
                input_text=query,
                options={"session_id": "test-session-123"}
            )
            
            print(f"   Result: {json.dumps(result, indent=4, default=str)}")
            
            if expected_intent is None:
                if result.get('intent') is None:
                    print(f"✅ PASS: Correctly identified no matching intent")
                else:
                    print(f"❌ FAIL: Expected no match but got '{result.get('intent')}'")
            else:
                if result.get('intent') == expected_intent:
                    print(f"✅ PASS: Correctly identified intent '{expected_intent}'")
                else:
                    print(f"❌ FAIL: Expected '{expected_intent}' but got '{result.get('intent')}'")
            
        except Exception as e:
            print(f"❌ Error during classification: {str(e)}")
            import traceback
            traceback.print_exc()

async def test_agent_selection():
    """Test agent selection based on intent"""
    print("\n🧪 Testing agent selection...")
    
    test_cases = [
        ("greeting", "Greeting Agent"),
        ("goodbye", "Farewell Agent"),
        ("nonexistent", None)
    ]
    
    for intent, expected_agent in test_cases:
        print(f"\n🔍 Testing intent: {intent}")
        try:
            agent = await RouterService.select_agent(intent)
            
            if expected_agent is None:
                if agent is None:
                    print("✅ PASS: Correctly returned no agent")
                else:
                    print(f"❌ FAIL: Expected no agent but got {agent.get('agent_name')}")
            else:
                if agent and agent.get('agent_name') == expected_agent:
                    print(f"✅ PASS: Correctly selected agent '{expected_agent}'")
                else:
                    actual = agent.get('agent_name') if agent else 'None'
                    print(f"❌ FAIL: Expected agent '{expected_agent}' but got '{actual}'")
            
            if agent:
                print(f"   Agent details: {json.dumps(agent, indent=4, default=str)}")
                
        except Exception as e:
            print(f"❌ Error during agent selection: {str(e)}")
            import traceback
            traceback.print_exc()

async def run_tests():
    """Run all tests"""
    print("🚀 Starting Router Service Tests")
    print("=" * 50)
    
    # Initialize database connection
    await init_supabase()
    
    # Set up test data
    await setup_test_data()
    
    # Run tests
    await test_intent_classification()
    await test_agent_selection()
    
    print("\n" + "=" * 50)
    print("✅ All tests completed")

if __name__ == "__main__":
    asyncio.run(run_tests())
