#!/usr/bin/env python3
"""Test script for workflow nodes and enhanced logging"""
import asyncio
import os
import json
from datetime import datetime
from dotenv import load_dotenv
from pprint import pprint

# Add parent directory to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.workflow_service import WorkflowService
from app.core.database import supabase

# Load environment variables
load_dotenv()

async def test_llm_node():
    """Test LLM node execution"""
    print("\n=== Testing LLM Node ===")
    
    node = {
        "type": "llm",
        "data": {
            "model": "gpt-3.5-turbo",
            "temperature": 0.7,
            "system_prompt": "You are a helpful assistant.",
            "prompt_template": "Tell me a joke about {topic}"
        }
    }
    
    input_data = {
        "topic": "artificial intelligence"
    }
    
    try:
        result = await WorkflowService._execute_workflow_node(node, input_data)
        print("LLM Node Result:")
        pprint(result)
        return True
    except Exception as e:
        print(f"Error testing LLM node: {e}")
        return False

async def test_api_node():
    """Test API node execution"""
    print("\n=== Testing API Node ===")
    
    node = {
        "type": "api",
        "data": {
            "method": "GET",
            "url": "https://httpbin.org/get?test={test_param}",
            "headers": {
                "User-Agent": "WorkflowTester/1.0"
            }
        }
    }
    
    input_data = {
        "test_param": "hello_world"
    }
    
    try:
        result = await WorkflowService._execute_workflow_node(node, input_data)
        print("API Node Result:")
        pprint(result)
        return True
    except Exception as e:
        print(f"Error testing API node: {e}")
        return False

async def test_knowledge_node():
    """Test Knowledge Base node execution"""
    print("\n=== Testing Knowledge Base Node ===")
    
    node = {
        "type": "knowledge",
        "data": {
            "kb_id": "test_kb",
            "query": "What is the capital of France?",
            "limit": 3
        }
    }
    
    input_data = {}  # No input needed for this test
    
    try:
        result = await WorkflowService._execute_workflow_node(node, input_data)
        print("Knowledge Node Result:")
        pprint(result)
        return True
    except Exception as e:
        print(f"Error testing Knowledge node: {e}")
        return False

async def test_workflow_with_logging():
    """Test workflow execution with enhanced logging"""
    print("\n=== Testing Workflow with Enhanced Logging ===")
    
    # Create a test workflow
    workflow = {
        "id": "test_workflow_" + datetime.now().strftime("%Y%m%d%H%M%S"),
        "name": "Test Workflow with Logging",
        "description": "Workflow to test enhanced logging",
        "nodes": [
            {
                "id": "node1",
                "type": "llm",
                "data": {
                    "model": "gpt-3.5-turbo",
                    "prompt_template": "What are the top 3 benefits of {topic}?"
                },
                "position": {"x": 100, "y": 100}
            },
            {
                "id": "node2",
                "type": "api",
                "data": {
                    "method": "GET",
                    "url": "https://httpbin.org/anything?input={llm_output}",
                    "headers": {"Content-Type": "application/json"}
                },
                "position": {"x": 400, "y": 100}
            }
        ],
        "edges": [
            {
                "id": "edge1",
                "source": "node1",
                "target": "node2",
                "data": {
                    "sourceHandle": "output",
                    "targetHandle": "input"
                }
            }
        ]
    }
    
    # Create workflow run
    run_data = {
        "workflow_id": workflow["id"],
        "input_data": {"topic": "artificial intelligence"},
        "status": "running",
        "started_at": datetime.now().isoformat()
    }
    
    run_result = await supabase.table("workflow_runs").insert(run_data).execute()
    run_id = run_result.data[0]["id"]
    
    try:
        # Execute the workflow
        result = await WorkflowService._execute_workflow_fsm(
            workflow={"data": workflow},
            input_data={"topic": "artificial intelligence"},
            run_id=run_id
        )
        
        print("Workflow Execution Result:")
        pprint(result)
        
        # Get the execution logs
        logs_result = await supabase.table("workflow_steps") \
            .select("*") \
            .eq("workflow_run_id", run_id) \
            .order("started_at") \
            .execute()
        
        print("\nExecution Logs:")
        for log in logs_result.data:
            print(f"\nNode {log['node_id']} ({log['status']}):")
            print(f"Started: {log.get('started_at')}")
            print(f"Ended: {log.get('ended_at')}")
            
            if log.get('execution_context'):
                print("Context:")
                pprint(log['execution_context'])
        
        return True
        
    except Exception as e:
        print(f"Error testing workflow execution: {e}")
        return False
    finally:
        # Clean up
        await supabase.table("workflow_runs").delete().eq("id", run_id).execute()
        await supabase.table("workflow_steps").delete().eq("workflow_run_id", run_id).execute()

async def main():
    """Run all tests"""
    await WorkflowService.initialize()
    
    try:
        # Test individual node types
        await test_llm_node()
        await test_api_node()
        await test_knowledge_node()
        
        # Test workflow with enhanced logging
        await test_workflow_with_logging()
        
    finally:
        await WorkflowService.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
