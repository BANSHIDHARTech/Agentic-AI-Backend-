#!/usr/bin/env python3
"""
Test script for database connection and basic CRUD operations.
"""
import asyncio
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.database import (
    init_supabase,
    get_supabase_client,
    test_connection,
    db_insert,
    db_select,
    db_update,
    db_delete,
    transaction
)

# Use the existing agents table for testing
TEST_TABLE = "agents"

async def setup_test_environment():
    """Initialize the test environment and verify the test table exists."""
    # Initialize Supabase client
    client = init_supabase()
    
    # Test connection
    conn_status = await test_connection()
    print("\n=== Connection Test ===")
    print(f"Status: {conn_status['status']}")
    if conn_status['status'] == 'error':
        print(f"Error: {conn_status['error']}")
        return False
    
    print(f"Database Version: {conn_status.get('version', 'unknown')}")
    print(f"Using table: {TEST_TABLE}")
    
    # Verify the test table exists and is accessible
    try:
        result = await asyncio.to_thread(
            lambda: client.table(TEST_TABLE).select("id").limit(1).execute()
        )
        print(f"Successfully accessed table: {TEST_TABLE}")
        return True
    except Exception as e:
        print(f"Error accessing table {TEST_TABLE}: {e}")
        print(f"\nPlease ensure the table '{TEST_TABLE}' exists and is accessible with the current credentials.")
        return False

async def test_crud_operations():
    """Test CRUD operations on the test table."""
    # Test data for the agents table
    test_data = {
        "name": f"Test Agent {int(time.time())}",  # Unique name for each test run
        "description": "A test agent for database operations",
        "system_prompt": "You are a helpful assistant.",
        "is_active": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat()
    }
    
    print("\n=== Testing CRUD Operations ===")
    
    # Test INSERT
    print("\n[TEST] INSERT operation...")
    try:
        inserted = await db_insert(TEST_TABLE, test_data)
        if not inserted or 'id' not in inserted:
            print("[ERROR] INSERT failed: No ID returned")
            return False
        
        item_id = inserted['id']
        print(f"✅ INSERT successful. ID: {item_id}")
        
        # Test SELECT
        print("\n[TEST] SELECT operation...")
        selected = await db_select(
            TEST_TABLE, 
            filters={"id": item_id}
        )
        
        if not selected or len(selected) == 0:
            print("[ERROR] SELECT failed: No results found")
            return False
            
        print(f"✅ SELECT successful. Found {len(selected)} records")
        
        # Test UPDATE
        print("\n[TEST] UPDATE operation...")
        update_data = {
            "name": f"Updated Test Agent {int(time.time())}",
            "is_active": False,
            "description": "Updated description for test agent",
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        
        # Try to get database version using a simple query instead of RPC
        try:
            # This is a standard PostgreSQL system view
            version_result = await asyncio.to_thread(
                lambda: get_supabase_client().table('pg_settings')
                              .select('setting')
                              .eq('name', 'server_version')
                              .single()
                              .execute()
            )
            db_version = version_result.data.get('setting', 'unknown') if hasattr(version_result, 'data') else 'unknown'
        except Exception as e:
            db_version = 'unknown (error: ' + str(e) + ')'
        
        # Get list of tables from information_schema
        try:
            tables_result = await asyncio.to_thread(
                lambda: get_supabase_client().table('information_schema.tables')
                              .select('table_name')
                              .eq('table_schema', 'public')
                              .execute()
            )
            tables = [t['table_name'] for t in tables_result.data] if hasattr(tables_result, 'data') else []
        except Exception as e:
            tables = ['error: ' + str(e)]
        
        logger.info("Successfully connected to the database")
        
        updated = await db_update(
            TEST_TABLE, 
            id=item_id, 
            data=update_data
        )
        
        if not updated or updated.get('name') != update_data['name']:
            print("❌ UPDATE failed: Record not updated correctly")
            return False
            
        print(f"✅ UPDATE successful. New name: {updated['name']}")
        
        # Skip transaction test as Supabase REST API doesn't support direct transaction management
        print("\n[INFO] Skipping transaction test - not supported with Supabase REST API")
        print("✅ All CRUD operations completed successfully")
        
        # Test DELETE
        print("\n[TEST] DELETE operation...")
        deleted = await db_delete(TEST_TABLE, item_id)
        
        if not deleted or deleted.get('id') != item_id:
            print("❌ DELETE failed: Record not deleted or invalid response")
            return False
            
        print(f"✅ DELETE successful. Deleted ID: {deleted['id']}")
        
        # Verify deletion
        verify = await db_select(
            TEST_TABLE,
            filters={"id": item_id}
        )
        
        if not verify or len(verify) == 0:
            print("✅ Deletion verified")
        else:
            print("❌ Deletion verification failed: Record still exists")
            return False
            
        return True
        
    except Exception as e:
        error_msg = f"Error during CRUD operations: {str(e)}"
        logger.error(error_msg, exc_info=True)
        print(f"[ERROR] {error_msg}")
        return False
    finally:
        # Clean up test data
        try:
            if 'item_id' in locals():
                await db_delete(TEST_TABLE, item_id)
                logger.info(f"Cleaned up test record with ID: {item_id}")
        except Exception as e:
            logger.warning(f"Failed to clean up test data: {str(e)}")

async def main():
    """
    Test database connection and basic operations.
    """
    logger.info("Starting database tests")
    print("=== Starting Database Tests ===\n")
    
    # Setup test environment
    if not await setup_test_environment():
        print("\n[ERROR] Test environment setup failed")
        return 1
    
    # Run CRUD tests
    if await test_crud_operations():
        print("\n[SUCCESS] All database tests passed!")
        return 0
    else:
        print("\n[ERROR] Some database tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
