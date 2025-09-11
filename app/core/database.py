import os
from typing import Any, Dict, List, Optional
from supabase import create_client, Client
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Supabase configuration
supabase_url = os.getenv("VITE_SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("VITE_SUPABASE_ANON_KEY")

if not supabase_url or not supabase_key:
    raise ValueError("Missing Supabase configuration. Please set VITE_SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in .env file")

# Create Supabase client instance
supabase: Optional[Client] = None

def get_supabase_client() -> Client:
    """Get or create a Supabase client instance"""
    global supabase
    if supabase is None:
        if not supabase_url or not supabase_key:
            raise ValueError("Supabase URL and key must be set in environment variables")
        supabase = create_client(supabase_url, supabase_key)
    return supabase

# Initialize the client when module is imported
supabase = get_supabase_client()

# Database helper functions
async def db_query(query_result) -> Any:
    """Execute a database query and handle errors"""
    try:
        if hasattr(query_result, 'data') and hasattr(query_result, 'error'):
            if query_result.error:
                raise Exception(query_result.error)
            return query_result.data
        return query_result
    except Exception as error:
        print(f"Database query error: {error}")
        raise error

async def db_insert(table: str, data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Insert data into a table"""
    result = supabase.from_(table).insert(data).execute()
    if result.error:
        raise Exception(result.error)
    return result.data

async def db_update(table: str, id: str, data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Update data in a table by ID"""
    result = supabase.from_(table).update(data).eq('id', id).execute()
    if result.error:
        raise Exception(result.error)
    return result.data

async def db_delete(table: str, id: str) -> bool:
    """Delete data from a table by ID"""
    result = supabase.from_(table).delete().eq('id', id).execute()
    if result.error:
        raise Exception(result.error)
    return True

async def db_select(table: str, columns: str = "*", filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Select data from a table with optional filters"""
    query = supabase.from_(table).select(columns)
    
    if filters:
        for key, value in filters.items():
            query = query.eq(key, value)
    
    result = query.execute()
    if result.error:
        raise Exception(result.error)
    return result.data

async def test_connection() -> bool:
    """Test database connection"""
    try:
        # Try a simple query to test the connection
        result = supabase.from_('agents').select('id', count='exact').limit(1).execute()
        # If we get here, the connection was successful
        return True
    except Exception as error:
        print(f"Database connection failed: {error}")
        return False

# Export the supabase client for direct use when needed
__all__ = ['supabase', 'db_query', 'db_insert', 'db_update', 'db_delete', 'db_select', 'test_connection']
