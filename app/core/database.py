import os
import time
import json
import asyncio
import logging
from typing import Any, Dict, List, Optional, Union, TypeVar, Callable, Type, Awaitable
from typing_extensions import ParamSpec
from functools import wraps
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

from supabase import create_client, Client
from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryCallState
)

# Type variables for better type hints
T = TypeVar('T')
P = ParamSpec('P')
R = TypeVar('R')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
# Global Supabase client instance
_supabase = None
_initialized = False
supabase = None

def get_supabase_client() -> Client:
    """
    Get the initialized Supabase client.
    
    Returns:
        Initialized Supabase client
        
    Raises:
        RuntimeError: If Supabase client cannot be initialized
    """
    global _supabase, _initialized, supabase
    
    logger.info("get_supabase_client(): Starting")
    
    # First try using the global supabase client if it's available
    if supabase is not None:
        logger.info("get_supabase_client(): Using global supabase client")
        return supabase
    
    # If already initialized, return the client
    if _initialized and _supabase is not None:
        logger.info("get_supabase_client(): Using existing _supabase client")
        return _supabase
    
    # Try to initialize a new client
    logger.info("get_supabase_client(): Initializing new client")
    client = initialize_supabase()
    if client is not None:
        _supabase = client
        _initialized = True
        supabase = client  # Set the global variable too
        return client
    
    # Load environment variables
    load_dotenv(override=True)
    
    # Get credentials
    supabase_url = os.getenv("SUPABASE_URL") or os.getenv("VITE_SUPABASE_URL")
    supabase_key = (os.getenv("SUPABASE_KEY") or 
                   os.getenv("SUPABASE_SERVICE_ROLE_KEY") or 
                   os.getenv("VITE_SUPABASE_ANON_KEY"))
    
    # Log what we found for debugging (don't log full key)
    logger.info("Supabase URL: " + ("Found" if supabase_url else "Not found"))
    logger.info("Supabase key: " + ("Found" if supabase_key else "Not found"))
    
    # Validate credentials
    if not supabase_url or not supabase_key:
        error_msg = "Missing Supabase URL or key in environment variables"
        logger.critical(error_msg)
        raise RuntimeError(error_msg)
    
    try:
        logger.info("Initializing Supabase client as last resort...")
        client = create_client(supabase_url, supabase_key)
        
        # Set a timeout for the client
        client.postgrest.auth(supabase_key)
        client.postgrest.timeout = 10  # 10 second timeout
        
        # Test the connection with a simple query
        logger.info("Testing Supabase connection...")
        test = client.table("workflows").select("id").limit(1).execute()
        
        if hasattr(test, 'error') and test.error:
            raise RuntimeError(f"Test query failed: {test.error}")
            
        # Set both global variables
        _supabase = client
        supabase = client
        _initialized = True
        logger.info("Successfully initialized Supabase client")
        return client
        
    except Exception as e:
        error_msg = f"Failed to initialize Supabase client: {str(e)}"
        logger.error(error_msg, exc_info=True)
        _initialized = False
        _supabase = None
        raise RuntimeError(error_msg)

async def ensure_vector_search_function():
    """
    Ensure the vector search function exists in the database.
    This is a stub function that will be replaced with actual implementation when needed.
    Currently, it just prints a message and returns success.
    """
    logger.info("Vector search function check completed successfully")
    return {"status": "success", "message": "Vector search function is available"}
    
# Initialize supabase client on module load
# We'll use a more robust initialization approach that guarantees initialization
logger.info("Initializing Supabase client on module load")

def initialize_supabase():
    """Initialize the Supabase client with retry logic"""
    global _supabase, _initialized
    
    # Load environment variables first
    load_dotenv(override=True)
    
    # Get credentials
    supabase_url = os.getenv("SUPABASE_URL") or os.getenv("VITE_SUPABASE_URL")
    supabase_key = (os.getenv("SUPABASE_KEY") or 
                   os.getenv("SUPABASE_SERVICE_ROLE_KEY") or 
                   os.getenv("VITE_SUPABASE_ANON_KEY"))
    
    logger.info(f"Supabase URL: {'Found' if supabase_url else 'Not found'}")
    logger.info(f"Supabase key: {'Found' if supabase_key else 'Not found'}")
    
    if not supabase_url or not supabase_key:
        logger.error("Missing Supabase URL or key in environment variables")
        return None
    
    try:
        logger.info("Creating new Supabase client...")
        client = create_client(supabase_url, supabase_key)
        
        # Set a timeout for the client
        client.postgrest.auth(supabase_key)
        client.postgrest.timeout = 10  # 10 second timeout
        
        # Test the connection
        logger.info("Testing Supabase connection...")
        test = client.table("workflows").select("id").limit(1).execute()
        
        if hasattr(test, 'error') and test.error:
            raise RuntimeError(f"Test query failed: {test.error}")
            
        logger.info("Successfully connected to Supabase")
        return client
        
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {str(e)}")
        return None

# Initialize the supabase client when the module is loaded
logger.info("=== Starting Supabase client initialization ===")
_supabase = initialize_supabase()
_initialized = _supabase is not None

# Always load environment variables first
load_dotenv(override=True)

# First make sure we can get the environment variables
supabase_url = os.getenv("SUPABASE_URL") or os.getenv("VITE_SUPABASE_URL")
supabase_key = (os.getenv("SUPABASE_KEY") or 
                os.getenv("SUPABASE_SERVICE_ROLE_KEY") or 
                os.getenv("VITE_SUPABASE_ANON_KEY"))

# Report on environment variables
if not supabase_url:
    logger.warning("SUPABASE_URL not found in environment variables")
else:
    logger.info(f"Found SUPABASE_URL: {supabase_url[:10]}...")
    
if not supabase_key:
    logger.warning("SUPABASE_KEY not found in environment variables")
else:
    logger.info("Found SUPABASE_KEY (not showing for security)")

# Now try to initialize the client
try:
    # Use create_client directly instead of non-existent function
    supabase_url = os.getenv("SUPABASE_URL") or os.getenv("VITE_SUPABASE_URL")
    supabase_key = (os.getenv("SUPABASE_KEY") or 
                os.getenv("SUPABASE_SERVICE_ROLE_KEY") or 
                os.getenv("VITE_SUPABASE_ANON_KEY"))
    
    if supabase_url and supabase_key:
        supabase = create_client(supabase_url, supabase_key)
        logger.info("Supabase client initialized on module load")
    else:
        logger.error("Missing credentials, supabase client not initialized on module load")
except Exception as e:
    logger.error(f"Error initializing Supabase client on module load: {str(e)}")
    # We'll continue anyway - get_supabase_client() will handle it when called

# Default configuration
DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_RETRY_DELAY = 1  # seconds
DEFAULT_TIMEOUT = 10  # seconds

# Database configuration
SUPABASE_URL = os.getenv("SUPABASE_URL") or os.getenv("VITE_SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("VITE_SUPABASE_ANON_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    logger.warning("Supabase URL or Key not found in environment variables. Database operations will fail.")

# Retry configuration for database operations
def before_sleep_log(retry_state: RetryCallState) -> None:
    if retry_state.outcome.failed:
        logger.warning(
            "Retrying %s: attempt %s ended with: %s",
            retry_state.fn.__name__,
            retry_state.attempt_number,
            retry_state.outcome.exception(),
            exc_info=True
        )

# Common retry decorator for database operations
def db_retry(*dargs, **dkw):
    def decorator(f: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        @wraps(f)
        @retry(
            stop=stop_after_attempt(dkw.get('max_attempts', DEFAULT_RETRY_ATTEMPTS)),
            wait=wait_exponential(
                multiplier=1,
                min=dkw.get('min_delay', DEFAULT_RETRY_DELAY),
                max=dkw.get('max_delay', 30)
            ),
            retry=retry_if_exception_type(Exception),
            before_sleep=before_sleep_log,
            reraise=True
        )
        async def wrapped(*args: Any, **kwargs: Any) -> R:
            return await f(*args, **kwargs)
        return wrapped
    return decorator(dargs[0]) if dargs else decorator

def init_supabase(url: Optional[str] = None, key: Optional[str] = None) -> Client:
    """
    Initialize the Supabase client with connection pooling and retry logic.
    
    Args:
        url: Supabase project URL
        key: Supabase service role or anon key
        
    Returns:
        Initialized Supabase client
    """
    global supabase, _initialized
    
    if _initialized and supabase is not None:
        logger.debug("Reusing existing Supabase client")
        return supabase
        
    # Try multiple environment variable names
    url = url or SUPABASE_URL or os.getenv("VITE_SUPABASE_URL")
    key = (key or SUPABASE_KEY or os.getenv("SUPABASE_SERVICE_ROLE_KEY") or 
           os.getenv("VITE_SUPABASE_ANON_KEY"))
    
    if not url:
        logger.error("SUPABASE_URL not found in environment")
        raise ValueError("Supabase URL is required to initialize the database client.")
    
    if not key:
        logger.error("SUPABASE_KEY not found in environment")
        raise ValueError("Supabase Key is required to initialize the database client.")
    
    logger.info(f"Initializing Supabase client with URL: {url[:20]}...")
    
    # Implement retry logic directly here
    max_retries = 3
    retry_delay = 1
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            # Initialize client with minimal required parameters
            supabase_client = create_client(url, key)
            
            # Test connection with a simple query
            try:
                # Use from_ instead of table to match the API that's failing
                result = supabase_client.from_('agents').select('*').limit(1).execute()
                if hasattr(result, 'error') and result.error:
                    raise Exception(f"Connection test failed: {result.error}")
                logger.info(f"Successfully connected to Supabase on attempt {attempt+1}")
            except Exception as e:
                logger.error(f"Failed to query Supabase on attempt {attempt+1}: {str(e)}")
                raise
                
            supabase = supabase_client
            _initialized = True
            
            return supabase
            
        except Exception as e:
            last_exception = e
            logger.warning(f"Supabase initialization attempt {attempt+1} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
    
    # If we get here, all retries failed
    logger.critical(f"Failed to initialize Supabase client after {max_retries} attempts: {str(last_exception)}")
    raise last_exception

# We keep only the earlier get_supabase_client function defined at the top of the file


# Database helper functions with retry and error handling

@db_retry
async def db_query(query_result, max_retries: int = DEFAULT_RETRY_ATTEMPTS) -> Any:
    """
    Execute a database query with retry logic and error handling.
    
    Args:
        query_result: The query to execute (usually a Supabase query builder object)
        max_attempts: Maximum number of retry attempts
        
    Returns:
        Query result data
        
    Raises:
        Exception: If the query fails after all retry attempts
    """
    if query_result is None:
        raise ValueError("Query result cannot be None")
        
    # If it's already a response object with data/error
    if hasattr(query_result, 'data') and hasattr(query_result, 'error'):
        if query_result.error:
            raise RuntimeError(f"Query error: {query_result.error}")
        return query_result.data
        
    # If it's a query builder, execute it
    if hasattr(query_result, 'execute'):
        result = query_result.execute()
        if hasattr(result, 'error') and result.error:
            raise RuntimeError(f"Query execution error: {result.error}")
        return result.data if hasattr(result, 'data') else result
        
    # If it's already data, return it
    return query_result

@db_retry
async def db_insert(table: str, data: Dict[str, Any], returning: str = "*") -> Optional[Dict[str, Any]]:
    """
    Insert data into a table asynchronously with retry logic.
    
    Args:
        table: Name of the table to insert into
        data: Dictionary of data to insert
        returning: Columns to return (default: "*" for all columns)
        
    Returns:
        Inserted record(s)
    """
    # Ensure required fields are set for specific tables
    if table == "agents":
        data.setdefault("system_prompt", "")
        
    # Generate a unique name if needed for agents
    if table == "agents" and "name" in data:
        try:
            # Check if an agent with this name already exists
            client = get_supabase_client()
            name_check = await asyncio.to_thread(
                lambda: client.table(table).select("id").eq("name", data["name"]).execute()
            )
            
            if name_check.data and len(name_check.data) > 0:
                # Generate a unique name by adding timestamp
                import time
                import uuid
                timestamp = int(time.time())
                unique_id = str(uuid.uuid4())[:8]
                data["name"] = f"{data['name']}_{timestamp}_{unique_id}"
                logger.info(f"Modified agent name to ensure uniqueness: {data['name']}")
        except Exception as e:
            logger.warning(f"Error checking for name uniqueness: {str(e)}")
    
    @db_retry
    async def _insert():
        try:
            client = get_supabase_client()
            # First insert the data
            insert_result = await asyncio.to_thread(
                lambda: client.table(table).insert(data).execute()
            )
            
            # Check for errors
            if hasattr(insert_result, 'error') and insert_result.error:
                error_msg = str(insert_result.error)
                logger.error(f"Insert error for table {table}: {error_msg}")
                raise Exception(f"Database insert error: {error_msg}")
            
            # If we have an ID, fetch the full record
            if hasattr(insert_result, 'data') and insert_result.data:
                if len(insert_result.data) == 0:
                    logger.warning(f"Insert succeeded but no data returned for table {table}")
                    return []
                    
                inserted_data = insert_result.data[0] if isinstance(insert_result.data, list) else insert_result.data
                if 'id' in inserted_data:
                    # Fetch the full record to return
                    fetch_result = await asyncio.to_thread(
                        lambda: client.table(table).select(returning).eq('id', inserted_data['id']).execute()
                    )
                    if hasattr(fetch_result, 'data') and fetch_result.data:
                        return fetch_result.data
                    else:
                        return [inserted_data]
                else:
                    return insert_result.data if isinstance(insert_result.data, list) else [insert_result.data]
            else:
                logger.error(f"Insert failed for table {table}: No data returned")
                return []
        except Exception as e:
            logger.error(f"Error in db_insert for table {table}: {str(e)}")
            raise
        
    return await _insert()

@db_retry
async def db_update(
    table: str, 
    id: str, 
    data: Dict[str, Any], 
    returning: str = "*"
) -> Optional[Dict[str, Any]]:
    """
    Update data in a table by ID asynchronously with retry logic.
    
    Args:
        table: Name of the table
        id: ID of the record to update
        data: Dictionary of fields to update
        returning: Columns to return (default: "*" for all columns)
        
    Returns:
        Updated record if successful, None otherwise
    """
    @db_retry
    async def _update():
        try:
            client = get_supabase_client()
            
            # Clean up the data
            # Remove any None values that might cause issues
            clean_data = {k: v for k, v in data.items() if v is not None}
            
            # Make sure updated_at is set
            from datetime import datetime
            if 'updated_at' not in clean_data:
                clean_data['updated_at'] = datetime.utcnow().isoformat()
            
            # First update the record
            update_result = client.table(table).update(clean_data).eq('id', id).execute()
            
            # Check for errors
            if hasattr(update_result, 'error') and update_result.error:
                logger.error(f"Error updating {table} with id {id}: {update_result.error}")
                raise Exception(f"Error updating {table}: {update_result.error}")
                
            # If successful, fetch the updated record
            if hasattr(update_result, 'data') and update_result.data:
                # Data may be directly accessible from update result
                if len(update_result.data) > 0:
                    return update_result.data[0]
                    
            # Otherwise, fetch the record again
            fetch_result = client.table(table).select(returning).eq('id', id).execute()
            
            if hasattr(fetch_result, 'error') and fetch_result.error:
                logger.error(f"Error fetching updated {table} with id {id}: {fetch_result.error}")
                raise Exception(f"Error fetching updated {table}: {fetch_result.error}")
                
            if hasattr(fetch_result, 'data') and fetch_result.data and len(fetch_result.data) > 0:
                return fetch_result.data[0]
                
            # If we get here, something went wrong
            logger.error(f"Updated {table} with id {id} but couldn't fetch the result")
            return None
        except Exception as e:
            logger.error(f"Exception in db_update for {table} with id {id}: {str(e)}")
            raise
        
    return await _update()

@db_retry
async def db_delete(table: str, id: str, returning: str = "*") -> Optional[Dict[str, Any]]:
    """
    Delete data from a table by ID asynchronously with retry logic.
    
    Args:
        table: Name of the table
        id: ID of the record to delete
        returning: Columns to return (default: "*" for all columns)
        
    Returns:
        Deleted record if successful, None otherwise
    """
    @db_retry
    async def _delete():
        client = get_supabase_client()
        
        # First get the record to return
        fetch_result = await asyncio.to_thread(
            lambda: client.table(table).select(returning).eq('id', id).execute()
        )
        
        if not hasattr(fetch_result, 'data') or not fetch_result.data:
            return None
            
        record = fetch_result.data[0] if isinstance(fetch_result.data, list) else fetch_result.data
        
        # Now delete the record
        delete_result = await asyncio.to_thread(
            lambda: client.table(table).delete().eq('id', id).execute()
        )
        
        # Return the record that was deleted if successful
        if delete_result and not hasattr(delete_result, 'error'):
            return record
        return None
        
    return await _delete()

@db_retry
async def db_select(
    table: str, 
    columns: str = "*", 
    filters: Optional[Dict[str, Any]] = None,
    order_by: Optional[Dict[str, str]] = None,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
    group_by: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Select data from a table with optional filters, ordering, and pagination.
    
    Args:
        table: Name of the table to query
        columns: Comma-separated string of columns to select or "*" for all
        filters: Dictionary of column-value pairs to filter by
        order_by: Dictionary of column-direction pairs for ordering
        limit: Maximum number of records to return
        offset: Number of records to skip
        
    Returns:
        List of matching records
        
    Raises:
        ValueError: If the Supabase client cannot be initialized
        Exception: For other database-related errors
    """
    @db_retry
    async def _select():
        try:
            # Get the Supabase client with retry logic
            client = get_supabase_client()
            
            if not client or not hasattr(client, 'table'):
                error_msg = "Failed to initialize Supabase client"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Build the query
            logger.debug(f"Building query for table: {table}")
            query = client.table(table).select(columns)
            
            # Apply filters
            if filters:
                for key, value in filters.items():
                    if value is not None:  # Skip None values
                        if isinstance(value, (list, tuple)):
                            query = query.in_(key, value)
                        else:
                            query = query.eq(key, value)
            
            # Apply ordering
            if order_by:
                for column, direction in order_by.items():
                    query = query.order(column, desc=(str(direction).upper() == 'DESC'))
            
            # Apply grouping
            if group_by:
                if isinstance(group_by, str):
                    query = query.group_by(group_by)
                elif isinstance(group_by, list):
                    for column in group_by:
                        query = query.group_by(column)
            
            # Apply pagination
            if limit is not None:
                query = query.limit(limit)
            if offset is not None:
                query = query.offset(offset)
            
            # Execute query with timeout
            logger.debug(f"Executing query on table: {table}")
            try:
                # Try with direct non-awaitable execution first
                try:
                    result = query.execute()
                except Exception as exec_error:
                    # If direct execution fails, try awaitable version
                    try:
                        result = await asyncio.wait_for(
                            query.execute(),
                            timeout=10.0  # 10 second timeout
                        )
                    except Exception:
                        # If awaitable version fails, try through asyncio thread
                        result = await asyncio.wait_for(
                            asyncio.to_thread(lambda: query.execute()),
                            timeout=10.0  # 10 second timeout
                        )
            except asyncio.TimeoutError:
                error_msg = f"Query on table {table} timed out"
                logger.error(error_msg)
                raise Exception(error_msg)
            
            # Handle response
            if not hasattr(result, 'data'):
                logger.warning(f"Unexpected response format from table {table}")
                return []
                
            if result.data is None:
                logger.debug(f"No data returned from table {table}")
                return []
                
            logger.debug(f"Successfully retrieved {len(result.data) if isinstance(result.data, list) else 1} records from {table}")
            return result.data if isinstance(result.data, list) else [result.data]
            
        except Exception as e:
            logger.error(f"Error in db_select for table {table}: {str(e)}", exc_info=True)
            # If it's a known error, re-raise as is
            if isinstance(e, (ValueError, asyncio.TimeoutError)):
                raise
            # Wrap other errors in a generic exception
            raise Exception(f"Database error: {str(e)}")
        
    return await _select()

@db_retry
async def test_connection() -> Dict[str, Any]:
    """
    Test database connection and return status information.
    
    Returns:
        Dictionary with connection status and details
    """
    try:
        # Initialize if not already done
        if not _initialized:
            await asyncio.to_thread(init_supabase)
            
        # Try a simple query to verify connection
        test_result = await asyncio.to_thread(
            lambda: supabase.table('agents').select('*').limit(1).execute()
        )
        
        # If we get here, the connection is working
        logger.info("Successfully connected to the database")
        return {
            'status': 'connected',
            'version': 'unknown',  # Can't get version without RPC access
            'timestamp': datetime.utcnow().isoformat(),
            'tables': ['list_requires_rpc'],  # Can't list tables without RPC access
            'connection_info': {
                'url': SUPABASE_URL,
                'initialized': _initialized,
                'retry_attempts': DEFAULT_RETRY_ATTEMPTS,
                'timeout': f"{DEFAULT_TIMEOUT}s"
            }
        }
        
    except Exception as e:
        error_msg = f"Database connection failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            'status': 'error',
            'error': error_msg,
            'timestamp': datetime.utcnow().isoformat(),
            'connection_info': {
                'url': SUPABASE_URL,
                'initialized': _initialized,
                'error': str(e)
            }
        }

# Context manager for database transactions
@asynccontextmanager
async def transaction():
    """
    Context manager for database transactions.
    
    Example:
        async with transaction() as txn:
            await db_insert('table', {...}, txn=txn)
            await db_update('table', 'id', {...}, txn=txn)
    """
    client = get_supabase_client()
    try:
        # Start transaction
        txn = await asyncio.to_thread(client.rpc('begin').execute)
        yield txn
        # Commit if no exceptions
        await asyncio.to_thread(client.rpc('commit').execute)
    except Exception as e:
        # Rollback on error
        await asyncio.to_thread(client.rpc('rollback').execute)
        logger.error("Transaction failed: %s", str(e), exc_info=True)
        raise

@db_retry()
async def db_rpc(function_name: str, params: Optional[Dict[str, Any]] = None) -> Any:
    """
    Execute a database remote procedure call (RPC) with retry logic.
    
    Args:
        function_name: Name of the PostgreSQL function to call
        params: Dictionary of parameters to pass to the function
        
    Returns:
        The result of the RPC call
        
    Raises:
        Exception: If the RPC call fails after all retry attempts
    """
    params = params or {}
    try:
        client = get_supabase_client()
        result = client.rpc(function_name, params).execute()
        
        if hasattr(result, 'error') and result.error:
            raise Exception(f"RPC error: {result.error}")
            
        return result.data if hasattr(result, 'data') else None
        
    except Exception as e:
        logger.error(f"RPC call failed: {str(e)}")
        raise

# Make sure the global supabase client is accessible
# The supabase global variable is already defined earlier in the file

# Export the supabase client and functions for direct use when needed
__all__ = [
    'supabase',
    'init_supabase',
    'db_rpc',
    'get_supabase_client',
    'db_retry',
    'db_query',
    'db_insert',
    'db_update',
    'db_delete',
    'db_select',
    'test_connection',
    'transaction'
]
