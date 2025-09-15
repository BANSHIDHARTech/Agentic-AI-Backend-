"""
Tool Service

Manages tool registration, execution, and lifecycle.
Provides built-in tools for common operations and supports custom tool execution.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from ..core.database import get_supabase_client
from ..core.models import ToolModel
from .knowledge_service import KnowledgeService

logger = logging.getLogger(__name__)


class ToolService:
    """Service for managing tools and tool execution"""
    
    @staticmethod
    async def create_tool(tool_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Create a new tool
        
        Args:
            tool_data: Tool data to create
            
        Returns:
            Created tool data or None if creation failed
        """
        try:
            tool_name = tool_data.get('name', 'unnamed_tool')
            
            # Convert to dict if it's a Pydantic model
            if hasattr(tool_data, 'dict'):
                validated_data = tool_data.dict()
            else:
                # Validate using model if possible
                try:
                    tool_model = ToolModel(**tool_data)
                    validated_data = tool_model.dict()
                except Exception as e:
                    logger.warning(f"Tool validation warning for '{tool_name}': {e}")
                    validated_data = tool_data  # Fallback to raw data
            
            # Ensure we have required fields
            if 'name' not in validated_data or not validated_data['name']:
                raise ValueError("Tool name is required")
            
            tool_name = validated_data['name']
                
            # Add timestamps
            now = datetime.utcnow().isoformat()
            validated_data['created_at'] = now
            validated_data['updated_at'] = now
            
            # Ensure is_active is set
            if 'is_active' not in validated_data:
                validated_data['is_active'] = True
            
            logger.debug(f"Creating tool: {tool_name}")
            supabase = get_supabase_client()
            response = supabase.table('tools').insert(validated_data).execute()
            
            # Check for successful insertion
            if not hasattr(response, 'data') or not response.data:
                error_msg = getattr(response, 'error', 'Unknown error')
                # Check for duplicate key error
                if hasattr(error_msg, 'code') and error_msg.code == '23505':
                    logger.info(f"ℹ️  Tool '{tool_name}' already exists (duplicate key)")
                    return None
                raise Exception(f"Failed to create tool '{tool_name}': {error_msg}")
            
            created_tool = response.data[0] if response.data else validated_data
            logger.info(f"✅ Successfully created tool: {tool_name}")
            return created_tool
            
        except Exception as error:
            logger.error(f"❌ Failed to create tool '{tool_name}': {str(error)}")
            return None
    
    @staticmethod
    def get_tool(tool_id: str) -> Dict[str, Any]:
        """
        Get a tool by ID
        
        Args:
            tool_id: ID of tool to retrieve
            
        Returns:
            Tool record
        """
        try:
            supabase = get_supabase_client()
            response = supabase.table('tools').select('*').eq('id', tool_id).execute()
            
            if response.data is None or len(response.data) == 0:
                raise Exception(f"Tool not found: {tool_id}")
            
            return response.data[0]
            
        except Exception as error:
            logger.error(f"❌ Failed to get tool {tool_id}: {error}")
            raise error
    
    @staticmethod
    def get_tool_by_name(name: str) -> Optional[Dict[str, Any]]:
        """
        Get a tool by name
        
        Args:
            name: Name of tool to retrieve
            
        Returns:
            Tool record or None if not found
        """
        try:
            supabase = get_supabase_client()
            # Make the Supabase call synchronously since supabase-py is sync
            response = supabase.table('tools').select('*').eq('name', name).eq('is_active', True).execute()
            
            if not hasattr(response, 'data') or not response.data:
                logger.debug(f"ℹ️  No active tool found with name: {name}")
                return None
                
            # Return the first matching tool
            tool = response.data[0] if response.data else None
            if tool:
                # Ensure all date fields are serializable
                return json.loads(json.dumps(tool, default=str))
            return None
            
        except Exception as error:
            logger.error(f"❌ Failed to get tool by name '{name}': {str(error)}")
            return None
    
    @staticmethod
    def get_all_tools() -> List[Dict[str, Any]]:
        """
        Get all active tools
        
        Returns:
            List of active tool records
        """
        try:
            supabase = get_supabase_client()
            response = supabase.table('tools').select('*').eq(
                'is_active', True
            ).order('created_at', desc=True).execute()
            
            if response.data is None:
                logger.warning("Failed to get tools from database")
                return []
            
            return response.data
            
        except Exception as error:
            logger.error(f"❌ Failed to get all tools: {error}")
            raise error
    
    @staticmethod
    async def update_tool(tool_id: str, tool_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update a tool
        
        Args:
            tool_id: ID of tool to update
            tool_data: Updated tool data
            
        Returns:
            Updated tool record
        """
        try:
            # Validate using model if available
            try:
                validated_data = ToolModel.validate(tool_data)
            except:
                validated_data = tool_data  # Use raw data if validation fails
            
            # Add updated timestamp
            validated_data['updated_at'] = datetime.utcnow().isoformat()
            
            supabase = get_supabase_client()
            response = supabase.table('tools').update(validated_data).eq('id', tool_id).execute()
            
            if response.data is None or len(response.data) == 0:
                raise Exception(f"Failed to update tool: {getattr(response, 'error', 'Tool not found')}")
            
            return response.data[0]
            
        except Exception as error:
            logger.error(f"❌ Failed to update tool {tool_id}: {error}")
            raise error
    
    @staticmethod
    async def delete_tool(tool_id: str) -> bool:
        """
        Delete a tool
        
        Args:
            tool_id: ID of tool to delete
            
        Returns:
            True if successful
        """
        try:
            supabase = get_supabase_client()
            response = supabase.table('tools').delete().eq('id', tool_id).execute()
            
            # Check if there was an error
            if hasattr(response, 'error') and response.error:
                raise Exception(f"Failed to delete tool: {response.error}")
            
            return True
            
        except Exception as error:
            logger.error(f"❌ Failed to delete tool {tool_id}: {error}")
            raise error
    
    @staticmethod
    async def execute_tool(tool_id: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a tool with given parameters
        
        Args:
            tool_id: ID of tool to execute
            parameters: Parameters to pass to the tool
            
        Returns:
            Tool execution result
        """
        if parameters is None:
            parameters = {}
            
        try:
            # Get the tool
            tool = ToolService.get_tool(tool_id)
            
            # Log tool execution start
            supabase = get_supabase_client()
            supabase.table('logs').insert({
                'event_type': 'tool_execution_start',
                'details': {
                    'tool_id': tool_id,
                    'tool_name': tool['name'],
                    'parameters': parameters,
                    'timestamp': datetime.utcnow().isoformat()
                },
                'created_at': datetime.utcnow().isoformat()
            }).execute()
            
            # Execute the tool function
            result = await ToolService._simulate_tool_execution(tool, parameters)
            
            # Log tool execution success
            supabase.table('logs').insert({
                'event_type': 'tool_execution_success',
                'details': {
                    'tool_id': tool_id,
                    'tool_name': tool['name'],
                    'parameters': parameters,
                    'result': result,
                    'timestamp': datetime.utcnow().isoformat()
                },
                'created_at': datetime.utcnow().isoformat()
            }).execute()
            
            return result
            
        except Exception as error:
            logger.error(f"❌ Tool execution failed for {tool_id}: {error}")
            
            # Log tool execution error
            try:
                supabase = get_supabase_client()
                supabase.table('logs').insert({
                    'event_type': 'tool_execution_error',
                    'details': {
                        'tool_id': tool_id,
                        'tool_name': tool.get('name', 'unknown') if 'tool' in locals() else 'unknown',
                        'parameters': parameters,
                        'error': str(error),
                        'timestamp': datetime.utcnow().isoformat()
                    },
                    'created_at': datetime.utcnow().isoformat()
                }).execute()
            except Exception as log_error:
                logger.error(f"Failed to log tool execution error: {log_error}")
            
            raise error
    
    @staticmethod
    async def _simulate_tool_execution(tool: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate tool execution based on tool name
        
        Args:
            tool: Tool record
            parameters: Tool parameters
            
        Returns:
            Simulated execution result
        """
        tool_name = tool['name'].lower()
        
        if tool_name == 'get_postpaid_balance':
            return {
                'customer_id': parameters.get('customer_id') or parameters.get('user_id', 'unknown'),
                'balance': 85.50,
                'currency': 'USD',
                'last_updated': datetime.utcnow().isoformat(),
                'account_status': 'active',
                'due_date': (datetime.utcnow() + timedelta(days=30)).isoformat()
            }
        
        elif tool_name == 'authenticate_user':
            return {
                'user_id': parameters.get('user_id', 'user_123'),
                'authenticated': True,
                'session_token': f'mock_token_{int(datetime.now().timestamp() * 1000)}',
                'expires_at': (datetime.utcnow() + timedelta(hours=24)).isoformat(),
                'user_role': 'customer'
            }
        
        elif tool_name == 'send_notification':
            return {
                'notification_id': f'notif_{int(datetime.now().timestamp() * 1000)}',
                'recipient': parameters.get('recipient') or parameters.get('user_id'),
                'message': parameters.get('message', 'Notification sent'),
                'status': 'sent',
                'sent_at': datetime.utcnow().isoformat()
            }
        
        elif tool_name == 'create_ticket':
            return {
                'ticket_id': f'ticket_{int(datetime.now().timestamp() * 1000)}',
                'customer_id': parameters.get('customer_id') or parameters.get('user_id'),
                'subject': parameters.get('subject', 'Support Request'),
                'status': 'open',
                'priority': parameters.get('priority', 'medium'),
                'created_at': datetime.utcnow().isoformat()
            }
        
        elif tool_name == 'knowledge_search':
            return await KnowledgeService.knowledge_search_tool(parameters)
        
        else:
            # Generic tool execution
            return {
                'tool_name': tool['name'],
                'executed': True,
                'parameters': parameters,
                'result': f"Tool {tool['name']} executed successfully",
                'timestamp': datetime.utcnow().isoformat()
            }
    
    @staticmethod
    async def register_builtin_tools() -> List[Dict[str, Any]]:
        """
        Register built-in tools in the database
        
        Returns:
            List of created or existing tool records
            
        Note:
            - Checks if tool already exists before creating
            - Skips creation if tool with same name exists
            - Returns list of all built-in tools (both existing and newly created)
        """
        builtin_tools = [
            {
                'name': 'get_postpaid_balance',
                'description': 'Retrieves the postpaid balance for a customer account',
                'function_code': 'function get_postpaid_balance(customer_id) { return { customer_id, balance: 85.50, currency: "USD" }; }',
                'parameters': {
                    'customer_id': {
                        'type': 'string',
                        'required': True,
                        'description': 'Customer ID to retrieve balance for'
                    }
                },
                'is_active': True,
                'created_at': datetime.utcnow().isoformat()
            },
            {
                'name': 'authenticate_user',
                'description': 'Authenticates user credentials and returns session information',
                'function_code': 'function authenticate_user(user_id, password) { return { user_id, authenticated: true, session_token: "token_123" }; }',
                'parameters': {
                    'user_id': {
                        'type': 'string',
                        'required': True,
                        'description': 'User ID to authenticate'
                    },
                    'password': {
                        'type': 'string',
                        'required': False,
                        'description': 'User password (optional for demo)'
                    }
                },
                'is_active': True,
                'created_at': datetime.utcnow().isoformat()
            },
            {
                'name': 'send_notification',
                'description': 'Sends a notification to a user',
                'function_code': 'function send_notification(recipient, message) { return { notification_id: "notif_123", status: "sent" }; }',
                'parameters': {
                    'recipient': {
                        'type': 'string',
                        'required': True,
                        'description': 'Recipient user ID'
                    },
                    'message': {
                        'type': 'string',
                        'required': True,
                        'description': 'Notification message'
                    }
                },
                'is_active': True,
                'created_at': datetime.utcnow().isoformat()
            },
            {
                'name': 'create_ticket',
                'description': 'Creates a support ticket for customer issues',
                'function_code': 'function create_ticket(customer_id, subject, priority) { return { ticket_id: "ticket_123", status: "open" }; }',
                'parameters': {
                    'customer_id': {
                        'type': 'string',
                        'required': True,
                        'description': 'Customer ID creating the ticket'
                    },
                    'subject': {
                        'type': 'string',
                        'required': True,
                        'description': 'Ticket subject/title'
                    },
                    'priority': {
                        'type': 'string',
                        'required': False,
                        'description': 'Ticket priority (low, medium, high)'
                    }
                },
                'is_active': True,
                'created_at': datetime.utcnow().isoformat()
            },
            {
                'name': 'knowledge_search',
                'description': 'Searches the knowledge base for relevant information using similarity search',
                'function_code': 'async function knowledge_search(query, limit, similarity_threshold, session_id) { return await KnowledgeService.knowledgeSearchTool({ query, limit, similarity_threshold, session_id }); }',
                'parameters': {
                    'query': {
                        'type': 'string',
                        'required': True,
                        'description': 'Search query to find relevant information'
                    },
                    'limit': {
                        'type': 'number',
                        'required': False,
                        'description': 'Maximum number of results to return (default: 3)'
                    },
                    'similarity_threshold': {
                        'type': 'number',
                        'required': False,
                        'description': 'Minimum similarity score (default: 0.7)'
                    },
                    'session_id': {
                        'type': 'string',
                        'required': False,
                        'description': 'Session identifier for tracking'
                    }
                },
                'is_active': True,
                'created_at': datetime.utcnow().isoformat()
            }
        ]
        
        results = []
        existing_tools = 0
        
        for tool in builtin_tools:
            tool_name = tool.get('name', 'unnamed_tool')
            try:
                # Check if tool already exists
                existing = ToolService.get_tool_by_name(tool_name)
                if existing:
                    logger.info(f"ℹ️  Tool '{tool_name}' already exists, skipping")
                    existing_tools += 1
                    results.append(existing)
                    continue
                
                # Tool doesn't exist, create it
                logger.debug(f"Creating new tool: {tool_name}")
                created = await ToolService.create_tool(tool)
                
                if created:
                    results.append(created)
                    logger.info(f"✅ Successfully registered tool: {tool_name}")
                else:
                    # If creation failed but didn't raise an exception, it might be a duplicate
                    existing = ToolService.get_tool_by_name(tool_name)
                    if existing:
                        logger.info(f"ℹ️  Tool '{tool_name}' was created by another process")
                        existing_tools += 1
                        results.append(existing)
                    else:
                        logger.error(f"❌ Failed to create tool '{tool_name}': Unknown error")
                
            except Exception as error:
                # Handle specific database errors
                error_msg = str(error).lower()
                if 'duplicate' in error_msg or 'already exists' in error_msg:
                    logger.info(f"ℹ️  Tool '{tool_name}' already exists, skipping")
                    existing_tools += 1
                    # Try to get the existing tool
                    existing = ToolService.get_tool_by_name(tool_name)
                    if existing:
                        results.append(existing)
                else:
                    logger.error(f"❌ Failed to register tool '{tool_name}': {error}")
        
        # Log summary
        new_tools = len(results) - existing_tools
        logger.info(f"✅ Tool registration complete - {new_tools} new tools, "
                  f"{existing_tools} existing tools, {len(builtin_tools)} total")
        
        return results
    
    @staticmethod
    async def get_tool_usage_stats() -> Dict[str, Any]:
        """
        Get tool usage statistics
        
        Returns:
            Tool usage analytics
        """
        try:
            supabase = get_supabase_client()
            
            # Get tool execution logs from last 30 days
            thirty_days_ago = (datetime.utcnow() - timedelta(days=30)).isoformat()
            
            response = supabase.table('logs').select(
                'event_type, details, created_at'
            ).in_(
                'event_type', ['tool_execution_start', 'tool_execution_success', 'tool_execution_error']
            ).gte('created_at', thirty_days_ago).execute()
            
            if response.data is None:
                logs = []
            else:
                logs = response.data
            
            # Analyze tool usage
            tool_stats = {}
            total_executions = 0
            successful_executions = 0
            failed_executions = 0
            
            for log in logs:
                event_type = log.get('event_type')
                details = log.get('details', {})
                tool_name = details.get('tool_name', 'unknown')
                
                if tool_name not in tool_stats:
                    tool_stats[tool_name] = {
                        'executions': 0,
                        'successes': 0,
                        'errors': 0,
                        'success_rate': 0
                    }
                
                if event_type == 'tool_execution_start':
                    tool_stats[tool_name]['executions'] += 1
                    total_executions += 1
                elif event_type == 'tool_execution_success':
                    tool_stats[tool_name]['successes'] += 1
                    successful_executions += 1
                elif event_type == 'tool_execution_error':
                    tool_stats[tool_name]['errors'] += 1
                    failed_executions += 1
            
            # Calculate success rates
            for tool_name, stats in tool_stats.items():
                if stats['executions'] > 0:
                    stats['success_rate'] = stats['successes'] / stats['executions']
            
            # Get most used tools
            most_used = sorted(
                tool_stats.items(),
                key=lambda x: x[1]['executions'],
                reverse=True
            )[:10]
            
            return {
                'total_executions': total_executions,
                'successful_executions': successful_executions,
                'failed_executions': failed_executions,
                'overall_success_rate': successful_executions / total_executions if total_executions > 0 else 0,
                'tool_stats': tool_stats,
                'most_used_tools': most_used,
                'period_days': 30,
                'generated_at': datetime.utcnow().isoformat()
            }
            
        except Exception as error:
            logger.error(f"❌ Failed to get tool usage stats: {error}")
            return {
                'total_executions': 0,
                'successful_executions': 0,
                'failed_executions': 0,
                'overall_success_rate': 0,
                'tool_stats': {},
                'most_used_tools': [],
                'period_days': 30,
                'generated_at': datetime.utcnow().isoformat(),
                'error': str(error)
            }
