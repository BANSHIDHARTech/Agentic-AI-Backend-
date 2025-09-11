"""
Built-in Tool Definitions for AI Agent Flow

This module provides built-in tool definitions and registration functionality.
These tools simulate common telecom operations like balance retrieval, authentication,
notifications, and support ticket creation.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


class BuiltinTool:
    """Base class for built-in tools"""
    
    def __init__(self, name: str, description: str, parameters: Dict[str, Any]):
        self.name = name
        self.description = description
        self.parameters = parameters
    
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool with given parameters"""
        raise NotImplementedError("Subclasses must implement execute method")
    
    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """Validate parameters against tool requirements"""
        for param_name, param_config in self.parameters.items():
            if param_config.get('required', False) and param_name not in params:
                raise ValueError(f"Required parameter '{param_name}' is missing")
        return True


class GetPostpaidBalanceTool(BuiltinTool):
    """Tool for retrieving postpaid balance for a customer account"""
    
    def __init__(self):
        super().__init__(
            name='get_postpaid_balance',
            description='Retrieves the postpaid balance for a customer account',
            parameters={
                'customer_id': {
                    'type': 'string',
                    'required': True,
                    'description': 'Customer ID to retrieve balance for'
                }
            }
        )
    
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute balance retrieval"""
        self.validate_parameters(params)
        
        # Simulate balance retrieval
        return {
            'customer_id': params['customer_id'],
            'balance': 85.50,
            'currency': 'USD',
            'last_updated': datetime.utcnow().isoformat(),
            'account_status': 'active',
            'due_date': (datetime.utcnow() + timedelta(days=30)).isoformat()
        }


class AuthenticateUserTool(BuiltinTool):
    """Tool for user authentication and session management"""
    
    def __init__(self):
        super().__init__(
            name='authenticate_user',
            description='Authenticates user credentials and returns session information',
            parameters={
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
            }
        )
    
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute user authentication"""
        self.validate_parameters(params)
        
        # Simulate authentication
        return {
            'user_id': params['user_id'],
            'authenticated': True,
            'session_token': f'mock_token_{int(datetime.now().timestamp() * 1000)}',
            'expires_at': (datetime.utcnow() + timedelta(hours=24)).isoformat(),
            'user_role': 'customer'
        }


class SendNotificationTool(BuiltinTool):
    """Tool for sending notifications to users"""
    
    def __init__(self):
        super().__init__(
            name='send_notification',
            description='Sends a notification to a user',
            parameters={
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
            }
        )
    
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute notification sending"""
        self.validate_parameters(params)
        
        # Simulate notification sending
        return {
            'notification_id': f'notif_{int(datetime.now().timestamp() * 1000)}',
            'recipient': params['recipient'],
            'message': params['message'],
            'status': 'sent',
            'sent_at': datetime.utcnow().isoformat()
        }


class CreateTicketTool(BuiltinTool):
    """Tool for creating support tickets"""
    
    def __init__(self):
        super().__init__(
            name='create_ticket',
            description='Creates a support ticket',
            parameters={
                'customer_id': {
                    'type': 'string',
                    'required': True,
                    'description': 'Customer ID'
                },
                'subject': {
                    'type': 'string',
                    'required': True,
                    'description': 'Ticket subject'
                },
                'priority': {
                    'type': 'string',
                    'required': False,
                    'description': 'Ticket priority (low, medium, high)'
                }
            }
        )
    
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute ticket creation"""
        self.validate_parameters(params)
        
        # Simulate ticket creation
        return {
            'ticket_id': f'ticket_{int(datetime.now().timestamp() * 1000)}',
            'customer_id': params['customer_id'],
            'subject': params['subject'],
            'status': 'open',
            'priority': params.get('priority', 'medium'),
            'created_at': datetime.utcnow().isoformat()
        }


# Registry of all built-in tools
BUILTIN_TOOLS: Dict[str, BuiltinTool] = {
    'get_postpaid_balance': GetPostpaidBalanceTool(),
    'authenticate_user': AuthenticateUserTool(),
    'send_notification': SendNotificationTool(),
    'create_ticket': CreateTicketTool()
}


async def register_builtin_tools() -> List[Dict[str, Any]]:
    """
    Register all built-in tools with the ToolService
    
    Returns:
        List of created tool records
    """
    try:
        from ..app.services.tool_service import ToolService
    except ImportError:
        # Fallback import path
        from app.services.tool_service import ToolService
    
    results = []
    
    for tool_name, tool_instance in BUILTIN_TOOLS.items():
        try:
            # Check if tool already exists
            try:
                existing = await ToolService.get_tool_by_name(tool_name)
                logger.info(f"Tool {tool_name} already exists, skipping registration")
                continue
            except Exception:
                # Tool doesn't exist, proceed with creation
                pass
            
            # Create sample execution result for function_code
            sample_params = {}
            for param_name, param_config in tool_instance.parameters.items():
                if param_config.get('required', False):
                    sample_params[param_name] = f"sample_{param_name}"
            
            sample_result = await tool_instance.execute(sample_params)
            
            # Create tool record
            tool_data = {
                'name': tool_instance.name,
                'description': tool_instance.description,
                'function_code': f"async function {tool_instance.name}(params) {{ return {json.dumps(sample_result, indent=2)}; }}",
                'parameters': tool_instance.parameters,
                'is_active': True,
                'created_at': datetime.utcnow().isoformat()
            }
            
            created = await ToolService.create_tool(tool_data)
            results.append(created)
            logger.info(f"✅ Registered built-in tool: {tool_name}")
            
        except Exception as error:
            logger.error(f"❌ Failed to register tool {tool_name}: {error}")
    
    logger.info(f"✅ Registered {len(results)} built-in tools")
    return results


async def execute_builtin_tool(tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a built-in tool by name
    
    Args:
        tool_name: Name of the tool to execute
        parameters: Parameters to pass to the tool
        
    Returns:
        Tool execution result
        
    Raises:
        ValueError: If tool not found
    """
    if tool_name not in BUILTIN_TOOLS:
        available_tools = list(BUILTIN_TOOLS.keys())
        raise ValueError(f"Built-in tool '{tool_name}' not found. Available tools: {available_tools}")
    
    tool_instance = BUILTIN_TOOLS[tool_name]
    return await tool_instance.execute(parameters)


def get_builtin_tool_info(tool_name: str) -> Dict[str, Any]:
    """
    Get information about a built-in tool
    
    Args:
        tool_name: Name of the tool
        
    Returns:
        Tool information dictionary
        
    Raises:
        ValueError: If tool not found
    """
    if tool_name not in BUILTIN_TOOLS:
        available_tools = list(BUILTIN_TOOLS.keys())
        raise ValueError(f"Built-in tool '{tool_name}' not found. Available tools: {available_tools}")
    
    tool_instance = BUILTIN_TOOLS[tool_name]
    return {
        'name': tool_instance.name,
        'description': tool_instance.description,
        'parameters': tool_instance.parameters
    }


def list_builtin_tools() -> List[str]:
    """
    List all available built-in tool names
    
    Returns:
        List of tool names
    """
    return list(BUILTIN_TOOLS.keys())


def get_all_builtin_tools_info() -> Dict[str, Dict[str, Any]]:
    """
    Get information about all built-in tools
    
    Returns:
        Dictionary mapping tool names to their information
    """
    return {
        tool_name: {
            'name': tool_instance.name,
            'description': tool_instance.description,
            'parameters': tool_instance.parameters
        }
        for tool_name, tool_instance in BUILTIN_TOOLS.items()
    }


# Export for backwards compatibility
builtin_tools = BUILTIN_TOOLS
