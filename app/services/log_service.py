"""
Log Service

Handles logging operations including creating, retrieving, and managing logs.
Provides analytics and cleanup functionality for system logs.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from ..core.database import get_supabase_client


class LogService:
    """Service for managing system logs and analytics"""
    
    @staticmethod
    async def get_logs(filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Get logs with optional filtering
        
        Args:
            filters: Optional filters including event_type, workflow_run_id, limit
            
        Returns:
            List of log records
        """
        if filters is None:
            filters = {}
            
        supabase = get_supabase_client()
        
        # Build query
        query = supabase.table('logs').select('*').order('created_at', desc=True)
        
        # Apply filters
        if filters.get('event_type'):
            query = query.eq('event_type', filters['event_type'])
            
        if filters.get('workflow_run_id'):
            query = query.eq('workflow_run_id', filters['workflow_run_id'])
            
        if filters.get('limit'):
            query = query.limit(filters['limit'])
        
        # Execute query
        response = await query.execute()
        
        if response.data is None:
            raise Exception(f"Failed to get logs: {getattr(response, 'error', 'Unknown error')}")
            
        return response.data
    
    @staticmethod
    async def create_log(
        event_type: str, 
        details: Dict[str, Any], 
        workflow_run_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a new log entry
        
        Args:
            event_type: Type of event being logged
            details: Event details and metadata
            workflow_run_id: Optional workflow run ID
            
        Returns:
            Created log record
        """
        supabase = get_supabase_client()
        
        log_data = {
            'event_type': event_type,
            'details': details,
            'workflow_run_id': workflow_run_id,
            'created_at': datetime.utcnow().isoformat()
        }
        
        response = await supabase.table('logs').insert(log_data).execute()
        
        if response.data is None or len(response.data) == 0:
            raise Exception(f"Failed to create log: {getattr(response, 'error', 'Unknown error')}")
            
        return response.data[0]
    
    @staticmethod
    async def get_workflow_run_logs(workflow_run_id: str) -> List[Dict[str, Any]]:
        """
        Get all logs for a specific workflow run
        
        Args:
            workflow_run_id: Workflow run ID
            
        Returns:
            List of log records for the workflow run
        """
        supabase = get_supabase_client()
        
        response = await supabase.table('logs').select('*').eq(
            'workflow_run_id', workflow_run_id
        ).order('created_at', desc=False).execute()
        
        if response.data is None:
            raise Exception(f"Failed to get workflow logs: {getattr(response, 'error', 'Unknown error')}")
            
        return response.data
    
    @staticmethod
    async def get_logs_by_event_type(event_type: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get logs filtered by event type
        
        Args:
            event_type: Event type to filter by
            limit: Maximum number of logs to return
            
        Returns:
            List of log records
        """
        supabase = get_supabase_client()
        
        response = await supabase.table('logs').select('*').eq(
            'event_type', event_type
        ).order('created_at', desc=True).limit(limit).execute()
        
        if response.data is None:
            raise Exception(f"Failed to get logs by event type: {getattr(response, 'error', 'Unknown error')}")
            
        return response.data
    
    @staticmethod
    async def delete_old_logs(days_old: int = 30) -> bool:
        """
        Delete logs older than specified number of days
        
        Args:
            days_old: Number of days to keep logs
            
        Returns:
            True if successful
        """
        supabase = get_supabase_client()
        
        cutoff_date = datetime.utcnow() - timedelta(days=days_old)
        cutoff_iso = cutoff_date.isoformat()
        
        response = await supabase.table('logs').delete().lt('created_at', cutoff_iso).execute()
        
        # Check if there was an error
        if hasattr(response, 'error') and response.error:
            raise Exception(f"Failed to delete old logs: {response.error}")
            
        return True
    
    @staticmethod
    async def get_log_stats() -> Dict[str, int]:
        """
        Get log statistics for the last 24 hours
        
        Returns:
            Dictionary with event type counts
        """
        supabase = get_supabase_client()
        
        # Get logs from last 24 hours
        twenty_four_hours_ago = datetime.utcnow() - timedelta(hours=24)
        
        response = await supabase.table('logs').select('event_type').gte(
            'created_at', twenty_four_hours_ago.isoformat()
        ).execute()
        
        if response.data is None:
            raise Exception(f"Failed to get log stats: {getattr(response, 'error', 'Unknown error')}")
        
        # Count events by type
        stats = {}
        for log in response.data:
            event_type = log.get('event_type', 'unknown')
            stats[event_type] = stats.get(event_type, 0) + 1
            
        return stats
    
    @staticmethod
    async def get_log_analytics(hours: int = 24) -> Dict[str, Any]:
        """
        Get comprehensive log analytics
        
        Args:
            hours: Number of hours to analyze
            
        Returns:
            Analytics data including counts, trends, and top events
        """
        supabase = get_supabase_client()
        
        # Get logs from specified time period
        start_time = datetime.utcnow() - timedelta(hours=hours)
        
        response = await supabase.table('logs').select('*').gte(
            'created_at', start_time.isoformat()
        ).order('created_at', desc=False).execute()
        
        if response.data is None:
            raise Exception(f"Failed to get log analytics: {getattr(response, 'error', 'Unknown error')}")
        
        logs = response.data
        
        # Calculate analytics
        total_logs = len(logs)
        event_counts = {}
        hourly_counts = {}
        workflow_counts = {}
        
        for log in logs:
            # Event type counts
            event_type = log.get('event_type', 'unknown')
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
            
            # Hourly distribution
            created_at = datetime.fromisoformat(log['created_at'].replace('Z', '+00:00'))
            hour_key = created_at.strftime('%Y-%m-%d %H:00')
            hourly_counts[hour_key] = hourly_counts.get(hour_key, 0) + 1
            
            # Workflow counts
            workflow_id = log.get('workflow_run_id')
            if workflow_id:
                workflow_counts[workflow_id] = workflow_counts.get(workflow_id, 0) + 1
        
        # Sort by frequency
        top_events = sorted(event_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        top_workflows = sorted(workflow_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'total_logs': total_logs,
            'time_period_hours': hours,
            'event_counts': event_counts,
            'hourly_distribution': hourly_counts,
            'top_events': top_events,
            'top_workflows': top_workflows,
            'average_logs_per_hour': total_logs / hours if hours > 0 else 0
        }
    
    @staticmethod
    async def search_logs(
        search_term: str, 
        event_types: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Search logs by content in details field
        
        Args:
            search_term: Term to search for in log details
            event_types: Optional list of event types to filter by
            limit: Maximum number of results
            
        Returns:
            List of matching log records
        """
        supabase = get_supabase_client()
        
        # Build query
        query = supabase.table('logs').select('*')
        
        # Filter by event types if provided
        if event_types:
            query = query.in_('event_type', event_types)
        
        # Execute query and filter in Python (since Supabase text search varies by setup)
        response = await query.order('created_at', desc=True).limit(limit * 2).execute()
        
        if response.data is None:
            raise Exception(f"Failed to search logs: {getattr(response, 'error', 'Unknown error')}")
        
        # Filter results by search term
        matching_logs = []
        search_lower = search_term.lower()
        
        for log in response.data:
            details = log.get('details', {})
            details_str = str(details).lower()
            
            if search_lower in details_str or search_lower in log.get('event_type', '').lower():
                matching_logs.append(log)
                
            if len(matching_logs) >= limit:
                break
        
        return matching_logs
    
    @staticmethod
    async def get_error_logs(hours: int = 24, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get error logs from specified time period
        
        Args:
            hours: Number of hours to look back
            limit: Maximum number of error logs
            
        Returns:
            List of error log records
        """
        error_event_types = [
            'error',
            'exception',
            'failure',
            'timeout',
            'connection_error',
            'validation_error'
        ]
        
        return await LogService.search_logs(
            search_term='error',
            event_types=error_event_types,
            limit=limit
        )
