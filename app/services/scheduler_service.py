"""
Scheduler Service

Cron-based workflow scheduling service using APScheduler.
Manages scheduled workflow executions with database persistence.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.executors.asyncio import AsyncIOExecutor
from ..core.database import get_supabase_client
from .workflow_service import WorkflowService

logger = logging.getLogger(__name__)


class SchedulerService:
    """Service for managing scheduled workflow executions"""
    
    _scheduler: Optional[AsyncIOScheduler] = None
    _initialized: bool = False
    _scheduled_jobs: Dict[str, str] = {}  # schedule_id -> job_id mapping
    
    @classmethod
    async def initialize(cls) -> None:
        """Initialize scheduler service"""
        if cls._initialized:
            return
        
        try:
            logger.info("⏰ Initializing scheduler service...")
            
            # Configure APScheduler
            jobstores = {
                'default': MemoryJobStore()
            }
            executors = {
                'default': AsyncIOExecutor()
            }
            job_defaults = {
                'coalesce': False,
                'max_instances': 3,
                'misfire_grace_time': 30
            }
            
            cls._scheduler = AsyncIOScheduler(
                jobstores=jobstores,
                executors=executors,
                job_defaults=job_defaults,
                timezone='UTC'
            )
            
            # Start the scheduler
            cls._scheduler.start()
            
            # Load existing schedules from database
            supabase = get_supabase_client()
            response = supabase.table('workflow_schedules').select('*').eq('active', True).execute()
            
            if hasattr(response, 'data') and response.data is not None:
                schedules = response.data
            else:
                logger.warning("No schedules found in database")
                schedules = []
            
            # Start each active schedule
            for schedule in schedules:
                await cls._start_schedule(schedule)
            
            cls._initialized = True
            logger.info(f"✅ Scheduler initialized with {len(schedules)} active schedules")
            
        except Exception as error:
            logger.error(f"❌ Failed to initialize scheduler: {error}")
            raise error
    
    @classmethod
    async def create_schedule(
        cls, 
        workflow_id: str, 
        cron_expression: str, 
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a new workflow schedule
        
        Args:
            workflow_id: ID of workflow to schedule
            cron_expression: Cron expression for scheduling
            options: Additional options including metadata
            
        Returns:
            Created schedule record
        """
        if options is None:
            options = {}
            
        try:
            # Validate cron expression by creating a trigger
            try:
                CronTrigger.from_crontab(cron_expression)
            except Exception as e:
                raise ValueError(f"Invalid cron expression: {e}")
            
            # Insert schedule into database
            supabase = get_supabase_client()
            schedule_data = {
                'workflow_id': workflow_id,
                'cron_expression': cron_expression,
                'active': True,
                'metadata': options.get('metadata', {}),
                'created_at': datetime.utcnow().isoformat()
            }
            
            response = await supabase.table('workflow_schedules').insert(schedule_data).execute()
            
            if response.data is None or len(response.data) == 0:
                raise Exception(f"Failed to create schedule: {getattr(response, 'error', 'Unknown error')}")
            
            schedule = response.data[0]
            
            # Start the schedule
            await cls._start_schedule(schedule)
            
            logger.info(f"✅ Created schedule {schedule['id']} for workflow {workflow_id}")
            return schedule
            
        except Exception as error:
            logger.error(f"❌ Failed to create schedule: {error}")
            raise error
    
    @classmethod
    async def _start_schedule(cls, schedule: Dict[str, Any]) -> None:
        """
        Start a schedule
        
        Args:
            schedule: Schedule record from database
        """
        try:
            if not cls._scheduler:
                await cls.initialize()
            
            # Create job ID
            job_id = f"schedule_{schedule['id']}"
            
            # Create cron trigger
            trigger = CronTrigger.from_crontab(schedule['cron_expression'], timezone='UTC')
            
            # Add job to scheduler
            cls._scheduler.add_job(
                cls._execute_scheduled_workflow,
                trigger=trigger,
                id=job_id,
                args=[schedule],
                replace_existing=True,
                name=f"Workflow {schedule['workflow_id']} Schedule"
            )
            
            # Track the job
            cls._scheduled_jobs[schedule['id']] = job_id
            
            logger.info(f"⏰ Started schedule {schedule['id']}: {schedule['cron_expression']}")
            
        except Exception as error:
            logger.error(f"❌ Failed to start schedule {schedule['id']}: {error}")
            raise error
    
    @classmethod
    async def _execute_scheduled_workflow(cls, schedule: Dict[str, Any]) -> None:
        """
        Execute a scheduled workflow
        
        Args:
            schedule: Schedule record from database
        """
        logger.info(f"⏰ Executing scheduled workflow: {schedule['workflow_id']}")
        
        try:
            # Execute the workflow
            run_id = f"scheduled_{schedule['id']}_{int(datetime.now().timestamp() * 1000)}"
            input_data = schedule.get('metadata', {}).get('input', {})
            
            result = await WorkflowService.run_workflow(
                schedule['workflow_id'],
                input_data,
                run_id
            )
            
            # Log successful execution
            supabase = get_supabase_client()
            await supabase.table('logs').insert({
                'event_type': 'scheduled_workflow_success',
                'details': {
                    'schedule_id': schedule['id'],
                    'workflow_id': schedule['workflow_id'],
                    'run_id': result.get('run_id', run_id),
                    'execution_time_ms': result.get('execution_time_ms', 0),
                    'timestamp': datetime.utcnow().isoformat()
                },
                'created_at': datetime.utcnow().isoformat()
            }).execute()
            
            logger.info(f"✅ Scheduled workflow {schedule['workflow_id']} completed successfully")
            
        except Exception as execution_error:
            logger.error(f"❌ Scheduled workflow {schedule['workflow_id']} failed: {execution_error}")
            
            # Log failed execution
            try:
                supabase = get_supabase_client()
                await supabase.table('logs').insert({
                    'event_type': 'scheduled_workflow_error',
                    'details': {
                        'schedule_id': schedule['id'],
                        'workflow_id': schedule['workflow_id'],
                        'error': str(execution_error),
                        'timestamp': datetime.utcnow().isoformat()
                    },
                    'created_at': datetime.utcnow().isoformat()
                }).execute()
            except Exception as log_error:
                logger.error(f"Failed to log scheduled workflow error: {log_error}")
    
    @classmethod
    async def stop_schedule(cls, schedule_id: str) -> None:
        """
        Stop a schedule
        
        Args:
            schedule_id: ID of schedule to stop
        """
        try:
            # Remove job from scheduler
            job_id = cls._scheduled_jobs.get(schedule_id)
            if job_id and cls._scheduler:
                try:
                    cls._scheduler.remove_job(job_id)
                except Exception:
                    pass  # Job might not exist
                
                cls._scheduled_jobs.pop(schedule_id, None)
            
            # Update database
            supabase = get_supabase_client()
            await supabase.table('workflow_schedules').update({
                'active': False,
                'updated_at': datetime.utcnow().isoformat()
            }).eq('id', schedule_id).execute()
            
            logger.info(f"⏹️  Stopped schedule {schedule_id}")
            
        except Exception as error:
            logger.error(f"❌ Failed to stop schedule {schedule_id}: {error}")
            raise error
    
    @classmethod
    async def list_schedules(cls) -> List[Dict[str, Any]]:
        """
        List all schedules
        
        Returns:
            List of schedule records with workflow information
        """
        try:
            supabase = get_supabase_client()
            
            # Note: Adjust the join syntax based on your Supabase setup
            response = await supabase.table('workflow_schedules').select(
                """
                *,
                workflows!inner(id, name, description)
                """
            ).order('created_at', desc=True).execute()
            
            if response.data is None:
                logger.warning("Failed to list schedules from database")
                return []
            
            return response.data
            
        except Exception as error:
            logger.error(f"❌ Failed to list schedules: {error}")
            raise error
    
    @classmethod
    async def update_schedule(cls, schedule_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update a schedule
        
        Args:
            schedule_id: ID of schedule to update
            updates: Updates to apply
            
        Returns:
            Updated schedule record
        """
        try:
            # Stop existing schedule
            await cls.stop_schedule(schedule_id)
            
            # Update database
            supabase = get_supabase_client()
            update_data = {
                **updates,
                'updated_at': datetime.utcnow().isoformat()
            }
            
            response = await supabase.table('workflow_schedules').update(update_data).eq(
                'id', schedule_id
            ).execute()
            
            if response.data is None or len(response.data) == 0:
                raise Exception(f"Failed to update schedule: {getattr(response, 'error', 'Schedule not found')}")
            
            schedule = response.data[0]
            
            # Restart if still active
            if schedule.get('active', False):
                await cls._start_schedule(schedule)
            
            logger.info(f"✅ Updated schedule {schedule_id}")
            return schedule
            
        except Exception as error:
            logger.error(f"❌ Failed to update schedule {schedule_id}: {error}")
            raise error
    
    @classmethod
    async def delete_schedule(cls, schedule_id: str) -> None:
        """
        Delete a schedule
        
        Args:
            schedule_id: ID of schedule to delete
        """
        try:
            # Stop the schedule
            await cls.stop_schedule(schedule_id)
            
            # Delete from database
            supabase = get_supabase_client()
            await supabase.table('workflow_schedules').delete().eq('id', schedule_id).execute()
            
            logger.info(f"🗑️  Deleted schedule {schedule_id}")
            
        except Exception as error:
            logger.error(f"❌ Failed to delete schedule {schedule_id}: {error}")
            raise error
    
    @classmethod
    async def get_schedule_by_id(cls, schedule_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a schedule by ID
        
        Args:
            schedule_id: ID of schedule to retrieve
            
        Returns:
            Schedule record or None if not found
        """
        try:
            supabase = get_supabase_client()
            response = await supabase.table('workflow_schedules').select('*').eq('id', schedule_id).execute()
            
            if response.data is None or len(response.data) == 0:
                return None
            
            return response.data[0]
            
        except Exception as error:
            logger.error(f"❌ Failed to get schedule {schedule_id}: {error}")
            return None
    
    @classmethod
    async def pause_schedule(cls, schedule_id: str) -> Dict[str, Any]:
        """
        Pause a schedule (stop execution but keep record)
        
        Args:
            schedule_id: ID of schedule to pause
            
        Returns:
            Updated schedule record
        """
        return await cls.update_schedule(schedule_id, {'active': False})
    
    @classmethod
    async def resume_schedule(cls, schedule_id: str) -> Dict[str, Any]:
        """
        Resume a paused schedule
        
        Args:
            schedule_id: ID of schedule to resume
            
        Returns:
            Updated schedule record
        """
        return await cls.update_schedule(schedule_id, {'active': True})
    
    @classmethod
    async def shutdown(cls) -> None:
        """Shutdown all schedules"""
        if not cls._initialized or not cls._scheduler:
            logger.info("ℹ️  Scheduler not initialized, nothing to shut down")
            return
            
        logger.info("⏹️  Shutting down scheduler...")
        
        try:
            cls._scheduler.shutdown(wait=False)
            cls._scheduler = None
            cls._initialized = False
            logger.info("✅ Scheduler shutdown complete")
        except Exception as error:
                logger.error(f"❌ Error during scheduler shutdown: {error}")
        
        cls._scheduled_jobs.clear()
        cls._initialized = False
        cls._scheduler = None
    
    @classmethod
    async def get_stats(cls) -> Dict[str, Any]:
        """
        Get schedule statistics
        
        Returns:
            Statistics about schedules and running jobs
        """
        try:
            supabase = get_supabase_client()
            response = await supabase.table('workflow_schedules').select('active').execute()
            
            if response.data is None:
                stats_data = []
            else:
                stats_data = response.data
            
            total = len(stats_data)
            active = len([s for s in stats_data if s.get('active', False)])
            inactive = total - active
            
            # Get running jobs count
            running_jobs = 0
            if cls._scheduler:
                running_jobs = len(cls._scheduler.get_jobs())
            
            return {
                'total_schedules': total,
                'active_schedules': active,
                'inactive_schedules': inactive,
                'running_jobs': running_jobs,
                'scheduler_running': cls._scheduler is not None and cls._scheduler.running
            }
            
        except Exception as error:
            logger.error(f"❌ Failed to get scheduler stats: {error}")
            return {
                'total_schedules': 0,
                'active_schedules': 0,
                'inactive_schedules': 0,
                'running_jobs': len(cls._scheduled_jobs),
                'scheduler_running': cls._scheduler is not None and cls._scheduler.running if cls._scheduler else False
            }
    
    @classmethod
    async def get_next_run_times(cls, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get next run times for active schedules
        
        Args:
            limit: Maximum number of schedules to return
            
        Returns:
            List of schedules with next run times
        """
        try:
            if not cls._scheduler:
                return []
            
            jobs = cls._scheduler.get_jobs()
            next_runs = []
            
            for job in jobs[:limit]:
                if job.next_run_time:
                    # Extract schedule ID from job ID
                    schedule_id = job.id.replace('schedule_', '') if job.id.startswith('schedule_') else job.id
                    
                    next_runs.append({
                        'schedule_id': schedule_id,
                        'job_id': job.id,
                        'job_name': job.name,
                        'next_run_time': job.next_run_time.isoformat(),
                        'trigger': str(job.trigger)
                    })
            
            # Sort by next run time
            next_runs.sort(key=lambda x: x['next_run_time'])
            
            return next_runs
            
        except Exception as error:
            logger.error(f"❌ Failed to get next run times: {error}")
            return []
