"""
Scheduler Service

Cron-based workflow scheduling service using APScheduler.
Manages scheduled workflow executions with database persistence.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Awaitable
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.executors.asyncio import AsyncIOExecutor
from apscheduler.job import Job
from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED, EVENT_JOB_MISSED
import json
import uuid

from ..core.database import get_supabase_client
from .workflow_service import WorkflowService
from .log_service import LogService
logger = logging.getLogger(__name__)

class SchedulerService:
    """Service for managing scheduled workflow executions"""
    
    _scheduler: Optional[AsyncIOScheduler] = None
    _initialized: bool = False
    _scheduled_jobs: Dict[str, str] = {}  # schedule_id -> job_id mapping
    _job_callbacks: Dict[str, Callable[[Dict], Awaitable[None]]] = {}
    
    @classmethod
    async def initialize(cls) -> None:
        """Initialize scheduler service with proper error handling and logging"""
        if cls._initialized:
            logger.warning("Scheduler already initialized")
            return
        
        try:
            logger.info("⏰ Initializing scheduler service...")
            
            # Configure APScheduler with explicit job store and executor
            jobstores = {
                'default': MemoryJobStore()
            }
            
            executors = {
                'default': AsyncIOExecutor()
            }
            
            job_defaults = {
                'coalesce': True,  # Combine multiple pending runs
                'max_instances': 3,  # Max concurrent instances per job
                'misfire_grace_time': 60 * 5,  # 5 minutes grace period
                'replace_existing': True  # Allow updating existing jobs
            }
            
            # Initialize scheduler with explicit timezone
            cls._scheduler = AsyncIOScheduler(
                jobstores=jobstores,
                executors=executors,
                job_defaults=job_defaults,
                timezone='UTC'
            )
            
            # Add event listeners
            cls._scheduler.add_listener(
                cls._job_executed_listener,
                EVENT_JOB_EXECUTED | EVENT_JOB_ERROR | EVENT_JOB_MISSED
            )
            
            # Start the scheduler
            cls._scheduler.start()
            
            # Load active schedules from database
            await cls._load_schedules()
            
            cls._initialized = True
            logger.info(f"✅ Scheduler initialized with {len(cls._scheduled_jobs)} active schedules")
            
        except Exception as error:
            logger.error(f"❌ Failed to initialize scheduler: {error}", exc_info=True)
            raise
    
    @classmethod
    async def shutdown(cls) -> None:
        """Gracefully shutdown the scheduler"""
        if cls._scheduler and cls._scheduler.running:
            logger.info("🛑 Shutting down scheduler...")
            cls._scheduler.shutdown(wait=True)
            cls._initialized = False
            logger.info("✅ Scheduler stopped")
    
    @classmethod
    async def _load_schedules(cls) -> None:
        """Load schedules from database and schedule them"""
        try:
            supabase = get_supabase_client()
            response = supabase.table('workflow_schedules') \
                .select('*') \
                .eq('active', True) \
                .execute()
            
            if not response.data:
                logger.info("No active schedules found in database")
                return
            
            for schedule in response.data:
                try:
                    await cls._start_schedule(schedule)
                except Exception as e:
                    logger.error(f"Failed to schedule workflow {schedule.get('id')}: {e}", exc_info=True)
                    await LogService.log(
                        "scheduler_error",
                        {
                            "message": f"Failed to schedule workflow {schedule.get('id')}",
                            "error": str(e),
                            "schedule_id": schedule.get('id')
                        }
                    )
                    
        except Exception as e:
            logger.error(f"Error loading schedules: {e}", exc_info=True)
            raise
    
    @classmethod
    async def _job_executed_listener(cls, event):
        """Handle job execution events"""
        job_id = event.job_id
        schedule_id = next((k for k, v in cls._scheduled_jobs.items() if v == job_id), None)
        
        if not schedule_id:
            return
            
        if event.exception:
            logger.error(f"❌ Job {job_id} failed: {event.exception}", exc_info=event.exception)
            await LogService.log(
                "scheduler_error",
                {
                    "job_id": job_id,
                    "schedule_id": schedule_id,
                    "error": str(event.exception),
                    "traceback": str(event.traceback) if hasattr(event, 'traceback') else None
                }
            )
        elif event.code == EVENT_JOB_MISSED:
            logger.warning(f"⏰ Job {job_id} missed scheduled run")
            await LogService.log(
                "scheduler_warning",
                {
                    "job_id": job_id,
                    "schedule_id": schedule_id,
                    "message": "Job missed scheduled run"
                }
            )
    
    @classmethod
    async def _execute_workflow(cls, workflow_id: str, schedule_id: str, options: Dict[str, Any] = None) -> None:
        """Execute a workflow from a scheduled job"""
        try:
            from .workflow_service import WorkflowService
            
            logger.info(f"🚀 Executing scheduled workflow {workflow_id} from schedule {schedule_id}")
            
            # Start workflow execution
            workflow_service = WorkflowService()
            result = await workflow_service.execute_workflow(
                workflow_id=workflow_id,
                input_data=options.get('input_data', {}),
                session_id=f"sched_{schedule_id}_{uuid.uuid4()}"
            )
            
            logger.info(f"✅ Completed scheduled workflow {workflow_id} with result: {result.get('status')}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error executing scheduled workflow {workflow_id}: {e}", exc_info=True)
            await LogService.log(
                "workflow_error",
                {
                    "workflow_id": workflow_id,
                    "schedule_id": schedule_id,
                    "error": str(e),
                    "options": options
                }
            )
            raise
    
    @classmethod
    async def _start_schedule(cls, schedule: Dict[str, Any]) -> None:
        """Start a schedule by adding it to the scheduler"""
        schedule_id = str(schedule.get('id'))
        workflow_id = schedule.get('workflow_id')
        cron_expression = schedule.get('cron_expression')
        options = schedule.get('options', {})
        
        if not all([schedule_id, workflow_id, cron_expression]):
            raise ValueError("Schedule is missing required fields (id, workflow_id, cron_expression)")
        
        # Check if already scheduled
        if schedule_id in cls._scheduled_jobs:
            logger.warning(f"Schedule {schedule_id} is already running")
            return
        
        try:
            # Parse cron expression
            trigger = CronTrigger.from_crontab(cron_expression)
            
            # Add job to scheduler
            job = cls._scheduler.add_job(
                cls._execute_workflow,
                trigger=trigger,
                args=[workflow_id, schedule_id],
                kwargs={'options': options},
                id=f"workflow_{workflow_id}_{schedule_id}",
                name=f"Workflow {workflow_id} - {schedule_id}",
                replace_existing=True
            )
            
            # Store job reference
            cls._scheduled_jobs[schedule_id] = job.id
            
            logger.info(f"✅ Scheduled workflow {workflow_id} with ID {job.id} (next run: {job.next_run_time})")
            
        except Exception as e:
            logger.error(f"Failed to schedule workflow {workflow_id}: {e}")
            raise
    
    @classmethod
    async def create_schedule(
        cls, 
        workflow_id: str, 
        cron_expression: str, 
        options: Optional[Dict[str, Any]] = None,
        active: bool = True
    ) -> Dict[str, Any]:
        """
        Create a new workflow schedule
        
        Args:
            workflow_id: ID of workflow to schedule
            cron_expression: Cron expression for scheduling
            options: Additional options including metadata
            active: Whether the schedule should be active immediately
            
        Returns:
            Created schedule record
        """
        if options is None:
            options = {}
            
        try:
            # Validate cron expression
            CronTrigger.from_crontab(cron_expression)
            
            # Create schedule in database
            supabase = get_supabase_client()
            schedule_data = {
                'workflow_id': workflow_id,
                'cron_expression': cron_expression,
                'options': options,
                'active': active,
                'next_run': None  # Will be set when the job is scheduled
            }
            
            response = supabase.table('workflow_schedules') \
                .insert(schedule_data) \
                .execute()
                
            schedule = response.data[0] if response.data else None
            
            if not schedule:
                raise ValueError("Failed to create schedule in database")
                
            # Start the schedule if active
            if active:
                await cls._start_schedule(schedule)
                
                # Update next run time
                job_id = cls._scheduled_jobs.get(str(schedule['id']))
                if job_id:
                    job = cls._scheduler.get_job(job_id)
                    if job and job.next_run_time:
                        supabase.table('workflow_schedules') \
                            .update({'next_run': job.next_run_time.isoformat()}) \
                            .eq('id', schedule['id']) \
                            .execute()
            
            logger.info(f"✅ Created schedule {schedule['id']} for workflow {workflow_id}")
            return schedule
            
        except Exception as e:
            logger.error(f"❌ Failed to create schedule: {e}")
            await LogService.log(
                "scheduler_error",
                {
                    "message": "Failed to create schedule",
                    "workflow_id": workflow_id,
                    "error": str(e)
                }
            )
            raise
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
