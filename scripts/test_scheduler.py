"""
Test Scheduler Service

This script tests the SchedulerService functionality including:
- Creating, updating, and deleting schedules
- Listing and searching schedules
- Pausing/resuming schedules
- Schedule execution
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from pprint import pprint

# Add parent directory to path to allow importing from app
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from app.core.database import init_supabase
from app.services.scheduler_service import SchedulerService
from app.services.log_service import LogService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_create_schedule():
    """Test creating a new schedule"""
    logger.info("\n=== Testing create_schedule ===")
    
    # Create a test schedule that runs every minute
    schedule = await SchedulerService.create_schedule(
        workflow_id=str(uuid.uuid4()),
        cron_expression="* * * * *",  # Every minute
        options={"test": True, "purpose": "test_schedule"},
        active=True
    )
    
    pprint(schedule)
    logger.info("✅ Schedule created successfully")
    return schedule['id']

async def test_get_schedule(schedule_id: str):
    """Test getting a schedule by ID"""
    logger.info("\n=== Testing get_schedule ===")
    
    schedule = await SchedulerService.get_schedule(schedule_id)
    assert schedule is not None, "Schedule not found"
    pprint(schedule)
    logger.info("✅ Schedule retrieved successfully")
    return schedule

async def test_update_schedule(schedule_id: str):
    """Test updating a schedule"""
    logger.info("\n=== Testing update_schedule ===")
    
    # Update to run every 2 minutes
    updated = await SchedulerService.update_schedule(
        schedule_id,
        cron_expression="*/2 * * * *",
        options={"test": True, "purpose": "updated_test_schedule"}
    )
    
    assert updated['cron_expression'] == "*/2 * * * *", "Cron expression not updated"
    pprint(updated)
    logger.info("✅ Schedule updated successfully")
    return updated

async def test_pause_resume_schedule(schedule_id: str):
    """Test pausing and resuming a schedule"""
    logger.info("\n=== Testing pause/resume schedule ===")
    
    # Pause the schedule
    paused = await SchedulerService.pause_schedule(schedule_id)
    assert paused['active'] is False, "Schedule not paused"
    logger.info("✅ Schedule paused successfully")
    
    # Verify it's paused
    schedule = await SchedulerService.get_schedule(schedule_id)
    assert schedule['active'] is False, "Schedule should be paused"
    
    # Resume the schedule
    resumed = await SchedulerService.resume_schedule(schedule_id)
    assert resumed['active'] is True, "Schedule not resumed"
    logger.info("✅ Schedule resumed successfully")
    
    # Verify it's active
    schedule = await SchedulerService.get_schedule(schedule_id)
    assert schedule['active'] is True, "Schedule should be active"
    
    return schedule

async def test_search_schedules():
    """Test searching schedules"""
    logger.info("\n=== Testing search_schedules ===")
    
    result = await SchedulerService.search_schedules(limit=5)
    assert 'schedules' in result, "No schedules in result"
    assert isinstance(result['schedules'], list), "Schedules should be a list"
    
    logger.info(f"Found {result['total']} total schedules")
    for i, s in enumerate(result['schedules'][:3]):  # Show first 3
        logger.info(f"  {i+1}. {s['id']} - {s['workflow_id']} - {s['cron_expression']}")
    
    logger.info("✅ Schedule search successful")
    return result

async def test_get_schedule_history(schedule_id: str):
    """Test getting schedule execution history"""
    logger.info("\n=== Testing get_schedule_history ===")
    
    history = await SchedulerService.get_schedule_history(schedule_id, limit=3)
    assert 'history' in history, "No history in result"
    
    logger.info(f"Found {history['total']} history entries")
    for i, entry in enumerate(history['history'][:3]):
        logger.info(f"  {i+1}. {entry.get('started_at')} - {entry.get('status')}")
    
    logger.info("✅ Schedule history retrieved successfully")
    return history

async def test_pause_all_schedules():
    """Test pausing all schedules"""
    logger.info("\n=== Testing pause_all_schedules ===")
    
    result = await SchedulerService.pause_all_schedules()
    assert 'paused_count' in result, "No paused_count in result"
    
    logger.info(f"✅ Paused {result['paused_count']} schedules")
    
    # Verify all schedules are paused
    active_schedules = await SchedulerService.search_schedules(active=True)
    assert active_schedules['count'] == 0, "Not all schedules were paused"
    
    return result

async def test_delete_schedule(schedule_id: str):
    """Test deleting a schedule"""
    logger.info("\n=== Testing delete_schedule ===")
    
    result = await SchedulerService.delete_schedule(schedule_id)
    assert result is True, "Delete failed"
    
    # Verify it's deleted
    schedule = await SchedulerService.get_schedule(schedule_id)
    assert schedule is None, "Schedule should be deleted"
    
    logger.info("✅ Schedule deleted successfully")
    return True

async def test_scheduler_stats():
    """Test getting scheduler statistics"""
    logger.info("\n=== Testing get_stats ===")
    
    stats = await SchedulerService.get_stats()
    pprint(stats)
    logger.info("✅ Scheduler stats retrieved successfully")
    return stats

async def test_next_run_times():
    """Test getting next run times"""
    logger.info("\n=== Testing get_next_run_times ===")
    
    next_runs = await SchedulerService.get_next_run_times(limit=3)
    pprint(next_runs)
    logger.info("✅ Next run times retrieved successfully")
    return next_runs

async def main():
    """Run all tests"""
    try:
        # Initialize services
        await init_supabase()
        await SchedulerService.initialize()
        
        logger.info("\n🚀 Starting Scheduler Service Tests\n")
        
        # Run tests
        schedule_id = await test_create_schedule()
        await test_get_schedule(schedule_id)
        await test_update_schedule(schedule_id)
        await test_pause_resume_schedule(schedule_id)
        await test_search_schedules()
        await test_scheduler_stats()
        await test_next_run_times()
        
        # These tests modify state, run them last
        await test_pause_all_schedules()
        await test_get_schedule_history(schedule_id)
        await test_delete_schedule(schedule_id)
        
        logger.info("\n✅ All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
    finally:
        # Cleanup
        await SchedulerService.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
