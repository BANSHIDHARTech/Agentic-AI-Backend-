from typing import Optional
from fastapi import APIRouter, HTTPException, Path
from ..core.models import AgentCreateRequest, AgentResponse, APIResponse
from ..services.agent_service import AgentService

router = APIRouter()

@router.post("/", response_model=AgentResponse, status_code=201)
async def create_or_update_agent(agent_data: AgentCreateRequest, id: Optional[str] = None):
    """Create or update agent"""
    try:
        if id:
            result = await AgentService.update_agent(id, agent_data.dict())
        else:
            result = await AgentService.create_agent(agent_data.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/")
async def get_all_agents():
    """Get all agents"""
    try:
        agents = await AgentService.get_all_agents()
        return agents
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent(agent_id: str = Path(..., description="Agent ID")):
    """Get agent by ID"""
    try:
        agent = await AgentService.get_agent(agent_id)
        return agent
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.put("/{agent_id}", response_model=AgentResponse)
async def update_agent(agent_id: str, agent_data: AgentCreateRequest):
    """Update agent"""
    try:
        result = await AgentService.update_agent(agent_id, agent_data.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.delete("/{agent_id}")
async def delete_agent(agent_id: str):
    """Delete agent"""
    try:
        await AgentService.delete_agent(agent_id)
        return {"message": "Agent deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/intent/{intent}")
async def find_agents_by_intent(intent: str):
    """Find agents by intent"""
    try:
        agents = await AgentService.find_agent_by_intent(intent)
        return agents
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/{agent_id}/execute")
async def execute_agent(agent_id: str, execution_data: dict):
    """Execute agent directly"""
    try:
        agent = await AgentService.get_agent(agent_id)
        result = await AgentService.execute_agent(agent, execution_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
