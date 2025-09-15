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
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=404, detail=f"Agent not found: {str(e)}")

@router.put("/{agent_id}", response_model=AgentResponse)
async def update_agent(agent_id: str, agent_data: AgentCreateRequest):
    """Update agent"""
    try:
        # Convert Pydantic model to dict
        agent_dict = agent_data.dict(exclude_unset=True)
        
        # Update the agent
        result = await AgentService.update_agent(agent_id, agent_dict)
        if not result:
            raise HTTPException(status_code=404, detail=f"Agent with ID {agent_id} not found")
        
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
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
        if not agents:
            # Return empty list instead of 404 when no agents found
            return []
        return agents
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error finding agents for intent '{intent}': {str(e)}")

@router.post("/{agent_id}/execute")
async def execute_agent(agent_id: str, execution_data: dict):
    """Execute agent directly"""
    try:
        # First, check if the agent exists
        try:
            agent = await AgentService.get_agent(agent_id)
            if not agent:
                raise HTTPException(status_code=404, detail=f"Agent with ID {agent_id} not found")
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=404, detail=f"Agent with ID {agent_id} not found: {str(e)}")
        
        # Validate input data
        if "input" not in execution_data:
            execution_data["input"] = "Hello"  # Default greeting if no input provided
            
        # Execute the agent
        try:
            result = await AgentService.execute_agent(agent, execution_data)
            return result
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Error executing agent: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))
