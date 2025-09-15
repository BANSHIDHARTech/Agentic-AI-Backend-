from typing import Dict, Any, List, Optional
from ..core.database import db_insert, db_update, db_select, db_delete
from ..core.models import AgentModel
from .llm_service import LLMService

class AgentService:
    """Agent management service for CRUD operations and execution"""
    
    @classmethod
    async def create_agent(cls, agent_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new agent"""
        try:
            # Validate using Pydantic model
            validated_data = AgentModel(**agent_data).dict()
            
            # Check if an agent with this name already exists
            try:
                name = validated_data.get("name")
                if name:
                    # Check if there's already an agent with this name
                    from ..core.database import get_supabase_client
                    supabase = get_supabase_client()
                    existing_agents = supabase.table("agents").select("id").eq("name", name).execute()
                    
                    if existing_agents.data and len(existing_agents.data) > 0:
                        # Generate a unique name by adding a timestamp
                        import time
                        timestamp = int(time.time())
                        validated_data["name"] = f"{name}_{timestamp}"
                        print(f"Agent name '{name}' already exists. Using '{validated_data['name']}' instead.")
            except Exception as e:
                print(f"Error checking for existing agent name: {e}")
                # Continue with the original name
            
            # Insert the agent into database
            try:
                result = await db_insert("agents", validated_data)
                if not result:
                    raise Exception("db_insert returned None or empty result")
                if not isinstance(result, list) or len(result) == 0:
                    raise Exception(f"Unexpected result format from db_insert: {result}")
                return result[0] if result else None
            except Exception as e:
                print(f"Error during agent insertion: {str(e)}")
                # Re-raise for API error handling
                raise Exception(f"Failed to insert agent: {str(e)}")
                
        except Exception as e:
            print(f"Error in create_agent: {str(e)}")
            raise Exception(f"Agent creation failed: {str(e)}")
    
    @classmethod
    async def update_agent(cls, agent_id: str, agent_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing agent"""
        try:
            # Check if agent exists first
            try:
                existing_agent = await cls.get_agent(agent_id)
                if not existing_agent:
                    raise Exception(f"Agent with ID {agent_id} not found")
            except Exception as e:
                print(f"Agent with ID {agent_id} not found: {str(e)}")
                raise Exception(f"Agent with ID {agent_id} not found")
            
            # Validate using Pydantic model
            try:
                validated_data = AgentModel(**agent_data).dict()
            except Exception as e:
                print(f"Validation error: {str(e)}")
                raise Exception(f"Invalid agent data: {str(e)}")
            
            # Add updated_at timestamp
            from datetime import datetime
            validated_data['updated_at'] = datetime.utcnow().isoformat()
            
            # Use direct Supabase update instead of db_update helper
            from ..core.database import get_supabase_client
            supabase = get_supabase_client()
            
            result = supabase.table("agents").update(validated_data).eq("id", agent_id).execute()
            
            if hasattr(result, 'error') and result.error:
                print(f"Update error: {result.error}")
                raise Exception(f"Failed to update agent: {result.error}")
                
            if not hasattr(result, 'data') or not result.data or len(result.data) == 0:
                print("Update succeeded but no data returned")
                # Fall back to getting the agent
                updated_agent = await cls.get_agent(agent_id)
                return updated_agent
            
            return result.data[0]
            
        except Exception as e:
            print(f"Error in update_agent: {str(e)}")
            raise Exception(f"Agent update failed: {str(e)}")
    
    @classmethod
    async def get_agent(cls, agent_id: str) -> Dict[str, Any]:
        """Get agent by ID"""
        from ..core.database import get_supabase_client
        
        supabase = get_supabase_client()
        result = supabase.table("agents").select("*").eq("id", agent_id).execute()
        
        if hasattr(result, 'error') and result.error:
            raise Exception(result.error)
        
        if not result.data:
            raise Exception(f"Agent with ID {agent_id} not found")
        
        return result.data[0]
    
    @classmethod
    async def get_agent_by_name(cls, name: str) -> Dict[str, Any]:
        """Get agent by name"""
        from ..core.database import get_supabase_client
        
        supabase = get_supabase_client()
        result = supabase.table("agents").select("*").eq("name", name).execute()
        
        if hasattr(result, 'error') and result.error:
            raise Exception(result.error)
        
        if not result.data:
            raise Exception(f"Agent with name {name} not found")
        
        return result.data[0]
    
    @classmethod
    async def get_all_agents(cls) -> List[Dict[str, Any]]:
        """Get all agents"""
        try:
            # Using the db_select helper which properly handles the response
            agents = await db_select("agents")
            return agents or []
        except Exception as e:
            print(f"Error getting agents: {str(e)}")
            return [] or []
    
    @classmethod
    async def delete_agent(cls, agent_id: str) -> bool:
        """Delete an agent"""
        return await db_delete("agents", agent_id)
    
    @classmethod
    async def find_agent_by_intent(cls, intent: str) -> List[Dict[str, Any]]:
        """Find agents that can handle a specific intent"""
        try:
            from ..core.database import get_supabase_client
            
            supabase = get_supabase_client()
            
            # There seems to be an issue with using .contains() directly with Supabase
            # Let's fetch all active agents and filter them in Python instead
            result = supabase.table("agents").select("*").eq("is_active", True).execute()
            
            if hasattr(result, 'error') and result.error:
                raise Exception(result.error)
            
            # Filter agents that have the specified intent in their input_intents array
            matching_agents = []
            if result.data:
                for agent in result.data:
                    input_intents = agent.get("input_intents", [])
                    if input_intents and intent in input_intents:
                        matching_agents.append(agent)
            
            return matching_agents
            
        except Exception as e:
            print(f"Error in find_agent_by_intent: {str(e)}")
            raise Exception(f"Failed to find agents for intent '{intent}': {str(e)}")
    
    @classmethod
    async def execute_agent(cls, agent: Dict[str, Any], input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an agent with given input"""
        try:
            # Check if agent is properly formatted
            if not isinstance(agent, dict):
                raise ValueError(f"Invalid agent type: {type(agent)}, expected dictionary")
                
            if "id" not in agent or "name" not in agent:
                raise ValueError(f"Invalid agent data: missing required fields (id, name)")
            
            # Validate input data
            if not isinstance(input_data, dict):
                raise ValueError(f"Invalid input_data type: {type(input_data)}, expected dictionary")
            
            # Initialize LLM service if not already done
            try:
                await LLMService.initialize()
            except Exception as e:
                print(f"Error initializing LLM service: {str(e)}")
                # Provide a mock response for testing if LLM service isn't available
                return {
                    "agent_id": agent["id"],
                    "agent_name": agent["name"],
                    "input": input_data.get("input", ""),
                    "output": f"[Test mode] I'm {agent.get('name', 'Agent')}. LLM service not available: {str(e)}",
                    "model_used": agent.get("model_name", "test-model"),
                    "provider_used": "mock",
                    "processing_time_ms": 0,
                    "success": False,
                    "context": input_data.get("context", {})
                }
            
            # Prepare the prompt
            user_input = input_data.get("input", "")
            context = input_data.get("context", {})
            
            # Build the prompt with context
            prompt = cls._build_agent_prompt(agent, user_input, context)
            
            # Generate completion using LLM service
            try:
                completion = await LLMService.generate_completion(
                    prompt=prompt,
                    model_name=agent.get("model_name", "gpt-4"),
                    system_prompt=agent.get("system_prompt"),
                    max_tokens=input_data.get("max_tokens", 1000),
                    temperature=input_data.get("temperature", 0.7)
                )
            except Exception as e:
                print(f"Error generating completion: {str(e)}")
                # Provide a mock response for testing if completion fails
                completion = {
                    "content": f"[Test mode] Error generating response: {str(e)}",
                    "model": agent.get("model_name", "test-model"),
                    "provider": "mock",
                    "processing_time_ms": 0
                }
            
            # Log the execution
            try:
                await cls._log_agent_execution(
                    agent_id=agent["id"],
                    input_data=input_data,
                    output_data=completion,
                    success=True
                )
            except Exception as e:
                print(f"Error logging execution: {str(e)}")
                # Continue even if logging fails
            
            return {
                "agent_id": agent["id"],
                "agent_name": agent["name"],
                "input": user_input,
                "output": completion["content"],
                "model_used": completion.get("model", agent.get("model_name", "unknown")),
                "provider_used": completion.get("provider", "unknown"),
                "processing_time_ms": completion.get("processing_time_ms", 0),
                "success": True,
                "context": context
            }
            
        except Exception as error:
            print(f"Agent execution error: {str(error)}")
            
            # Try to log the failure, but don't fail if logging fails
            try:
                await cls._log_agent_execution(
                    agent_id=agent.get("id", "unknown"),
                    input_data=input_data,
                    output_data={},
                    success=False,
                    error_message=str(error)
                )
            except Exception as log_error:
                print(f"Error logging execution failure: {str(log_error)}")
            
            raise Exception(f"Agent execution failed: {error}")
    
    @classmethod
    def _build_agent_prompt(cls, agent: Dict[str, Any], user_input: str, context: Dict[str, Any]) -> str:
        """Build the complete prompt for agent execution"""
        prompt_parts = []
        
        # Add context if provided
        if context:
            prompt_parts.append("Context:")
            for key, value in context.items():
                prompt_parts.append(f"- {key}: {value}")
            prompt_parts.append("")
        
        # Add the user input
        prompt_parts.append("User Input:")
        prompt_parts.append(user_input)
        
        # Add any specific instructions from the agent
        if agent.get("input_intents"):
            prompt_parts.append("")
            prompt_parts.append("This agent specializes in:")
            for intent in agent["input_intents"]:
                prompt_parts.append(f"- {intent}")
        
        return "\n".join(prompt_parts)
    
    @classmethod
    async def _log_agent_execution(
        cls,
        agent_id: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        success: bool,
        error_message: Optional[str] = None
    ):
        """Log agent execution to database"""
        try:
            from ..core.database import get_supabase_client
            
            log_data = {
                "agent_id": agent_id,
                "input_data": input_data,
                "output_data": output_data,
                "success": success,
                "error_message": error_message,
                "processing_time_ms": output_data.get("processing_time_ms", 0),
                "created_at": "now()"
            }
            
            supabase = get_supabase_client()
            result = supabase.table("agent_executions").insert(log_data).execute()
            if hasattr(result, 'error') and result.error:
                print(f"Failed to log agent execution: {result.error}")
                
        except Exception as error:
            print(f"Failed to log agent execution: {error}")
    
    @classmethod
    async def get_agent_stats(cls, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get agent execution statistics"""
        try:
            from ..core.database import get_supabase_client
            
            supabase = get_supabase_client()
            query = supabase.table("agent_executions").select("*")
            
            if agent_id:
                query = query.eq("agent_id", agent_id)
            
            result = query.execute()
            
            if hasattr(result, 'error') and result.error:
                raise Exception(result.error)
            
            executions = result.data or []
            
            if not executions:
                return {
                    "total_executions": 0,
                    "success_rate": 0,
                    "avg_processing_time_ms": 0,
                    "agent_breakdown": {}
                }
            
            total = len(executions)
            successful = len([e for e in executions if e["success"]])
            avg_time = sum(e["processing_time_ms"] or 0 for e in executions) / total
            
            # Agent breakdown
            agent_breakdown = {}
            for execution in executions:
                aid = execution["agent_id"]
                if aid not in agent_breakdown:
                    agent_breakdown[aid] = {
                        "total": 0,
                        "successful": 0,
                        "avg_time": 0
                    }
                
                agent_breakdown[aid]["total"] += 1
                if execution["success"]:
                    agent_breakdown[aid]["successful"] += 1
            
            # Calculate success rates for each agent
            for aid in agent_breakdown:
                stats = agent_breakdown[aid]
                stats["success_rate"] = stats["successful"] / stats["total"] if stats["total"] > 0 else 0
                
                agent_executions = [e for e in executions if e["agent_id"] == aid]
                stats["avg_time"] = sum(e["processing_time_ms"] or 0 for e in agent_executions) / len(agent_executions)
            
            return {
                "total_executions": total,
                "success_rate": successful / total,
                "avg_processing_time_ms": avg_time,
                "agent_breakdown": agent_breakdown
            }
            
        except Exception as error:
            print(f"Failed to get agent stats: {error}")
            return {
                "total_executions": 0,
                "success_rate": 0,
                "avg_processing_time_ms": 0,
                "agent_breakdown": {},
                "error": str(error)
            }
