from typing import Dict, Any, List, Optional
from ..core.database import supabase, db_insert, db_update, db_select, db_delete
from ..core.models import AgentModel
from .llm_service import LLMService

class AgentService:
    """Agent management service for CRUD operations and execution"""
    
    @classmethod
    async def create_agent(cls, agent_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new agent"""
        # Validate using Pydantic model
        validated_data = AgentModel(**agent_data).dict()
        
        result = await db_insert("agents", validated_data)
        return result[0] if result else None
    
    @classmethod
    async def update_agent(cls, agent_id: str, agent_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing agent"""
        # Validate using Pydantic model
        validated_data = AgentModel(**agent_data).dict()
        
        result = await db_update("agents", agent_id, validated_data)
        return result[0] if result else None
    
    @classmethod
    async def get_agent(cls, agent_id: str) -> Dict[str, Any]:
        """Get agent by ID"""
        result = supabase.from_("agents").select("*").eq("id", agent_id).execute()
        
        if result.error:
            raise Exception(result.error)
        
        if not result.data:
            raise Exception(f"Agent with ID {agent_id} not found")
        
        return result.data[0]
    
    @classmethod
    async def get_agent_by_name(cls, name: str) -> Dict[str, Any]:
        """Get agent by name"""
        result = supabase.from_("agents").select("*").eq("name", name).execute()
        
        if result.error:
            raise Exception(result.error)
        
        if not result.data:
            raise Exception(f"Agent with name {name} not found")
        
        return result.data[0]
    
    @classmethod
    async def get_all_agents(cls) -> List[Dict[str, Any]]:
        """Get all agents"""
        result = supabase.from_("agents").select("*").eq("is_active", True).order("created_at", desc=True).execute()
        
        if result.error:
            raise Exception(result.error)
        
        return result.data or []
    
    @classmethod
    async def delete_agent(cls, agent_id: str) -> bool:
        """Delete an agent"""
        return await db_delete("agents", agent_id)
    
    @classmethod
    async def find_agent_by_intent(cls, intent: str) -> List[Dict[str, Any]]:
        """Find agents that can handle a specific intent"""
        result = supabase.from_("agents").select("*").contains("input_intents", [intent]).eq("is_active", True).execute()
        
        if result.error:
            raise Exception(result.error)
        
        return result.data or []
    
    @classmethod
    async def execute_agent(cls, agent: Dict[str, Any], input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an agent with given input"""
        try:
            # Initialize LLM service if not already done
            await LLMService.initialize()
            
            # Prepare the prompt
            user_input = input_data.get("input", "")
            context = input_data.get("context", {})
            
            # Build the prompt with context
            prompt = cls._build_agent_prompt(agent, user_input, context)
            
            # Generate completion using LLM service
            completion = await LLMService.generate_completion(
                prompt=prompt,
                model_name=agent.get("model_name", "gpt-4"),
                system_prompt=agent.get("system_prompt"),
                max_tokens=input_data.get("max_tokens", 1000),
                temperature=input_data.get("temperature", 0.7)
            )
            
            # Log the execution
            await cls._log_agent_execution(
                agent_id=agent["id"],
                input_data=input_data,
                output_data=completion,
                success=True
            )
            
            return {
                "agent_id": agent["id"],
                "agent_name": agent["name"],
                "input": user_input,
                "output": completion["content"],
                "model_used": completion["model"],
                "provider_used": completion["provider"],
                "processing_time_ms": completion["processing_time_ms"],
                "success": True,
                "context": context
            }
            
        except Exception as error:
            # Log the failed execution
            await cls._log_agent_execution(
                agent_id=agent["id"],
                input_data=input_data,
                output_data={},
                success=False,
                error_message=str(error)
            )
            
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
            log_data = {
                "agent_id": agent_id,
                "input_data": input_data,
                "output_data": output_data,
                "success": success,
                "error_message": error_message,
                "processing_time_ms": output_data.get("processing_time_ms", 0),
                "created_at": "now()"
            }
            
            result = supabase.from_("agent_executions").insert(log_data).execute()
            if result.error:
                print(f"Failed to log agent execution: {result.error}")
                
        except Exception as error:
            print(f"Failed to log agent execution: {error}")
    
    @classmethod
    async def get_agent_stats(cls, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get agent execution statistics"""
        try:
            query = supabase.from_("agent_executions").select("*")
            
            if agent_id:
                query = query.eq("agent_id", agent_id)
            
            result = query.execute()
            
            if result.error:
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
