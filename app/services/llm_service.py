import os
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime

from openai import AsyncOpenAI
import anthropic
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_cohere import ChatCohere

from ..core.database import supabase

class LLMService:
    """Multi-Provider LLM Integration with Fallback
    
    Supports OpenAI, Cohere, and Anthropic with automatic fallback
    and comprehensive performance logging.
    """
    
    providers: Dict[str, Any] = {
        "openai": None,
        "cohere": None,
        "anthropic": None
    }
    
    initialized: bool = False
    
    @classmethod
    async def initialize(cls):
        """Initialize all LLM providers"""
        if cls.initialized:
            return
        
        try:
            # Initialize OpenAI
            if os.getenv("OPENAI_API_KEY"):
                cls.providers["openai"] = ChatOpenAI(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    model="gpt-4"
                )
                print("✅ OpenAI initialized")
            
            # Initialize Cohere
            if os.getenv("COHERE_API_KEY"):
                cls.providers["cohere"] = ChatCohere(
                    cohere_api_key=os.getenv("COHERE_API_KEY"),
                    model="command-r-plus"
                )
                print("✅ Cohere initialized")
            
            # Initialize Anthropic
            if os.getenv("ANTHROPIC_API_KEY"):
                cls.providers["anthropic"] = ChatAnthropic(
                    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
                    model="claude-3-sonnet-20240229"
                )
                print("✅ Anthropic initialized")
            
            cls.initialized = True
            
            if not any(cls.providers.values()):
                print("⚠️  No LLM providers configured. Please set API keys in environment variables.")
            
        except Exception as error:
            print(f"❌ LLM Service initialization error: {error}")
            raise error
    
    @classmethod
    async def generate_completion(
        cls,
        prompt: str,
        model_name: str = "gpt-4",
        max_tokens: int = 1000,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        provider_preference: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate completion with fallback support"""
        
        if not cls.initialized:
            await cls.initialize()
        
        start_time = datetime.now()
        
        # Determine provider order
        provider_order = cls._get_provider_order(model_name, provider_preference)
        
        last_error = None
        
        for provider_name in provider_order:
            provider = cls.providers.get(provider_name)
            if not provider:
                continue
            
            try:
                result = await cls._call_provider(
                    provider=provider,
                    provider_name=provider_name,
                    prompt=prompt,
                    model_name=model_name,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system_prompt=system_prompt
                )
                
                # Log successful completion
                processing_time = (datetime.now() - start_time).total_seconds() * 1000
                await cls._log_completion(
                    provider_name=provider_name,
                    model_name=model_name,
                    prompt=prompt,
                    response=result["content"],
                    processing_time_ms=processing_time,
                    success=True
                )
                
                return {
                    "content": result["content"],
                    "provider": provider_name,
                    "model": model_name,
                    "processing_time_ms": processing_time,
                    "usage": result.get("usage", {}),
                    "success": True
                }
                
            except Exception as error:
                last_error = error
                print(f"❌ {provider_name} failed: {error}")
                continue
        
        # All providers failed
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        await cls._log_completion(
            provider_name="all_providers",
            model_name=model_name,
            prompt=prompt,
            response="",
            processing_time_ms=processing_time,
            success=False,
            error_message=str(last_error)
        )
        
        raise Exception(f"All LLM providers failed. Last error: {last_error}")
    
    @classmethod
    async def _call_provider(
        cls,
        provider: Any,
        provider_name: str,
        prompt: str,
        model_name: str,
        max_tokens: int,
        temperature: float,
        system_prompt: Optional[str]
    ) -> Dict[str, Any]:
        """Call specific provider"""
        
        if provider_name == "openai":
            return await cls._call_openai(provider, prompt, system_prompt, max_tokens, temperature)
        elif provider_name == "anthropic":
            return await cls._call_anthropic(provider, prompt, system_prompt, max_tokens, temperature)
        elif provider_name == "cohere":
            return await cls._call_cohere(provider, prompt, system_prompt, max_tokens, temperature)
        else:
            raise Exception(f"Unknown provider: {provider_name}")
    
    @classmethod
    async def _call_openai(
        cls,
        provider: ChatOpenAI,
        prompt: str,
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float
    ) -> Dict[str, Any]:
        """Call OpenAI provider"""
        from langchain.schema import HumanMessage, SystemMessage
        
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        
        response = await provider.ainvoke(messages)
        
        return {
            "content": response.content,
            "usage": getattr(response, "usage_metadata", {})
        }
    
    @classmethod
    async def _call_anthropic(
        cls,
        provider: ChatAnthropic,
        prompt: str,
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float
    ) -> Dict[str, Any]:
        """Call Anthropic provider"""
        from langchain.schema import HumanMessage, SystemMessage
        
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        
        response = await provider.ainvoke(messages)
        
        return {
            "content": response.content,
            "usage": getattr(response, "usage_metadata", {})
        }
    
    @classmethod
    async def _call_cohere(
        cls,
        provider: ChatCohere,
        prompt: str,
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float
    ) -> Dict[str, Any]:
        """Call Cohere provider"""
        from langchain.schema import HumanMessage, SystemMessage
        
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        
        response = await provider.ainvoke(messages)
        
        return {
            "content": response.content,
            "usage": getattr(response, "usage_metadata", {})
        }
    
    @classmethod
    def _get_provider_order(cls, model_name: str, provider_preference: Optional[str]) -> List[str]:
        """Determine provider order based on model and preference"""
        
        # If specific provider requested
        if provider_preference and provider_preference in cls.providers:
            available_providers = [p for p in cls.providers.keys() if cls.providers[p] is not None]
            if provider_preference in available_providers:
                # Put preferred provider first
                order = [provider_preference]
                order.extend([p for p in available_providers if p != provider_preference])
                return order
        
        # Default order based on model
        if "gpt" in model_name.lower():
            return ["openai", "anthropic", "cohere"]
        elif "claude" in model_name.lower():
            return ["anthropic", "openai", "cohere"]
        elif "command" in model_name.lower():
            return ["cohere", "openai", "anthropic"]
        else:
            return ["openai", "anthropic", "cohere"]
    
    @classmethod
    async def _log_completion(
        cls,
        provider_name: str,
        model_name: str,
        prompt: str,
        response: str,
        processing_time_ms: float,
        success: bool,
        error_message: Optional[str] = None
    ):
        """Log completion to database"""
        try:
            log_data = {
                "provider_name": provider_name,
                "model_name": model_name,
                "prompt_length": len(prompt),
                "response_length": len(response),
                "processing_time_ms": processing_time_ms,
                "success": success,
                "error_message": error_message,
                "created_at": datetime.now().isoformat()
            }
            
            result = supabase.from_("llm_completions").insert(log_data).execute()
            if result.error:
                print(f"Failed to log completion: {result.error}")
                
        except Exception as error:
            print(f"Failed to log completion: {error}")
    
    @classmethod
    async def get_completion_stats(cls) -> Dict[str, Any]:
        """Get completion statistics"""
        try:
            # Get overall stats
            result = supabase.from_("llm_completions").select("*").execute()
            if result.error:
                raise Exception(result.error)
            
            completions = result.data
            
            if not completions:
                return {
                    "total_completions": 0,
                    "success_rate": 0,
                    "avg_processing_time_ms": 0,
                    "provider_stats": {}
                }
            
            total = len(completions)
            successful = len([c for c in completions if c["success"]])
            avg_time = sum(c["processing_time_ms"] for c in completions) / total
            
            # Provider stats
            provider_stats = {}
            for completion in completions:
                provider = completion["provider_name"]
                if provider not in provider_stats:
                    provider_stats[provider] = {
                        "total": 0,
                        "successful": 0,
                        "avg_time": 0
                    }
                
                provider_stats[provider]["total"] += 1
                if completion["success"]:
                    provider_stats[provider]["successful"] += 1
            
            # Calculate success rates and avg times for each provider
            for provider in provider_stats:
                stats = provider_stats[provider]
                stats["success_rate"] = stats["successful"] / stats["total"] if stats["total"] > 0 else 0
                
                provider_completions = [c for c in completions if c["provider_name"] == provider]
                stats["avg_time"] = sum(c["processing_time_ms"] for c in provider_completions) / len(provider_completions)
            
            return {
                "total_completions": total,
                "success_rate": successful / total,
                "avg_processing_time_ms": avg_time,
                "provider_stats": provider_stats
            }
            
        except Exception as error:
            print(f"Failed to get completion stats: {error}")
            return {
                "total_completions": 0,
                "success_rate": 0,
                "avg_processing_time_ms": 0,
                "provider_stats": {},
                "error": str(error)
            }
