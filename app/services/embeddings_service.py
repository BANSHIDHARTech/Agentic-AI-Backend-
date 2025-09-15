import os
from typing import List, Dict, Any, Optional
import asyncio
import numpy as np
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings.base import Embeddings
import logging
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Service for generating and managing text embeddings."""
    
    def __init__(self, model_name: str = "text-embedding-ada-002"):
        """Initialize the embedding service.
        
        Args:
            model_name: Name of the embedding model to use
        """
        self.model_name = model_name
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.embeddings = OpenAIEmbeddings(
            model=model_name,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception)
    )
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            # Use LangChain's async embedding
            return await self.embeddings.aembed_documents(texts)
            
        except Exception as e:
            logger.error(f"Error getting embeddings: {str(e)}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception)
    )
    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text.
        
        Args:
            text: Input text to embed
            
        Returns:
            Embedding vector
        """
        try:
            # Use LangChain's async embedding
            return await self.embeddings.aembed_query(text)
            
        except Exception as e:
            logger.error(f"Error getting embedding: {str(e)}")
            raise
    
    async def get_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        try:
            emb1, emb2 = await asyncio.gather(
                self.get_embedding(text1),
                self.get_embedding(text2)
            )
            return self._cosine_similarity(emb1, emb2)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            raise
    
    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm == 0 or b_norm == 0:
            return 0.0
        return np.dot(a, b) / (a_norm * b_norm)

# Singleton instance
embedding_service = EmbeddingService()
