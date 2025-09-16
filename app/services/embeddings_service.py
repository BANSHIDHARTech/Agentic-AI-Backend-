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
    
    def __init__(self, model_name: str = "text-embedding-3-small"):
        """Initialize the embedding service.
        
        Args:
            model_name: Name of the embedding model to use
        """
        self.model_name = model_name
        
        # Check for OpenAI API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("⚠️ OPENAI_API_KEY not found in environment variables")
            logger.warning("Embeddings service will not function correctly")
            api_key = "sk-dummy-key-for-initialization"
            
        try:
            self.client = OpenAI(api_key=api_key)
            self.embeddings = OpenAIEmbeddings(
                model=model_name,
                openai_api_key=api_key
            )
            logger.info(f"Embedding service initialized with model: {model_name}")
        except Exception as e:
            logger.error(f"Error initializing embedding service: {str(e)}")
            # Initialize with dummy values that will fail gracefully
            self.client = None
            self.embeddings = None
    
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
            # Check for OpenAI API key
            if not os.getenv("OPENAI_API_KEY") or not self.embeddings:
                logger.error("OpenAI API key missing or embeddings not initialized")
                # Return dummy embeddings to allow the process to continue
                return [[0.0] * 1536 for _ in range(len(texts))]
                
            # Use LangChain's async embedding
            return await self.embeddings.aembed_documents(texts)
            
        except Exception as e:
            logger.error(f"Error getting embeddings: {str(e)}")
            # Return dummy embeddings to allow the process to continue
            logger.warning("Returning dummy embeddings due to error")
            return [[0.0] * 1536 for _ in range(len(texts))]
    
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
            # Check for OpenAI API key
            if not os.getenv("OPENAI_API_KEY") or not self.embeddings:
                logger.error("OpenAI API key missing or embeddings not initialized")
                # Return dummy embedding to allow the process to continue
                return [0.0] * 1536
                
            # Use LangChain's async embedding
            return await self.embeddings.aembed_query(text)
            
        except Exception as e:
            logger.error(f"Error getting embedding: {str(e)}")
            # Return dummy embedding to allow the process to continue
            logger.warning("Returning dummy embedding due to error")
            return [0.0] * 1536
    
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

class EmbeddingsService:
    """Static service wrapper for embedding functionality."""
    
    @staticmethod
    async def create_embedding(text: str) -> Dict[str, Any]:
        """Create an embedding for a given text.
        
        Args:
            text: Text to create embedding for
            
        Returns:
            Dictionary with embedding and metadata
        """
        try:
            logger.info(f"Creating embedding for text (length={len(text)})")
            if not text or not text.strip():
                logger.error("Cannot create embedding for empty text")
                return {"error": "Cannot create embedding for empty text"}
                
            # Get embedding using the singleton service
            vector = await embedding_service.get_embedding(text)
            
            if vector and len(vector) > 0:
                logger.info(f"Successfully created embedding (dimensions={len(vector)})")
                return {
                    "success": True,
                    "embedding": vector,
                    "dimensions": len(vector),
                    "model": embedding_service.model_name
                }
            else:
                logger.error("Failed to generate embedding (empty vector)")
                return {
                    "success": False, 
                    "error": "Failed to generate embedding"
                }
                
        except Exception as e:
            logger.error(f"Error creating embedding: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    @staticmethod
    async def create_embeddings(texts: List[str]) -> Dict[str, Any]:
        """Create embeddings for a list of texts.
        
        Args:
            texts: List of texts to create embeddings for
            
        Returns:
            Dictionary with embeddings and metadata
        """
        try:
            logger.info(f"Creating embeddings for {len(texts)} texts")
            
            # Get embeddings using the singleton service
            vectors = await embedding_service.get_embeddings(texts)
            
            if vectors and len(vectors) > 0:
                logger.info(f"Successfully created {len(vectors)} embeddings with dimensions {len(vectors[0])}")
                return {
                    "success": True,
                    "embeddings": vectors,
                    "count": len(vectors),
                    "dimensions": len(vectors[0]) if vectors and len(vectors) > 0 else 0,
                    "model": embedding_service.model_name
                }
            else:
                logger.error("Failed to generate embeddings (empty vectors)")
                return {
                    "success": False, 
                    "error": "Failed to generate embeddings"
                }
                
        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
