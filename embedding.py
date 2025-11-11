"""
Embedding module using HuggingFace models
"""
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import Optional
from utils import get_logger

logger = get_logger(__name__)


class EmbeddingManager:
    """Manage embeddings using HuggingFace models"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize EmbeddingManager
        
        Args:
            model_name: HuggingFace model name for embeddings
        """
        self.model_name = model_name
        self._embeddings: Optional[HuggingFaceEmbeddings] = None
        logger.info(f"EmbeddingManager initialized with model: {model_name}")
    
    def get_embeddings(self) -> HuggingFaceEmbeddings:
        """
        Get or create embeddings instance
        
        Returns:
            HuggingFaceEmbeddings instance
        """
        if self._embeddings is None:
            logger.info("Loading embedding model...")
            try:
                self._embeddings = HuggingFaceEmbeddings(
                    model_name=self.model_name,
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                logger.info("Embedding model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading embedding model: {e}")
                raise
        
        return self._embeddings
    
    def embed_query(self, query: str) -> list:
        """
        Embed a single query
        
        Args:
            query: Query text
            
        Returns:
            Query embedding vector
        """
        embeddings = self.get_embeddings()
        try:
            return embeddings.embed_query(query)
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            raise
    
    def embed_documents(self, documents: list) -> list:
        """
        Embed multiple documents
        
        Args:
            documents: List of document texts
            
        Returns:
            List of document embedding vectors
        """
        embeddings = self.get_embeddings()
        try:
            return embeddings.embed_documents(documents)
        except Exception as e:
            logger.error(f"Error embedding documents: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings
        
        Returns:
            Embedding dimension
        """
        # For all-MiniLM-L6-v2, dimension is 384
        model_dimensions = {
            "sentence-transformers/all-MiniLM-L6-v2": 384,
            "sentence-transformers/all-mpnet-base-v2": 768,
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 384
        }
        return model_dimensions.get(self.model_name, 384)


def get_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:
    """
    Convenience function to get embeddings
    
    Args:
        model_name: HuggingFace model name
        
    Returns:
        HuggingFaceEmbeddings instance
    """
    manager = EmbeddingManager(model_name)
    return manager.get_embeddings()