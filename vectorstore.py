"""
Vector store module using FAISS
"""
import os
from typing import List, Optional, Tuple
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from utils import get_logger

logger = get_logger(__name__)


class VectorStoreManager:
    """Manage FAISS vector store for document retrieval"""
    
    def __init__(self, persist_directory: str = "./vectorstore"):
        """
        Initialize VectorStoreManager
        
        Args:
            persist_directory: Directory to persist vector store
        """
        self.persist_directory = persist_directory
        self.vectorstore: Optional[FAISS] = None
        
        # Create directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        logger.info(f"VectorStoreManager initialized with directory: {persist_directory}")
    
    def create_vectorstore(self, 
                          documents: List[Document], 
                          embeddings: HuggingFaceEmbeddings) -> FAISS:
        """
        Create a new vector store from documents
        
        Args:
            documents: List of Document objects
            embeddings: Embedding model instance
            
        Returns:
            FAISS vector store
        """
        try:
            logger.info(f"Creating vector store with {len(documents)} documents...")
            self.vectorstore = FAISS.from_documents(documents, embeddings)
            logger.info("Vector store created successfully")
            return self.vectorstore
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            raise
    
    def save_vectorstore(self, name: str = "index"):
        """
        Save vector store to disk
        
        Args:
            name: Name of the index
        """
        if self.vectorstore is None:
            logger.warning("No vector store to save")
            return
        
        try:
            save_path = os.path.join(self.persist_directory, name)
            self.vectorstore.save_local(save_path)
            logger.info(f"Vector store saved to {save_path}")
        except Exception as e:
            logger.error(f"Error saving vector store: {e}")
            raise
    
    def load_vectorstore(self, 
                        embeddings: HuggingFaceEmbeddings,
                        name: str = "index") -> Optional[FAISS]:
        """
        Load vector store from disk
        
        Args:
            embeddings: Embedding model instance
            name: Name of the index
            
        Returns:
            FAISS vector store or None if not found
        """
        try:
            load_path = os.path.join(self.persist_directory, name)
            if os.path.exists(load_path):
                self.vectorstore = FAISS.load_local(
                    load_path, 
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info(f"Vector store loaded from {load_path}")
                return self.vectorstore
            else:
                logger.warning(f"No vector store found at {load_path}")
                return None
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            return None
    
    def search_vectorstore(self, 
                          query: str, 
                          top_k: int = 5) -> List[Tuple[Document, float]]:
        """
        Search vector store for similar documents
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of tuples (Document, similarity_score)
        """
        if self.vectorstore is None:
            logger.warning("No vector store available for search")
            return []
        
        try:
            # Use similarity_search_with_score for relevance scores
            results = self.vectorstore.similarity_search_with_score(query, k=top_k)
            logger.info(f"Found {len(results)} results for query")
            return results
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            return []
    
    def add_documents(self, documents: List[Document]):
        """
        Add new documents to existing vector store
        
        Args:
            documents: List of Document objects
        """
        if self.vectorstore is None:
            logger.warning("No vector store available to add documents")
            return
        
        try:
            self.vectorstore.add_documents(documents)
            logger.info(f"Added {len(documents)} documents to vector store")
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise
    
    def get_document_count(self) -> int:
        """
        Get total number of documents in vector store
        
        Returns:
            Document count
        """
        if self.vectorstore is None:
            return 0
        
        try:
            return self.vectorstore.index.ntotal
        except Exception as e:
            logger.error(f"Error getting document count: {e}")
            return 0
    
    def clear_vectorstore(self):
        """Clear the current vector store"""
        self.vectorstore = None
        logger.info("Vector store cleared")
    
    def delete_vectorstore(self, name: str = "index"):
        """
        Delete vector store from disk
        
        Args:
            name: Name of the index
        """
        try:
            import shutil
            delete_path = os.path.join(self.persist_directory, name)
            if os.path.exists(delete_path):
                shutil.rmtree(delete_path)
                logger.info(f"Vector store deleted from {delete_path}")
        except Exception as e:
            logger.error(f"Error deleting vector store: {e}")


def create_vectorstore(documents: List[Document], 
                      embeddings: HuggingFaceEmbeddings) -> FAISS:
    """
    Convenience function to create vector store
    
    Args:
        documents: List of Document objects
        embeddings: Embedding model instance
        
    Returns:
        FAISS vector store
    """
    manager = VectorStoreManager()
    return manager.create_vectorstore(documents, embeddings)