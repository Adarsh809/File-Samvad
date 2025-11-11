"""
Search module for document retrieval and web search
"""
from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import FAISS
from duckduckgo_search import DDGS
from utils import get_logger

logger = get_logger(__name__)


class SearchManager:
    """Manage document and web search"""
    
    def __init__(self, relevance_threshold: float = 0.5):
        """
        Initialize SearchManager
        
        Args:
            relevance_threshold: Minimum similarity score for relevance
        """
        self.relevance_threshold = relevance_threshold
        logger.info(f"SearchManager initialized with threshold: {relevance_threshold}")
    
    def search_docs(self, 
                   query: str, 
                   vectorstore: FAISS,
                   top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search documents in vector store
        
        Args:
            query: Search query
            vectorstore: FAISS vector store
            top_k: Number of results to return
            
        Returns:
            List of result dictionaries with content, metadata, and scores
        """
        try:
            results = vectorstore.similarity_search_with_score(query, k=top_k)
            
            formatted_results = []
            for doc, score in results:
                # FAISS returns distance, convert to similarity (lower distance = higher similarity)
                similarity = 1 / (1 + score)  # Normalize distance to 0-1 range
                
                result = {
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'score': similarity,
                    'type': 'file',
                    'source': doc.metadata.get('source', 'Unknown')
                }
                formatted_results.append(result)
            
            logger.info(f"Found {len(formatted_results)} document results")
            return formatted_results
        
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    def check_relevance(self, results: List[Dict[str, Any]]) -> bool:
        """
        Check if search results are relevant based on threshold
        
        Args:
            results: List of search results with scores
            
        Returns:
            True if results are relevant, False otherwise
        """
        if not results:
            return False
        
        # Check if best result meets threshold
        best_score = max(r['score'] for r in results)
        is_relevant = best_score >= self.relevance_threshold
        
        logger.info(f"Best score: {best_score:.3f}, Threshold: {self.relevance_threshold}, Relevant: {is_relevant}")
        return is_relevant
    
    def web_search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Perform web search using DuckDuckGo
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of web search results
        """
        try:
            logger.info(f"Performing web search for: {query}")
            
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            
            formatted_results = []
            for result in results:
                formatted_result = {
                    'content': result.get('body', ''),
                    'title': result.get('title', ''),
                    'url': result.get('href', ''),
                    'type': 'web',
                    'source': 'DuckDuckGo',
                    'score': 1.0  # Web results don't have similarity scores
                }
                formatted_results.append(formatted_result)
            
            logger.info(f"Found {len(formatted_results)} web results")
            return formatted_results
        
        except Exception as e:
            logger.error(f"Error performing web search: {e}")
            return []
    
    def hybrid_search(self, 
                     query: str,
                     vectorstore: Optional[FAISS] = None,
                     doc_top_k: int = 5,
                     web_max_results: int = 3) -> Dict[str, Any]:
        """
        Perform hybrid search: try document search first, fallback to web
        
        Args:
            query: Search query
            vectorstore: FAISS vector store (optional)
            doc_top_k: Number of document results
            web_max_results: Number of web results
            
        Returns:
            Dictionary with search results and metadata
        """
        search_result = {
            'query': query,
            'doc_results': [],
            'web_results': [],
            'used_web_search': False,
            'is_relevant': False
        }
        
        # Try document search first
        if vectorstore is not None:
            doc_results = self.search_docs(query, vectorstore, doc_top_k)
            search_result['doc_results'] = doc_results
            search_result['is_relevant'] = self.check_relevance(doc_results)
        
        # If not relevant, perform web search
        if not search_result['is_relevant']:
            logger.info("Document results not relevant, performing web search...")
            web_results = self.web_search(query, web_max_results)
            search_result['web_results'] = web_results
            search_result['used_web_search'] = True
        
        return search_result
    
    def format_context(self, search_result: Dict[str, Any]) -> str:
        """
        Format search results into context string
        
        Args:
            search_result: Dictionary with search results
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        # Add document results
        if search_result['doc_results']:
            context_parts.append("=== Information from uploaded documents ===")
            for idx, result in enumerate(search_result['doc_results'][:3], 1):
                source = result['metadata'].get('source', 'Unknown')
                content = result['content'][:500]  # Limit content length
                context_parts.append(f"\n[Document {idx} - {source}]\n{content}")
        
        # Add web results
        if search_result['web_results']:
            context_parts.append("\n\n=== Information from web search ===")
            for idx, result in enumerate(search_result['web_results'][:3], 1):
                title = result['title']
                url = result['url']
                content = result['content'][:500]  # Limit content length
                context_parts.append(f"\n[Web Result {idx} - {title}]\nURL: {url}\n{content}")
        
        return "\n".join(context_parts)


def search_docs(query: str, vectorstore: FAISS, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Convenience function to search documents
    
    Args:
        query: Search query
        vectorstore: FAISS vector store
        top_k: Number of results
        
    Returns:
        List of search results
    """
    manager = SearchManager()
    return manager.search_docs(query, vectorstore, top_k)


def web_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Convenience function for web search
    
    Args:
        query: Search query
        max_results: Maximum results
        
    Returns:
        List of web search results
    """
    manager = SearchManager()
    return manager.web_search(query, max_results)