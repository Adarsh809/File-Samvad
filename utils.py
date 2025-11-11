"""
Utility functions for the Agentic RAG Chatbot
"""
import re
import logging
from typing import List, Dict, Any
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance"""
    return logging.getLogger(name)

def clean_text(text: str) -> str:
    """
    Clean and normalize text
    
    Args:
        text: Raw text string
        
    Returns:
        Cleaned text string
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\-\:\;\(\)]', '', text)
    
    # Remove multiple punctuation
    text = re.sub(r'([\.!\?]){2,}', r'\1', text)
    
    return text.strip()

def calculate_similarity_score(query_embedding: List[float], 
                               doc_embedding: List[float]) -> float:
    """
    Calculate cosine similarity between two embeddings
    
    Args:
        query_embedding: Query vector
        doc_embedding: Document vector
        
    Returns:
        Similarity score (0-1)
    """
    query_vec = np.array(query_embedding)
    doc_vec = np.array(doc_embedding)
    
    # Cosine similarity
    similarity = np.dot(query_vec, doc_vec) / (
        np.linalg.norm(query_vec) * np.linalg.norm(doc_vec)
    )
    
    return float(similarity)

def format_sources(sources: List[Dict[str, Any]]) -> str:
    """
    Format source information for display
    
    Args:
        sources: List of source dictionaries
        
    Returns:
        Formatted source string
    """
    if not sources:
        return "No sources available"
    
    formatted = []
    for idx, source in enumerate(sources, 1):
        source_type = source.get('type', 'unknown')
        content = source.get('content', '')[:100] + '...'
        
        if source_type == 'file':
            metadata = source.get('metadata', {})
            filename = metadata.get('source', 'Unknown file')
            formatted.append(f"{idx}. 📄 {filename}: {content}")
        elif source_type == 'web':
            url = source.get('url', 'Unknown URL')
            formatted.append(f"{idx}. 🌐 {url}: {content}")
    
    return "\n".join(formatted)

def truncate_text(text: str, max_length: int = 1000) -> str:
    """
    Truncate text to maximum length
    
    Args:
        text: Input text
        max_length: Maximum character length
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."

def chunk_text_by_sentences(text: str, chunk_size: int = 500, 
                            overlap: int = 50) -> List[str]:
    """
    Chunk text by sentences with overlap
    
    Args:
        text: Input text
        chunk_size: Target chunk size in characters
        overlap: Overlap between chunks
        
    Returns:
        List of text chunks
    """
    # Split by sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence)
        
        if current_length + sentence_length > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            # Keep last few sentences for overlap
            overlap_sentences = []
            overlap_length = 0
            for s in reversed(current_chunk):
                if overlap_length + len(s) <= overlap:
                    overlap_sentences.insert(0, s)
                    overlap_length += len(s)
                else:
                    break
            current_chunk = overlap_sentences
            current_length = overlap_length
        
        current_chunk.append(sentence)
        current_length += sentence_length
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def create_context_window(chunks: List[str], max_chunks: int = 5) -> str:
    """
    Create a context window from multiple chunks
    
    Args:
        chunks: List of text chunks
        max_chunks: Maximum number of chunks to include
        
    Returns:
        Combined context string
    """
    selected_chunks = chunks[:max_chunks]
    return "\n\n".join(selected_chunks)

def estimate_tokens(text: str) -> int:
    """
    Estimate token count (rough approximation)
    
    Args:
        text: Input text
        
    Returns:
        Estimated token count
    """
    # Rough estimate: 1 token ≈ 4 characters
    return len(text) // 4

def validate_file_type(filename: str) -> bool:
    """
    Validate if file type is supported
    
    Args:
        filename: Name of the file
        
    Returns:
        True if supported, False otherwise
    """
    supported_extensions = ['.pdf', '.docx', '.txt', '.csv', '.pptx', '.xlsx']
    return any(filename.lower().endswith(ext) for ext in supported_extensions)

def get_file_extension(filename: str) -> str:
    """
    Get file extension
    
    Args:
        filename: Name of the file
        
    Returns:
        File extension (lowercase)
    """
    return filename.lower().split('.')[-1] if '.' in filename else ''