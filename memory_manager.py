"""
Memory management module for conversation history
"""
from typing import Dict, Optional
from langchain.memory.buffer import ConversationBufferMemory
from utils import get_logger

logger = get_logger(__name__)


class MemoryManager:
    """Manage conversation memory for chat sessions"""
    
    def __init__(self):
        """Initialize MemoryManager"""
        self.sessions: Dict[str, ConversationBufferMemory] = {}
        logger.info("MemoryManager initialized")
    
    def get_memory(self, session_id: str) -> ConversationBufferMemory:
        """
        Get or create memory for a session
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            ConversationBufferMemory instance for the session
        """
        if session_id not in self.sessions:
            logger.info(f"Creating new memory for session: {session_id}")
            self.sessions[session_id] = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="output"
            )
        
        return self.sessions[session_id]
    
    def add_message(self, session_id: str, user_input: str, ai_output: str):
        """
        Add a conversation turn to memory
        
        Args:
            session_id: Session identifier
            user_input: User's message
            ai_output: AI's response
        """
        memory = self.get_memory(session_id)
        try:
            memory.save_context(
                {"input": user_input},
                {"output": ai_output}
            )
            logger.info(f"Added message to session {session_id}")
        except Exception as e:
            logger.error(f"Error adding message to memory: {e}")
    
    def get_chat_history(self, session_id: str) -> str:
        """
        Get formatted chat history for a session
        
        Args:
            session_id: Session identifier
            
        Returns:
            Formatted chat history string
        """
        memory = self.get_memory(session_id)
        try:
            history = memory.load_memory_variables({})
            messages = history.get("chat_history", [])
            
            # Format messages
            formatted_history = []
            for msg in messages:
                role = "Human" if msg.type == "human" else "AI"
                formatted_history.append(f"{role}: {msg.content}")
            
            return "\n".join(formatted_history)
        except Exception as e:
            logger.error(f"Error getting chat history: {e}")
            return ""
    
    def get_recent_messages(self, session_id: str, n: int = 5) -> str:
        """
        Get recent N messages from chat history
        
        Args:
            session_id: Session identifier
            n: Number of recent messages to retrieve
            
        Returns:
            Formatted recent messages
        """
        memory = self.get_memory(session_id)
        try:
            history = memory.load_memory_variables({})
            messages = history.get("chat_history", [])
            
            # Get last N messages
            recent_messages = messages[-n:] if len(messages) > n else messages
            
            formatted_history = []
            for msg in recent_messages:
                role = "Human" if msg.type == "human" else "AI"
                formatted_history.append(f"{role}: {msg.content}")
            
            return "\n".join(formatted_history)
        except Exception as e:
            logger.error(f"Error getting recent messages: {e}")
            return ""
    
    def clear_session(self, session_id: str):
        """
        Clear memory for a specific session
        
        Args:
            session_id: Session identifier
        """
        if session_id in self.sessions:
            self.sessions[session_id].clear()
            logger.info(f"Cleared memory for session: {session_id}")
    
    def delete_session(self, session_id: str):
        """
        Delete a session completely
        
        Args:
            session_id: Session identifier
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Deleted session: {session_id}")
    
    def get_session_count(self) -> int:
        """
        Get number of active sessions
        
        Returns:
            Number of sessions
        """
        return len(self.sessions)
    
    def get_message_count(self, session_id: str) -> int:
        """
        Get number of messages in a session
        
        Args:
            session_id: Session identifier
            
        Returns:
            Number of messages
        """
        memory = self.get_memory(session_id)
        try:
            history = memory.load_memory_variables({})
            messages = history.get("chat_history", [])
            return len(messages)
        except Exception as e:
            logger.error(f"Error getting message count: {e}")
            return 0
    
    def export_session(self, session_id: str) -> Dict:
        """
        Export session data
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dictionary with session data
        """
        memory = self.get_memory(session_id)
        try:
            history = memory.load_memory_variables({})
            messages = history.get("chat_history", [])
            
            exported_data = {
                "session_id": session_id,
                "message_count": len(messages),
                "messages": [
                    {
                        "role": "human" if msg.type == "human" else "ai",
                        "content": msg.content
                    }
                    for msg in messages
                ]
            }
            return exported_data
        except Exception as e:
            logger.error(f"Error exporting session: {e}")
            return {"session_id": session_id, "message_count": 0, "messages": []}


# Global memory manager instance
_memory_manager = MemoryManager()


def get_memory(session_id: str) -> ConversationBufferMemory:
    """
    Convenience function to get memory for a session
    
    Args:
        session_id: Session identifier
        
    Returns:
        ConversationBufferMemory instance
    """
    return _memory_manager.get_memory(session_id)