"""
LangGraph agent workflow for Agentic RAG
"""
from typing import Dict, Any, TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from search import SearchManager
from memory_manager import MemoryManager
from utils import get_logger
import os

logger = get_logger(__name__)


class AgentState(TypedDict):
    """State object for the agent workflow"""
    query: str
    vectorstore: Optional[Any]
    search_results: Dict[str, Any]
    context: str
    answer: str
    source_type: str
    session_id: str
    chat_history: str


class AgenticRAGGraph:
    """LangGraph workflow for Agentic RAG"""
    
    def __init__(self, 
                 groq_api_key: str,
                 model_name: str = "llama-3.1-70b-versatile",
                 relevance_threshold: float = 0.5):
        """
        Initialize AgenticRAGGraph
        
        Args:
            groq_api_key: Groq API key
            model_name: Groq model name
            relevance_threshold: Relevance threshold for document search
        """
        self.groq_api_key = groq_api_key
        self.model_name = model_name
        
        # Initialize components
        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=model_name,
            temperature=0.7
        )
        
        self.search_manager = SearchManager(relevance_threshold=relevance_threshold)
        self.memory_manager = MemoryManager()
        
        # Build the graph
        self.graph = self._build_graph()
        
        logger.info(f"AgenticRAGGraph initialized with model: {model_name}")

    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow
        
        Returns:
            Compiled Graph
        """
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("check_relevance", self.check_relevance_node)
        workflow.add_node("web_search", self.web_search_node)
        workflow.add_node("compose_answer", self.compose_answer_node)
        workflow.add_node("update_memory", self.update_memory_node)
        
        # Define edges
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "check_relevance")
        
        # Conditional edge based on relevance
        workflow.add_conditional_edges(
            "check_relevance",
            self._decide_search_route,
            {
                "compose": "compose_answer",
                "web_search": "web_search"
            }
        )
        
        workflow.add_edge("web_search", "compose_answer")
        workflow.add_edge("compose_answer", "update_memory")
        workflow.add_edge("update_memory", END)
        
        return workflow.compile()
    
    def _decide_search_route(self, state: AgentState) -> str:
        """
        Decide whether to use web search or proceed to answer
        """
        query = state.get("query", "").lower()
        search_results = state.get("search_results", {})
        is_relevant = search_results.get("is_relevant", False)

        # Keywords that indicate real-time or general info → force web search
        web_keywords = [
            "today", "now", "current", "latest", "time", "date",
            "who", "where", "when", "what", "news", "weather",
            "president", "prime minister", "capital", "temperature"
        ]

        if any(word in query for word in web_keywords):
            logger.info("Real-time/general query detected → forcing web search")
            return "web_search"

        if is_relevant:
            logger.info("Documents are relevant → composing answer")
            return "compose"

        logger.info("Documents not relevant → performing web search")
        return "web_search"


    
    def retrieve_node(self, state: AgentState) -> AgentState:
        """
        Node 1: Retrieve relevant documents from vector store
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with search results
        """
        logger.info("Executing retrieve node")
        
        query = state["query"]
        vectorstore = state.get("vectorstore")
        
        if vectorstore is not None:
            # Perform document search
            search_results = self.search_manager.hybrid_search(
                query=query,
                vectorstore=vectorstore,
                doc_top_k=5,
                web_max_results=0  # Don't do web search yet
            )
        else:
            # No documents available
            search_results = {
                'query': query,
                'doc_results': [],
                'web_results': [],
                'used_web_search': False,
                'is_relevant': False
            }
        
        state["search_results"] = search_results
        return state
    
    def check_relevance_node(self, state: AgentState) -> AgentState:
        """
        Node 2: Check if retrieved documents are relevant
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with relevance check
        """
        logger.info("Executing check_relevance node")
        
        search_results = state["search_results"]
        doc_results = search_results.get("doc_results", [])
        
        # Check relevance
        is_relevant = self.search_manager.check_relevance(doc_results)
        state["search_results"]["is_relevant"] = is_relevant
        
        if is_relevant:
            state["source_type"] = "documents"
        
        return state
    
    def web_search_node(self, state: AgentState) -> AgentState:
        """
        Node 3: Perform web search if documents not relevant
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with web search results
        """
        logger.info("Executing web_search node")
        
        query = state["query"]
        
        # Perform web search
        web_results = self.search_manager.web_search(query, max_results=3)
        
        state["search_results"]["web_results"] = web_results
        state["search_results"]["used_web_search"] = True
        state["source_type"] = "web"
        
        return state
    
    def compose_answer_node(self, state: AgentState) -> AgentState:
        """
        Node 4: Compose answer using LLM
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with generated answer
        """
        logger.info("Executing compose_answer node")
        
        query = state["query"]
        search_results = state["search_results"]
        chat_history = state.get("chat_history", "")
        
        # Format context from search results
        context = self.search_manager.format_context(search_results)
        state["context"] = context
        
        # Create prompt
        system_prompt = """You are a smart, confident AI assistant that can use both uploaded documents and web search results to answer user questions accurately.

Rules:
- If the context is from uploaded files, rely primarily on that information.
- If the context includes web search results, clearly use them to provide current or general information.
- If a question is about real-time information (like today's date, current time, news, or people), always use web search context to answer.
- Respond clearly and naturally, without apologies or hesitation."""

        user_prompt = f"""Previous conversation:
{chat_history}

Context:
{context}

Question: {query}

Please provide a clear and helpful answer based on the context above."""
        
        # Generate answer
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm.invoke(messages)
            answer = response.content
            
            state["answer"] = answer
            logger.info("Answer generated successfully")
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            state["answer"] = "I apologize, but I encountered an error while generating the answer."
        
        return state
    
    def update_memory_node(self, state: AgentState) -> AgentState:
        """
        Node 5: Update conversation memory
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state
        """
        logger.info("Executing update_memory node")
        
        session_id = state.get("session_id", "default")
        query = state["query"]
        answer = state["answer"]
        
        # Add to memory
        self.memory_manager.add_message(session_id, query, answer)
        
        logger.info("Memory updated successfully")
        return state
    
    def invoke(self, 
               query: str, 
               vectorstore: Optional[Any] = None,
               session_id: str = "default") -> Dict[str, Any]:
        """
        Run the agent workflow
        
        Args:
            query: User query
            vectorstore: FAISS vector store
            session_id: Session identifier
            
        Returns:
            Dictionary with answer and metadata
        """
        logger.info(f"Invoking agent for query: {query}")
        
        # Get chat history
        chat_history = self.memory_manager.get_recent_messages(session_id, n=5)
        
        # Initialize state
        initial_state = AgentState(
            query=query,
            vectorstore=vectorstore,
            search_results={},
            context="",
            answer="",
            source_type="unknown",
            session_id=session_id,
            chat_history=chat_history
        )
        
        # Run the graph
        try:
            final_state = self.graph.invoke(initial_state)
            
            result = {
                "answer": final_state["answer"],
                "source_type": final_state["source_type"],
                "used_web_search": final_state["search_results"].get("used_web_search", False),
                "context": final_state["context"],
                "model": self.model_name
            }
            
            logger.info("Agent workflow completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error in agent workflow: {e}")
            return {
                "answer": "I apologize, but I encountered an error processing your request.",
                "source_type": "error",
                "used_web_search": False,
                "context": "",
                "model": self.model_name
            }