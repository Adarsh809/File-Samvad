"""
Streamlit application for Agentic RAG Chatbot
"""
import streamlit as st
import os
from datetime import datetime
from data_loader import DataLoader
from embedding import EmbeddingManager
from vectorstore import VectorStoreManager
from graph_agent import AgenticRAGGraph
from utils import get_logger, validate_file_type
import uuid

logger = get_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="File Samvad",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1F77B4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .source-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 0.5rem;
        font-size: 0.875rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    .source-docs {
        background-color: #D4EDDA;
        color: #155724;
    }
    .source-web {
        background-color: #CCE5FF;
        color: #004085;
    }
    .model-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 0.5rem;
        font-size: 0.75rem;
        background-color: #F0F0F0;
        color: #333;
        margin-left: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "vectorstore_ready" not in st.session_state:
        st.session_state.vectorstore_ready = False
    
    if "data_loader" not in st.session_state:
        st.session_state.data_loader = DataLoader()
    
    if "embedding_manager" not in st.session_state:
        st.session_state.embedding_manager = EmbeddingManager()
    
    if "vectorstore_manager" not in st.session_state:
        st.session_state.vectorstore_manager = VectorStoreManager()
    
    if "agent" not in st.session_state:
        st.session_state.agent = None
    
    if "uploaded_files_info" not in st.session_state:
        st.session_state.uploaded_files_info = []


def setup_sidebar():
    """Setup sidebar with configuration and file upload"""
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        # API Key input
        groq_api_key = st.text_input(
            "Groq API Key",
            type="password",
            help="Enter your Groq API key"
        )
        
        # Model selection
        model_name = st.selectbox(
            "Select Groq Model",
            [
                "llama-3.1-8b-instant",
                "mixtral-8x7b-32768",
                "gemma2-9b-it",
                "llama-3.1-70b-versatile"
            ],
            index=0
        )
        
        # Relevance threshold
        relevance_threshold = st.slider(
            "Relevance Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Minimum similarity score for document relevance"
        )
        
        st.markdown("---")
        
        # Initialize agent
        if groq_api_key:
            if st.session_state.agent is None or \
               st.session_state.get("current_api_key") != groq_api_key or \
               st.session_state.get("current_model") != model_name:
                try:
                    st.session_state.agent = AgenticRAGGraph(
                        groq_api_key=groq_api_key,
                        model_name=model_name,
                        relevance_threshold=relevance_threshold
                    )
                    st.session_state.current_api_key = groq_api_key
                    st.session_state.current_model = model_name
                    st.success("✅ Agent initialized!")
                except Exception as e:
                    st.error(f"Error initializing agent: {e}")
        else:
            st.warning("Please enter your Groq API key")
        
        st.markdown("---")
        st.markdown("## 📁 Document Upload")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload files",
            type=["pdf", "docx", "txt", "csv", "pptx", "xlsx"],
            accept_multiple_files=True,
            help="Upload PDF, DOCX, TXT, CSV, PPTX, or XLSX files"
        )
        
        if uploaded_files:
            if st.button("Process Files", type="primary"):
                process_uploaded_files(uploaded_files)
        
        # Display uploaded files info
        if st.session_state.uploaded_files_info:
            st.markdown("### Processed Files")
            for file_info in st.session_state.uploaded_files_info:
                st.text(f"📄 {file_info['filename']}")
                st.caption(f"Chunks: {file_info['chunks']}")
        
        st.markdown("---")
        
        # Session management
        st.markdown("## 🔄 Session Management")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear Chat"):
                st.session_state.messages = []
                st.rerun()
        
        with col2:
            if st.button("New Session"):
                st.session_state.session_id = str(uuid.uuid4())
                st.session_state.messages = []
                st.rerun()
        
        # Statistics
        st.markdown("---")
        st.markdown("### 📊 Statistics")
        st.metric("Messages", len(st.session_state.messages))
        if st.session_state.vectorstore_ready:
            doc_count = st.session_state.vectorstore_manager.get_document_count()
            st.metric("Document Chunks", doc_count)


def process_uploaded_files(uploaded_files):
    """Process uploaded files and create vector store"""
    with st.spinner("Processing files..."):
        try:
            # Prepare files for processing
            files_data = []
            for uploaded_file in uploaded_files:
                file_bytes = uploaded_file.read()
                filename = uploaded_file.name
                
                if validate_file_type(filename):
                    files_data.append((file_bytes, filename))
            
            if not files_data:
                st.error("No valid files to process")
                return
            
            # Load and process documents
            documents = st.session_state.data_loader.process_files(files_data)
            
            if not documents:
                st.error("No documents were created from the files")
                return
            
            # Create embeddings
            embeddings = st.session_state.embedding_manager.get_embeddings()
            
            # Create vector store
            vectorstore = st.session_state.vectorstore_manager.create_vectorstore(
                documents, embeddings
            )
            
            # Update session state
            st.session_state.vectorstore_ready = True
            
            # Store file info
            st.session_state.uploaded_files_info = []
            for file_bytes, filename in files_data:
                file_info = {
                    "filename": filename,
                    "chunks": len([d for d in documents if d.metadata.get("source") == filename])
                }
                st.session_state.uploaded_files_info.append(file_info)
            
            st.success(f"✅ Processed {len(files_data)} files with {len(documents)} chunks!")
            
        except Exception as e:
            st.error(f"Error processing files: {e}")
            logger.error(f"File processing error: {e}")


def display_chat_message(role, content, source_type=None, model=None):
    """Display a chat message with styling"""
    with st.chat_message(role):
        st.markdown(content)
        
        if role == "assistant" and source_type:
            # Display source badge
            if source_type == "documents":
                badge_html = '<span class="source-badge source-docs">📄 From Documents</span>'
            elif source_type == "web":
                badge_html = '<span class="source-badge source-web">🌐 From Web Search</span>'
            else:
                badge_html = ""
            
            # Display model badge
            if model:
                badge_html += f'<span class="model-badge">🤖 {model}</span>'
            
            st.markdown(badge_html, unsafe_allow_html=True)


def main():
    """Main application function"""
    initialize_session_state()
    setup_sidebar()
    
    # Main header
    st.markdown('<h1 class="main-header">🤖 File Samvad Chatbot</h1>', unsafe_allow_html=True)
    st.markdown("Chat with your documents using LangGraph, Groq LLMs, and web search fallback")
    
    # Display chat messages
    for message in st.session_state.messages:
        display_chat_message(
            message["role"],
            message["content"],
            message.get("source_type"),
            message.get("model")
        )
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Check if agent is ready
        if st.session_state.agent is None:
            st.error("Please configure the Groq API key in the sidebar")
            return
        
        # Display user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        display_chat_message("user", prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Get vectorstore if available
                    vectorstore = None
                    if st.session_state.vectorstore_ready:
                        vectorstore = st.session_state.vectorstore_manager.vectorstore
                    
                    # Invoke agent
                    result = st.session_state.agent.invoke(
                        query=prompt,
                        vectorstore=vectorstore,
                        session_id=st.session_state.session_id
                    )
                    
                    answer = result["answer"]
                    source_type = result["source_type"]
                    model = result["model"]
                    
                    # Display answer
                    st.markdown(answer)
                    
                    # Display source badge
                    if source_type == "documents":
                        badge_html = '<span class="source-badge source-docs">📄 From Documents</span>'
                    elif source_type == "web":
                        badge_html = '<span class="source-badge source-web">🌐 From Web Search</span>'
                    else:
                        badge_html = ""
                    
                    badge_html += f'<span class="model-badge">🤖 {model}</span>'
                    st.markdown(badge_html, unsafe_allow_html=True)
                    
                    # Save assistant message
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "source_type": source_type,
                        "model": model
                    })
                    
                except Exception as e:
                    error_msg = f"Error generating response: {e}"
                    st.error(error_msg)
                    logger.error(error_msg)


if __name__ == "__main__":
    main()