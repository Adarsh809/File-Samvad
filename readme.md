# 🤖 File Samvad - Agentic RAG Chatbot with LangGraph
# Deployed at [Streamlit](https://file-samvad-809.streamlit.app/)

A production-ready, modular Retrieval-Augmented Generation (RAG) chatbot built with LangGraph, featuring:
- Multi-format document support (PDF, DOCX, TXT, CSV, PPTX, XLSX)
- HuggingFace embeddings for semantic search
- Groq LLM integration for fast inference
- Intelligent web search fallback
- Conversation memory management
- Streamlit web interface

## 🏗️ Architecture

The chatbot uses a **LangGraph workflow** with the following nodes:

```
User Query → Retrieve (from docs) → Check Relevance → [Relevant?]
                                                          ↓
                                                    Yes → Compose Answer
                                                          ↓
                                                    No → Web Search → Compose Answer
                                                          ↓
                                                    Update Memory → End
```

## 📋 Features

- ✅ **Multi-Format Support**: Upload PDF, DOCX, TXT, CSV, PPTX, XLSX files
- ✅ **Semantic Search**: FAISS vector store with HuggingFace embeddings
- ✅ **Smart Fallback**: Automatic web search when documents lack information
- ✅ **Conversation Memory**: Context-aware responses using LangChain memory
- ✅ **Modular Design**: Clean separation of concerns across modules
- ✅ **Source Attribution**: Clear labeling of information sources
- ✅ **Session Management**: Multiple conversation sessions
- ✅ **Real-time Processing**: Progress indicators and status updates

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/Adarsh809/File-Samvad
cd File-Samvad
```

### 2. Create virtual environment

```bash
uv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
uv add -r requirements.txt
```

### 4. Set up API keys

Get your Groq API key from [https://console.groq.com](https://console.groq.com)

## 🎯 Usage

### Run the Streamlit app

```bash
streamlit run app.py
```

### Using the chatbot

1. **Configure API Key**: Enter your Groq API key in the sidebar
2. **Select Model**: Choose from available Groq models 
3. **Upload Documents**: Upload one or more files (PDF, DOCX, etc.)
4. **Process Files**: Click "Process Files" to create the vector store
5. **Start Chatting**: Ask questions about your documents!

## 📁 Project Structure

```
agentic_rag_chatbot/
│
├── app.py                 # Streamlit UI and main application
├── graph_agent.py         # LangGraph workflow definition
├── data_loader.py         # Multi-format file processing
├── embedding.py           # HuggingFace embeddings manager
├── vectorstore.py         # FAISS vector store operations
├── search.py              # Document search + web search
├── memory_manager.py      # Conversation memory handling
├── utils.py               # Helper functions and utilities
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🔧 Configuration

### Model Options

Available Groq models:
- `llama-3.1-8b-instant` (default) - Fastest responses
- `llama-3.1-70b-versatile` - Best balance of speed and quality
- `mixtral-8x7b-32768` - Large context window
- `gemma2-9b-it` - Efficient and accurate

### Relevance Threshold

Adjust the relevance threshold (0.0 - 1.0) to control when web search is triggered:
- **Higher values (0.7-0.9)**: More strict, triggers web search more often
- **Lower values (0.3-0.5)**: More lenient, uses documents more often

### Embedding Models

Default: `sentence-transformers/all-MiniLM-L6-v2`

Other options:
- `sentence-transformers/all-mpnet-base-v2` (higher quality, slower)
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (multilingual)

## 🧩 Module Descriptions

### `data_loader.py`
- Supports multiple file formats
- Text extraction and cleaning
- Chunking with RecursiveCharacterTextSplitter
- Document metadata management

### `embedding.py`
- HuggingFace embeddings integration
- Lazy loading for efficiency
- Normalized embeddings for better similarity

### `vectorstore.py`
- FAISS vector store creation and management
- Similarity search with scores
- Persistent storage support
- Add/delete operations

### `search.py`
- Document search with relevance scoring
- DuckDuckGo web search integration
- Hybrid search strategy
- Context formatting

### `memory_manager.py`
- Conversation buffer memory
- Session management
- Chat history retrieval
- Export capabilities

### `graph_agent.py`
- LangGraph workflow orchestration
- Five-node agent architecture
- Groq LLM integration
- State management

### `app.py`
- Streamlit UI components
- File upload and processing
- Chat interface
- Session state management

## 📊 How It Works

### 1. Document Processing
```
Upload Files → Extract Text → Clean Text → Chunk Text → Create Embeddings → Store in FAISS
```

### 2. Query Processing
```
User Query → Embed Query → Search Documents → Check Relevance
    ↓
[High Relevance] → Use Document Context → Generate Answer
    ↓
[Low Relevance] → Web Search → Use Web Results → Generate Answer
```

### 3. Memory Management
```
Each Conversation Turn → Save to Memory → Include in Next Query Context
```

## 🔍 Example Queries

**From Documents:**
- "What are the main points in the uploaded report?"
- "Summarize the findings from the research paper"
- "What does the document say about XYZ?"

**Web Fallback:**
- "What's the latest news about AI?"
- "Who won the recent election?"
- "What's the weather like today?"

**Context-Aware:**
- "Tell me more about that" (references previous conversation)
- "Can you explain it differently?" (uses chat history)

## 🛠️ Customization

### Add New File Types

Edit `data_loader.py`:
```python
def load_custom_format(self, file_bytes: bytes, filename: str) -> str:
    # Your custom loader logic
    return extracted_text
```

### Change Vector Store

Edit `vectorstore.py` to use Chroma or other vector stores:
```python
from langchain_community.vectorstores import Chroma
# Replace FAISS with Chroma
```

### Modify Agent Workflow

Edit `graph_agent.py` to add/remove nodes or change logic:
```python
workflow.add_node("custom_node", self.custom_node_function)
workflow.add_edge("check_relevance", "custom_node")
```

## 📈 Performance Tips

1. **Chunk Size**: Adjust `chunk_size` in DataLoader for optimal retrieval
2. **Top-K Results**: Modify `top_k` in search functions for more/fewer results
3. **Model Selection**: Use smaller models for faster responses
4. **Relevance Threshold**: Fine-tune based on your use case

## 🐛 Troubleshooting

### "No module named X"
```bash
uv add -r requirements.txt
```

### "API Key Error"
- Verify your Groq API key is correct
- Check API key has sufficient credits

### "Vector Store Not Found"
- Ensure files are processed before querying
- Check the vectorstore directory exists

### "Memory Error"
- Reduce chunk size in data_loader.py
- Process fewer files at once

## 📝 License

This project is open-source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🙏 Acknowledgments

- LangChain & LangGraph for the framework
- Groq for fast LLM inference
- HuggingFace for embeddings
- Streamlit for the UI framework

## 📧 Contact

For questions or support, please open an issue on the repository.

---

**Built with ❤️ using LangGraph, Groq, and Streamlit**
