# RAG System with LlamaIndex, Groq & FAISS

A complete Retrieval-Augmented Generation (RAG) system that combines LlamaIndex for document processing, Groq for fast LLM inference, and FAISS for efficient vector storage and retrieval.

## Features

- 🔄 **Document Loading**: Supports PDF, TXT, DOCX, and Markdown files
- 🧠 **Smart Chunking**: Configurable text chunking with overlap
- 🚀 **Fast Embeddings**: Hugging Face or OpenAI embedding models  
- ⚡ **Vector Storage**: FAISS for efficient similarity search
- 🤖 **Groq LLM**: High-speed inference with Llama, Mixtral, or Gemma models
- 💾 **Persistence**: Automatic saving and loading of vector indices
- 🎯 **Interactive CLI**: User-friendly command-line interface
- 🔍 **Debug Mode**: Inspect retrieved document chunks

## Quick Start

### 1. Installation

```bash
# Clone or download the project files
# Navigate to the project directory

# Install dependencies
pip install -r requirements.txt
```

### 2. API Key Setup

Get your Groq API key from [https://console.groq.com/keys](https://console.groq.com/keys)

Create a `.env` file:
```bash
cp .env.example .env
```

Edit `.env` and add your API key:
```env
GROQ_API_KEY=your_actual_groq_api_key_here
```

### 3. Add Your Documents

Place your documents in the `data/` folder:
```
data/
├── document1.pdf
├── document2.txt
├── research_paper.pdf
└── notes.md
```

Supported formats: PDF, TXT, DOCX, MD

### 4. Run the System

```bash
python app.py
```

The system will:
- Automatically create sample documents if none are found
- Build embeddings and vector index
- Start an interactive chat interface

## Project Structure

```
rag-system/
├── app.py                 # Main application entry point
├── config.py             # Configuration settings
├── document_loader.py    # Document loading and processing
├── vector_store_manager.py # FAISS vector store management
├── rag_system.py         # Core RAG implementation
├── requirements.txt      # Python dependencies
├── .env.example         # Environment variables template
├── README.md           # This file
├── data/               # Your documents go here
└── storage/            # Persistent vector storage
    ├── faiss_index.index
    └── faiss_store.pkl
```

## Configuration Options

Edit `config.py` or set environment variables:

### LLM Models (Groq)
- `llama3-8b-8192` - Fast, good for simple queries
- `llama3-70b-8192` - More capable, default choice  
- `mixtral-8x7b-32768` - Large context window
- `gemma-7b-it` - Google's Gemma model

### Embedding Models

**Hugging Face (Free):**
- `BAAI/bge-small-en-v1.5` - Fast, good quality (default)
- `BAAI/bge-base-en-v1.5` - Better quality, slower
- `sentence-transformers/all-MiniLM-L6-v2` - Lightweight

**OpenAI (Requires API key):**
- `text-embedding-3-small` - Good balance
- `text-embedding-3-large` - Best quality

### Other Settings
```python
CHUNK_SIZE = 1024          # Text chunk size
CHUNK_OVERLAP = 200        # Overlap between chunks  
SIMILARITY_TOP_K = 5       # Number of chunks to retrieve
TEMPERATURE = 0.1          # LLM temperature
MAX_TOKENS = 1024         # Max response length
```

## Usage Examples

### Basic Query
```
Q: What is machine learning?
A: Machine Learning is a subset of artificial intelligence that enables systems to automatically learn and improve from experience without being explicitly programmed...
```

### Debug Mode
```
debug: What is AI?
```
Shows retrieved document chunks with similarity scores.

### Interactive Commands
- Type your question naturally
- `debug:<question>` - Show retrieved chunks
- `quit`, `exit`, or `q` - Exit program

## API Integration

You can also use the RAG system programmatically:

```python
from config import Config
from rag_system import RAGSystem

# Initialize
config = Config()
rag = RAGSystem(config)

# Load documents and create index
rag.initialize_system(use_sample_docs=True)

# Query
response = rag.query("What is artificial intelligence?")
print(response)

# Add new documents
rag.add_documents("path/to/new/documents")
```

## Performance Tips

1. **Embedding Models**: Hugging Face models run locally but are slower on first load. OpenAI models are faster but require API calls.

2. **Document Size**: For large documents, consider adjusting `CHUNK_SIZE` and `CHUNK_OVERLAP`.

3. **FAISS Index**: The system uses a flat index (exact search). For larger datasets, consider implementing IVF indices.

4. **Memory**: The system loads embedding models into memory. Ensure adequate RAM for larger models.

## Troubleshooting

### Common Issues

**"GROQ_API_KEY must be set"**
- Ensure your `.env` file contains a valid Groq API key
- Check that the `.env` file is in the project root directory

**"No documents found"**
- Place supported files (.pdf, .txt, .docx, .md) in the `data/` folder
- The system will create sample documents if none are found

**"Failed to load embedding model"**
- Check your internet connection (for first-time model download)
- Ensure sufficient disk space for model caching
- Try switching to a different embedding model

**Slow performance**
- First run downloads embedding models (~400MB for BGE models)
- Consider using OpenAI embeddings for faster startup
- Reduce `SIMILARITY_TOP_K` for faster retrieval

### Memory Issues
- Use smaller embedding models (`all-MiniLM-L6-v2`)
- Reduce `CHUNK_SIZE` and `MAX_TOKENS`
- Process documents in batches for very large collections

## Advanced Usage

### Custom Embedding Models

Add your own embedding model:

```python
from sentence_transformers import SentenceTransformer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# In config.py
EMBEDDING_MODEL = "your-custom-model-name"
```

### Multiple Document Sources

```python
# Add documents from multiple directories
rag.add_documents("documents/research/")
rag.add_documents("documents/manuals/")
```

### Custom Retrievers

```python
# Get custom retriever with different settings
retriever = rag.vector_store_manager.get_retriever(similarity_top_k=10)
```

## Dependencies

Core dependencies:
- `llama-index` - Document processing and RAG framework
- `groq` - Groq API client
- `faiss-cpu` - Vector similarity search
- `sentence-transformers` - Embedding models
- `rich` - Beautiful terminal interface

See `requirements.txt` for complete list.

## License

This project is provided as-is for educational and development purposes. Please check the licenses of individual dependencies.

## Contributing

Feel free to submit issues and enhancement requests!

## Support

- **Groq Documentation**: [https://console.groq.com/docs](https://console.groq.com/docs)
- **LlamaIndex Documentation**: [https://docs.llamaindex.ai/](https://docs.llamaindex.ai/)
- **FAISS Documentation**: [https://faiss.ai/](https://faiss.ai/)