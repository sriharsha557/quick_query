"""
Main RAG system implementation using LlamaIndex, Groq, and FAISS.
"""
from typing import List, Optional
from llama_index.core import Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core import get_response_synthesizer

from config import Config
from document_loader import DocumentLoader
from vector_store_manager import VectorStoreManager
import logging

logger = logging.getLogger(__name__)

class RAGSystem:
    """Complete RAG system implementation."""
    
    def __init__(self, config: Config):
        self.config = config
        config.validate()
        
        # Initialize components
        self.llm = self._setup_llm()
        self.embedding_model = self._setup_embedding_model()
        self.document_loader = DocumentLoader(config)
        self.vector_store_manager = VectorStoreManager(config, self.embedding_model)
        
        # Configure global settings
        Settings.llm = self.llm
        Settings.embed_model = self.embedding_model
        Settings.chunk_size = config.CHUNK_SIZE
        Settings.chunk_overlap = config.CHUNK_OVERLAP
        
        # Initialize components
        self.retriever: Optional[BaseRetriever] = None
        self.query_engine = None
        
    def _setup_llm(self) -> Groq:
        """Initialize Groq LLM."""
        return Groq(
            model=self.config.GROQ_MODEL,
            api_key=self.config.GROQ_API_KEY,
            temperature=self.config.TEMPERATURE,
            max_tokens=self.config.MAX_TOKENS
        )
    
    def _setup_embedding_model(self):
        """Initialize embedding model (Hugging Face or OpenAI)."""
        if self.config.EMBEDDING_MODEL.startswith("text-embedding"):
            # OpenAI embedding model
            if not self.config.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY required for OpenAI embeddings")
            
            return OpenAIEmbedding(
                model=self.config.EMBEDDING_MODEL,
                api_key=self.config.OPENAI_API_KEY
            )
        else:
            # Hugging Face embedding model
            return HuggingFaceEmbedding(
                model_name=self.config.EMBEDDING_MODEL
            )
    
    def initialize_system(self, use_sample_docs: bool = False) -> bool:
        """Initialize the RAG system with documents."""
        try:
            # Load documents
            documents = self.document_loader.load_documents()
            
            # Use sample documents if no real documents found
            if not documents and use_sample_docs:
                logger.info("No documents found in data directory, using sample documents")
                documents = self.document_loader.get_sample_documents()
            
            if not documents:
                logger.error("No documents available for indexing")
                return False
            
            # Create nodes from documents
            nodes = self.document_loader.create_nodes(documents)
            if not nodes:
                logger.error("Failed to create nodes from documents")
                return False
            
            # Create or load vector index
            index = self.vector_store_manager.create_or_load_index(nodes)
            
            # Setup retriever and query engine
            self.retriever = self.vector_store_manager.get_retriever()
            
            # Create custom query engine with response synthesizer
            response_synthesizer = get_response_synthesizer(
                response_mode=ResponseMode.COMPACT,
                streaming=False
            )
            
            self.query_engine = RetrieverQueryEngine(
                retriever=self.retriever,
                response_synthesizer=response_synthesizer,
            )
            
            logger.info("RAG system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing RAG system: {e}")
            return False
    
    def query(self, question: str) -> str:
        """Query the RAG system."""
        if not self.query_engine:
            return "Error: RAG system not initialized. Please run initialize_system() first."
        
        try:
            logger.info(f"Processing query: {question}")
            response = self.query_engine.query(question)
            
            # Log retrieved context for debugging
            if hasattr(response, 'source_nodes') and response.source_nodes:
                logger.info(f"Retrieved {len(response.source_nodes)} relevant chunks")
                for i, node in enumerate(response.source_nodes):
                    logger.debug(f"Chunk {i+1} score: {node.score:.3f}")
            
            return str(response)
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return f"Error processing query: {str(e)}"
    
    def add_documents(self, new_documents_path: str = None) -> bool:
        """Add new documents to the existing index."""
        try:
            if new_documents_path:
                # Load documents from specific path
                from llama_index.core import SimpleDirectoryReader
                reader = SimpleDirectoryReader(input_dir=new_documents_path)
                documents = reader.load_data()
            else:
                # Reload from default data directory
                documents = self.document_loader.load_documents()
            
            if not documents:
                logger.warning("No new documents to add")
                return False
            
            # Create nodes
            nodes = self.document_loader.create_nodes(documents)
            if not nodes:
                logger.error("Failed to create nodes from new documents")
                return False
            
            # Add to existing index
            self.vector_store_manager.add_nodes(nodes)
            
            logger.info(f"Added {len(documents)} new documents to the index")
            return True
            
        except Exception as e:
            logger.error(f"Error adding new documents: {e}")
            return False
    
    def get_retriever_results(self, query: str, top_k: int = None) -> List[str]:
        """Get raw retriever results for debugging."""
        if not self.retriever:
            return ["Error: RAG system not initialized"]
        
        try:
            k = top_k or self.config.SIMILARITY_TOP_K
            nodes = self.retriever.retrieve(query)
            
            results = []
            for i, node in enumerate(nodes[:k]):
                score = getattr(node, 'score', 'N/A')
                source = node.metadata.get('source', 'Unknown')
                text_preview = node.text[:200] + "..." if len(node.text) > 200 else node.text
                
                results.append(f"Chunk {i+1} (Score: {score}, Source: {source}):\n{text_preview}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving results: {e}")
            return [f"Error: {str(e)}"]