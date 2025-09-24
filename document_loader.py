"""
Document loader module for the RAG system.
Handles loading of PDF and text documents from the data directory.
"""
from pathlib import Path
from typing import List
from llama_index.core import Document, SimpleDirectoryReader
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import BaseNode
from config import Config
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentLoader:
    """Handles document loading and processing for the RAG system."""
    
    def __init__(self, config: Config):
        self.config = config
        self.node_parser = SimpleNodeParser.from_defaults(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
        )
    
    def load_documents(self) -> List[Document]:
        """Load documents from the data directory."""
        if not self.config.DATA_DIR.exists():
            logger.warning(f"Data directory {self.config.DATA_DIR} does not exist")
            return []
        
        # Check if there are any files in the data directory
        supported_extensions = ['.pdf', '.txt', '.docx', '.md']
        files = [
            f for f in self.config.DATA_DIR.iterdir() 
            if f.is_file() and f.suffix.lower() in supported_extensions
        ]
        
        if not files:
            logger.warning(f"No supported files found in {self.config.DATA_DIR}")
            logger.info(f"Supported extensions: {supported_extensions}")
            return []
        
        logger.info(f"Found {len(files)} files to process")
        
        try:
            # Use LlamaIndex's SimpleDirectoryReader
            reader = SimpleDirectoryReader(
                input_dir=str(self.config.DATA_DIR),
                required_exts=supported_extensions,
                recursive=True
            )
            
            documents = reader.load_data()
            logger.info(f"Loaded {len(documents)} documents")
            
            return documents
            
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            return []
    
    def create_nodes(self, documents: List[Document]) -> List[BaseNode]:
        """Create nodes from documents using the configured chunk settings."""
        if not documents:
            logger.warning("No documents provided for node creation")
            return []
        
        try:
            nodes = self.node_parser.get_nodes_from_documents(documents)
            logger.info(f"Created {len(nodes)} nodes from {len(documents)} documents")
            return nodes
            
        except Exception as e:
            logger.error(f"Error creating nodes: {e}")
            return []
    
    def get_sample_documents(self) -> List[Document]:
        """Create sample documents if no files are found in data directory."""
        sample_docs = [
            Document(
                text="""
                Artificial Intelligence (AI) is a branch of computer science that aims to create 
                intelligent machines that work and react like humans. AI systems are designed to 
                perform tasks that typically require human intelligence, such as visual perception, 
                speech recognition, decision-making, and language translation.
                
                Machine Learning is a subset of AI that provides systems the ability to automatically 
                learn and improve from experience without being explicitly programmed. ML focuses on 
                the development of computer programs that can access data and use it to learn for themselves.
                """,
                metadata={"source": "ai_basics.txt", "type": "sample"}
            ),
            Document(
                text="""
                Natural Language Processing (NLP) is a field of AI that gives computers the ability 
                to understand, interpret, and manipulate human language. NLP draws from many disciplines, 
                including computer science and computational linguistics, to help computers understand 
                human language in written and verbal forms.
                
                Common NLP applications include language translation, sentiment analysis, chatbots, 
                and document summarization. Modern NLP systems use deep learning techniques to achieve 
                high performance on these tasks.
                """,
                metadata={"source": "nlp_overview.txt", "type": "sample"}
            )
        ]
        
        logger.info("Created sample documents for demonstration")
        return sample_docs