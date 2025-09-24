"""
Vector store manager for FAISS integration with LlamaIndex.
"""
from typing import List, Optional
import faiss
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import BaseNode
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.embeddings import BaseEmbedding
from config import Config
import pickle
import logging

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """Manages FAISS vector store operations."""
    
    def __init__(self, config: Config, embedding_model: BaseEmbedding):
        self.config = config
        self.embedding_model = embedding_model
        self.vector_store: Optional[FaissVectorStore] = None
        self.storage_context: Optional[StorageContext] = None
        self.index: Optional[VectorStoreIndex] = None
        
        # FAISS index file paths
        self.faiss_index_path = self.config.PERSIST_DIR / "faiss_index.index"
        self.faiss_store_path = self.config.PERSIST_DIR / "faiss_store.pkl"
    
    def _create_faiss_index(self) -> faiss.Index:
        """Create a new FAISS index."""
        # Create a flat index for simplicity (you can use IndexIVFFlat for larger datasets)
        index = faiss.IndexFlatIP(self.config.VECTOR_STORE_DIM)  # Inner product for cosine similarity
        return index
    
    def _load_existing_index(self) -> Optional[faiss.Index]:
        """Load existing FAISS index if available."""
        if self.faiss_index_path.exists():
            try:
                index = faiss.read_index(str(self.faiss_index_path))
                logger.info(f"Loaded existing FAISS index with {index.ntotal} vectors")
                return index
            except Exception as e:
                logger.error(f"Error loading existing index: {e}")
                return None
        return None
    
    def _save_index(self, index: faiss.Index) -> None:
        """Save FAISS index to disk."""
        try:
            faiss.write_index(index, str(self.faiss_index_path))
            logger.info(f"Saved FAISS index to {self.faiss_index_path}")
        except Exception as e:
            logger.error(f"Error saving FAISS index: {e}")
    
    def _save_vector_store(self, vector_store: FaissVectorStore) -> None:
        """Save vector store metadata to disk."""
        try:
            with open(self.faiss_store_path, 'wb') as f:
                pickle.dump(vector_store, f)
            logger.info(f"Saved vector store metadata to {self.faiss_store_path}")
        except Exception as e:
            logger.error(f"Error saving vector store metadata: {e}")
    
    def _load_vector_store(self, faiss_index: faiss.Index) -> Optional[FaissVectorStore]:
        """Load vector store metadata from disk."""
        if self.faiss_store_path.exists():
            try:
                with open(self.faiss_store_path, 'rb') as f:
                    vector_store = pickle.load(f)
                # Update the FAISS index reference
                vector_store._faiss_index = faiss_index
                logger.info("Loaded vector store metadata")
                return vector_store
            except Exception as e:
                logger.error(f"Error loading vector store metadata: {e}")
                return None
        return None
    
    def create_or_load_index(self, nodes: List[BaseNode] = None) -> VectorStoreIndex:
        """Create a new index or load existing one."""
        # Try to load existing index
        existing_faiss_index = self._load_existing_index()
        
        if existing_faiss_index is not None and self.faiss_store_path.exists():
            # Load existing vector store
            vector_store = self._load_vector_store(existing_faiss_index)
            if vector_store:
                self.vector_store = vector_store
                self.storage_context = StorageContext.from_defaults(vector_store=vector_store)
                
                # Create index from existing storage context
                self.index = VectorStoreIndex.from_vector_store(
                    vector_store=vector_store,
                    embed_model=self.embedding_model
                )
                
                logger.info("Loaded existing vector index")
                return self.index
        
        # Create new index if no existing one or loading failed
        logger.info("Creating new vector index")
        faiss_index = self._create_faiss_index()
        self.vector_store = FaissVectorStore(faiss_index=faiss_index)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        
        if nodes:
            # Create index with nodes
            self.index = VectorStoreIndex(
                nodes=nodes,
                storage_context=self.storage_context,
                embed_model=self.embedding_model,
                show_progress=True
            )
        else:
            # Create empty index
            self.index = VectorStoreIndex(
                nodes=[],
                storage_context=self.storage_context,
                embed_model=self.embedding_model
            )
        
        # Save the index and vector store
        self._save_index(faiss_index)
        self._save_vector_store(self.vector_store)
        
        logger.info("Created and saved new vector index")
        return self.index
    
    def add_nodes(self, nodes: List[BaseNode]) -> None:
        """Add new nodes to the existing index."""
        if not self.index:
            raise ValueError("Index not initialized. Call create_or_load_index first.")
        
        if not nodes:
            logger.warning("No nodes provided to add")
            return
        
        try:
            # Add nodes to the index
            for node in nodes:
                self.index.insert(node)
            
            # Save updated index
            if self.vector_store and hasattr(self.vector_store, '_faiss_index'):
                self._save_index(self.vector_store._faiss_index)
                self._save_vector_store(self.vector_store)
            
            logger.info(f"Added {len(nodes)} nodes to the index")
            
        except Exception as e:
            logger.error(f"Error adding nodes to index: {e}")
            raise
    
    def get_retriever(self, similarity_top_k: int = None):
        """Get a retriever from the index."""
        if not self.index:
            raise ValueError("Index not initialized. Call create_or_load_index first.")
        
        k = similarity_top_k or self.config.SIMILARITY_TOP_K
        return self.index.as_retriever(similarity_top_k=k)
    
    def get_query_engine(self, **kwargs):
        """Get a query engine from the index."""
        if not self.index:
            raise ValueError("Index not initialized. Call create_or_load_index first.")
        
        return self.index.as_query_engine(**kwargs)