"""
Centralized ChromaDB Manager to prevent instance conflicts
"""
import chromadb
from chromadb.utils import embedding_functions
import logging
from typing import Optional, Dict
import threading

logger = logging.getLogger(__name__)

class ChromaDBManager:
    """
    Singleton manager for ChromaDB to ensure consistent settings across all components
    """
    _instance = None
    _lock = threading.Lock()
    _client = None
    _embedding_function = None
    _collections_cache = {}
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the ChromaDB manager"""
        if self._client is None:
            self._initialize_client()
    
    def _initialize_client(self):
        """Initialize ChromaDB client with consistent settings"""
        try:
            # Use the new ChromaDB API without deprecated settings
            self._client = chromadb.PersistentClient(
                path="./chroma_db"
            )
            
            # Initialize embedding function once
            self._embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="BAAI/bge-large-en-v1.5"
            )
            
            logger.info("ChromaDB client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {e}")
            raise
    
    def get_client(self):
        """Get the ChromaDB client"""
        if self._client is None:
            self._initialize_client()
        return self._client
    
    def get_embedding_function(self):
        """Get the consistent embedding function"""
        if self._embedding_function is None:
            self._embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="BAAI/bge-large-en-v1.5"
            )
        return self._embedding_function
    
    def create_collection(self, name: str, metadata: Optional[Dict] = None):
        """Create a new collection with consistent settings"""
        try:
            client = self.get_client()
            
            # Delete if exists
            try:
                client.delete_collection(name)
                logger.info(f"Deleted existing collection: {name}")
            except:
                pass
            
            # Create with consistent embedding function
            collection = client.create_collection(
                name=name,
                embedding_function=self.get_embedding_function(),
                metadata=metadata or {"hnsw:space": "cosine"}
            )
            
            # Cache it
            self._collections_cache[name] = collection
            logger.info(f"Created collection: {name}")
            
            return collection
            
        except Exception as e:
            logger.error(f"Failed to create collection {name}: {e}")
            raise
    
    def get_collection(self, name: str):
        """Get a collection, using cache if available"""
        # Check cache first
        if name in self._collections_cache:
            return self._collections_cache[name]
        
        try:
            client = self.get_client()
            
            # Try to get with embedding function
            try:
                collection = client.get_collection(
                    name=name,
                    embedding_function=self.get_embedding_function()
                )
                self._collections_cache[name] = collection
                return collection
                
            except Exception as e:
                logger.warning(f"Could not get collection with embedding function: {e}")
                
                # Try without embedding function as fallback
                collection = client.get_collection(name=name)
                self._collections_cache[name] = collection
                logger.info(f"Got collection {name} without embedding function")
                return collection
                
        except Exception as e:
            logger.error(f"Failed to get collection {name}: {e}")
            raise
    
    def get_or_create_collection(self, name: str, metadata: Optional[Dict] = None):
        """Get existing collection or create if doesn't exist"""
        try:
            return self.get_collection(name)
        except:
            return self.create_collection(name, metadata)
    
    def clear_cache(self):
        """Clear the collections cache"""
        self._collections_cache.clear()
    
    def reset(self):
        """Reset the entire ChromaDB manager"""
        with self._lock:
            self._collections_cache.clear()
            self._client = None
            self._embedding_function = None
            self._initialize_client()

# Global instance
chroma_manager = ChromaDBManager()