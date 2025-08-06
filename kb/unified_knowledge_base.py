"""
Unified Knowledge Base for the multilingual multi-agent support system.
Integrates all document processors and provides semantic search capabilities.
"""

import os
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np

# Vector storage imports
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

# Embedding imports
from sentence_transformers import SentenceTransformer

# Document processors
from .processors.base_processor import DocumentChunk, ProcessingResult, processor_registry
from .processors.csv_processor import CSVProcessor
from .processors.docx_processor import DOCXProcessor
from .processors.pdf_processor import PDFProcessor
from .processors.xlsx_processor import XLSXProcessor
from .processors.txt_processor import TXTProcessor

# Configuration and utilities
from utils.config_loader import get_config
from utils.language_utils import detect_language, translate_to_english

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Result from knowledge base search."""
    chunk: DocumentChunk
    score: float
    rank: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'chunk': self.chunk.to_dict(),
            'score': self.score,
            'rank': self.rank
        }

@dataclass
class KnowledgeBaseStats:
    """Statistics about the knowledge base."""
    total_documents: int
    total_chunks: int
    total_characters: int
    languages: List[str]
    file_formats: List[str]
    last_updated: str
    
class UnifiedKnowledgeBase:
    """Unified knowledge base with multi-format support and semantic search."""
    
    def __init__(self, 
                 config_path: str = None,
                 index_path: str = None,
                 metadata_path: str = None):
        """
        Initialize the unified knowledge base.
        
        Args:
            config_path: Path to configuration file
            index_path: Path to store vector index
            metadata_path: Path to store metadata
        """
        self.config = get_config()
        
        # Paths
        self.index_path = index_path or self.config.get('knowledge_base.index_path', 'kb/vector_index')
        self.metadata_path = metadata_path or self.config.get('knowledge_base.metadata_path', 'kb/metadata.json')
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)
        
        # Configuration
        self.vector_db = self.config.get('knowledge_base.vector_db', 'faiss')
        self.embedding_dimension = self.config.get('knowledge_base.embedding_dimension', 384)
        self.similarity_threshold = self.config.get('knowledge_base.similarity_threshold', 0.75)
        self.max_results = self.config.get('knowledge_base.max_results', 10)
        
        # Initialize components
        self.embedding_model = None
        self.vector_index = None
        self.chunks: List[DocumentChunk] = []
        self.metadata: Dict[str, Any] = {}
        
        # Register processors
        self._register_processors()
        
        # Initialize embedding model
        self._initialize_embedding_model()
        
        # Load existing index if available
        self.load_index()
    
    def _register_processors(self):
        """Register all document processors."""
        processor_registry.register(CSVProcessor())
        processor_registry.register(DOCXProcessor())
        processor_registry.register(PDFProcessor())
        processor_registry.register(XLSXProcessor())
        processor_registry.register(TXTProcessor())
        
        logger.info(f"Registered {len(processor_registry.processors)} document processors")
    
    def _initialize_embedding_model(self):
        """Initialize the embedding model."""
        model_name = self.config.get('languages.embedding_model', 
                                   'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        
        try:
            self.embedding_model = SentenceTransformer(model_name)
            logger.info(f"Loaded embedding model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def add_document(self, file_path: str, force_reprocess: bool = False) -> bool:
        """
        Add a document to the knowledge base.
        
        Args:
            file_path: Path to the document
            force_reprocess: Whether to reprocess if already exists
            
        Returns:
            True if document was added successfully
        """
        try:
            # Check if document already exists
            if not force_reprocess and self._document_exists(file_path):
                logger.info(f"Document already exists: {file_path}")
                return True
            
            # Get appropriate processor
            processor = processor_registry.get_processor(file_path)
            if not processor:
                logger.warning(f"No processor available for: {file_path}")
                return False
            
            # Process document
            logger.info(f"Processing document: {file_path}")
            result = processor.process_document(file_path)
            
            if not result.success:
                logger.error(f"Failed to process {file_path}: {result.error_message}")
                return False
            
            # Generate embeddings for chunks
            for chunk in result.chunks:
                embedding = self._generate_embedding(chunk.content)
                chunk.embedding = embedding.tolist()
            
            # Add chunks to knowledge base
            start_index = len(self.chunks)
            self.chunks.extend(result.chunks)
            
            # Update vector index
            if self.vector_index is None:
                self._initialize_vector_index()
            
            # Add embeddings to index
            embeddings = np.array([chunk.embedding for chunk in result.chunks])
            if self.vector_db == 'faiss':
                self.vector_index.add(embeddings)
            
            # Update metadata
            self._update_metadata(file_path, result, start_index)
            
            logger.info(f"Added {len(result.chunks)} chunks from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding document {file_path}: {e}")
            return False
    
    def add_documents_from_directory(self, 
                                   directory: str, 
                                   recursive: bool = True,
                                   file_patterns: List[str] = None) -> Tuple[int, int]:
        """
        Add all supported documents from a directory.
        
        Args:
            directory: Directory path
            recursive: Whether to search recursively
            file_patterns: File patterns to match (e.g., ['*.pdf', '*.docx'])
            
        Returns:
            Tuple of (successful_count, total_count)
        """
        if not os.path.exists(directory):
            logger.error(f"Directory not found: {directory}")
            return 0, 0
        
        # Get supported extensions
        supported_extensions = processor_registry.get_supported_extensions()
        
        # Find files
        files_to_process = []
        if recursive:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    if any(file.lower().endswith(ext) for ext in supported_extensions):
                        files_to_process.append(file_path)
        else:
            for file in os.listdir(directory):
                file_path = os.path.join(directory, file)
                if os.path.isfile(file_path) and any(file.lower().endswith(ext) for ext in supported_extensions):
                    files_to_process.append(file_path)
        
        # Process files
        successful_count = 0
        total_count = len(files_to_process)
        
        logger.info(f"Found {total_count} files to process in {directory}")
        
        for file_path in files_to_process:
            if self.add_document(file_path):
                successful_count += 1
            
            # Log progress
            if (successful_count + 1) % 10 == 0:
                logger.info(f"Processed {successful_count + 1}/{total_count} files")
        
        logger.info(f"Successfully processed {successful_count}/{total_count} files")
        return successful_count, total_count
    
    def search(self, 
               query: str, 
               max_results: int = None,
               language: str = None,
               file_types: List[str] = None,
               min_score: float = None) -> List[SearchResult]:
        """
        Search the knowledge base.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            language: Language filter
            file_types: File type filter
            min_score: Minimum similarity score
            
        Returns:
            List of search results
        """
        if not query.strip():
            return []
        
        max_results = max_results or self.max_results
        min_score = min_score or self.similarity_threshold
        
        try:
            # Detect query language and translate if needed
            query_lang = detect_language(query).language
            
            # If query is not in English, translate for better search
            search_query = query
            if query_lang != 'en':
                translation_result = translate_to_english(query, query_lang)
                if translation_result.confidence > 0.7:
                    search_query = translation_result.translated_text
                    logger.debug(f"Translated query from {query_lang}: {search_query}")
            
            # Generate query embedding
            query_embedding = self._generate_embedding(search_query)
            
            # Search vector index
            if self.vector_index is None or len(self.chunks) == 0:
                logger.warning("No documents in knowledge base")
                return []
            
            # Perform vector search
            if self.vector_db == 'faiss':
                scores, indices = self.vector_index.search(
                    query_embedding.reshape(1, -1), 
                    min(max_results * 2, len(self.chunks))  # Get more results for filtering
                )
                
                # Convert to results
                results = []
                for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                    if idx < len(self.chunks) and score >= min_score:
                        chunk = self.chunks[idx]
                        
                        # Apply filters
                        if language and chunk.language != language:
                            continue
                        if file_types and not any(chunk.source_file.endswith(ft) for ft in file_types):
                            continue
                        
                        results.append(SearchResult(
                            chunk=chunk,
                            score=float(score),
                            rank=len(results) + 1
                        ))
                        
                        if len(results) >= max_results:
                            break
                
                return results
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
        
        return []
    
    def get_similar_chunks(self, chunk_id: str, max_results: int = 5) -> List[SearchResult]:
        """Get chunks similar to a given chunk."""
        # Find the chunk
        target_chunk = None
        for chunk in self.chunks:
            if chunk.id == chunk_id:
                target_chunk = chunk
                break
        
        if not target_chunk:
            return []
        
        # Use the chunk's content as query
        return self.search(target_chunk.content, max_results=max_results)
    
    def get_stats(self) -> KnowledgeBaseStats:
        """Get knowledge base statistics."""
        languages = set()
        file_formats = set()
        total_characters = 0
        
        for chunk in self.chunks:
            languages.add(chunk.language)
            file_ext = os.path.splitext(chunk.source_file)[1].lower()
            file_formats.add(file_ext)
            total_characters += len(chunk.content)
        
        return KnowledgeBaseStats(
            total_documents=len(set(chunk.source_file for chunk in self.chunks)),
            total_chunks=len(self.chunks),
            total_characters=total_characters,
            languages=sorted(list(languages)),
            file_formats=sorted(list(file_formats)),
            last_updated=datetime.now().isoformat()
        )
    
    def save_index(self) -> bool:
        """Save the vector index and metadata to disk."""
        try:
            # Save vector index
            if self.vector_index is not None and self.vector_db == 'faiss':
                faiss.write_index(self.vector_index, self.index_path)
            
            # Save chunks and metadata
            data_to_save = {
                'chunks': [chunk.to_dict() for chunk in self.chunks],
                'metadata': self.metadata,
                'config': {
                    'vector_db': self.vector_db,
                    'embedding_dimension': self.embedding_dimension,
                    'embedding_model': self.embedding_model.model_name if self.embedding_model else None
                },
                'stats': asdict(self.get_stats())
            }
            
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved knowledge base with {len(self.chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            return False
    
    def load_index(self) -> bool:
        """Load the vector index and metadata from disk."""
        try:
            # Load metadata first
            if not os.path.exists(self.metadata_path):
                logger.info("No existing metadata found, starting fresh")
                return True
            
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Load chunks
            self.chunks = [DocumentChunk.from_dict(chunk_data) for chunk_data in data.get('chunks', [])]
            self.metadata = data.get('metadata', {})
            
            # Load vector index
            if os.path.exists(self.index_path) and self.vector_db == 'faiss':
                self.vector_index = faiss.read_index(self.index_path)
            elif self.chunks:
                # Rebuild index if chunks exist but index doesn't
                self._rebuild_vector_index()
            
            logger.info(f"Loaded knowledge base with {len(self.chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False
    
    def _document_exists(self, file_path: str) -> bool:
        """Check if document already exists in the knowledge base."""
        return any(chunk.source_file == file_path for chunk in self.chunks)
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        if self.embedding_model is None:
            raise RuntimeError("Embedding model not initialized")
        
        return self.embedding_model.encode(text, convert_to_numpy=True)
    
    def _initialize_vector_index(self):
        """Initialize the vector index."""
        if self.vector_db == 'faiss' and FAISS_AVAILABLE:
            self.vector_index = faiss.IndexFlatIP(self.embedding_dimension)  # Inner product for similarity
        else:
            raise RuntimeError(f"Vector database {self.vector_db} not available or supported")
    
    def _rebuild_vector_index(self):
        """Rebuild the vector index from existing chunks."""
        if not self.chunks:
            return
        
        logger.info("Rebuilding vector index...")
        self._initialize_vector_index()
        
        # Extract embeddings
        embeddings = []
        for chunk in self.chunks:
            if chunk.embedding:
                embeddings.append(chunk.embedding)
            else:
                # Generate embedding if missing
                embedding = self._generate_embedding(chunk.content)
                chunk.embedding = embedding.tolist()
                embeddings.append(chunk.embedding)
        
        # Add to index
        if embeddings and self.vector_db == 'faiss':
            embeddings_array = np.array(embeddings)
            self.vector_index.add(embeddings_array)
        
        logger.info(f"Rebuilt vector index with {len(embeddings)} embeddings")
    
    def _update_metadata(self, file_path: str, result: ProcessingResult, start_index: int):
        """Update metadata for a processed document."""
        self.metadata[file_path] = {
            'processed_at': datetime.now().isoformat(),
            'chunks_start': start_index,
            'chunks_count': len(result.chunks),
            'processing_time': result.processing_time,
            'document_metadata': result.metadata
        }
    
    def clear(self):
        """Clear all data from the knowledge base."""
        self.chunks = []
        self.metadata = {}
        self.vector_index = None
        
        # Remove saved files
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        if os.path.exists(self.metadata_path):
            os.remove(self.metadata_path)
        
        logger.info("Cleared knowledge base")

# Global knowledge base instance
knowledge_base = UnifiedKnowledgeBase()

def get_knowledge_base() -> UnifiedKnowledgeBase:
    """Get the global knowledge base instance."""
    return knowledge_base