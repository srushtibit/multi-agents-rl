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

        # Normalize index path robustly before creating directories
        try:
            idx_candidate = self.index_path
            if os.path.isdir(idx_candidate):
                # Provided path is a directory
                self.index_dir = idx_candidate
                self.index_path = os.path.join(self.index_dir, 'faiss.index')
            else:
                root, ext = os.path.splitext(idx_candidate)
                if ext:
                    # Has an extension, treat as an index file path
                    self.index_dir = os.path.dirname(idx_candidate) or '.'
                else:
                    if os.path.exists(idx_candidate) and os.path.isfile(idx_candidate):
                        # Exists as a file without extension -> treat as file path
                        self.index_dir = os.path.dirname(idx_candidate) or '.'
                        self.index_path = idx_candidate
                    else:
                        # Treat as a directory path
                        self.index_dir = idx_candidate
                        self.index_path = os.path.join(self.index_dir, 'faiss.index')
        except Exception:
            # Fallback to placing index under kb directory
            self.index_dir = os.path.join('kb', 'vector_index')
            self.index_path = os.path.join(self.index_dir, 'faiss.index')

        # Ensure directories exist after normalization
        os.makedirs(self.index_dir, exist_ok=True)
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
        self.processed_files: set[str] = set()
        self._next_faiss_id: int = 0
        
        # Register processors
        self._register_processors()
        
        # Initialize embedding model
        self._initialize_embedding_model()
        
        # Load existing index if available
        self.load_index()

    def _is_index_ip(self) -> bool:
        """Best-effort detection whether current FAISS index uses Inner Product metric."""
        if self.vector_db != 'faiss' or not FAISS_AVAILABLE or self.vector_index is None:
            return True
        try:
            # Some FAISS wrappers (IndexIDMap2) expose inner index as `index`
            inner_index = getattr(self.vector_index, 'index', self.vector_index)
            type_name = type(inner_index).__name__
            if 'L2' in type_name:
                return False
            if 'IP' in type_name:
                return True
            # Fall back to metric_type attribute if present (0=IP, 1=L2)
            if hasattr(inner_index, 'metric_type'):
                return int(getattr(inner_index, 'metric_type')) == 0
        except Exception:
            pass
        # Default to IP to match our own created indices
        return True
    
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
            
            # Generate embeddings for chunks (batched for speed)
            texts = [chunk.content for chunk in result.chunks]
            try:
                batch_embeddings = self.embedding_model.encode(
                    texts, convert_to_numpy=True, batch_size=32, show_progress_bar=False
                )
            except Exception:
                batch_embeddings = [self._generate_embedding(t) for t in texts]
            for chunk, embedding in zip(result.chunks, batch_embeddings):
                if isinstance(embedding, np.ndarray) and embedding.dtype != np.float32:
                    embedding = embedding.astype(np.float32)
                chunk.embedding = (embedding.tolist() if isinstance(embedding, np.ndarray) else embedding)
            
            # Add chunks to knowledge base
            start_index = len(self.chunks)
            self.chunks.extend(result.chunks)
            
            # Update vector index
            if self.vector_index is None:
                self._initialize_vector_index()
            
            # Add embeddings to index
            embeddings = np.ascontiguousarray(
                np.array([chunk.embedding for chunk in result.chunks], dtype=np.float32)
            )
            # Normalize embeddings if using IP
            try:
                if FAISS_AVAILABLE and self.vector_db == 'faiss':
                    faiss.normalize_L2(embeddings)
            except Exception:
                pass
            if self.vector_db == 'faiss' and FAISS_AVAILABLE:
                try:
                    num = embeddings.shape[0]
                    # Prefer add_with_ids if available on the index
                    if hasattr(self.vector_index, 'add_with_ids'):
                        if not hasattr(self, '_next_faiss_id'):
                            self._next_faiss_id = int(getattr(self.vector_index, 'ntotal', 0))
                        ids = np.arange(self._next_faiss_id, self._next_faiss_id + num, dtype='int64')
                        self.vector_index.add_with_ids(embeddings, ids)
                        self._next_faiss_id += num
                    else:
                        self.vector_index.add(embeddings)
                except Exception as e:
                    logger.error(f"Failed to add embeddings to FAISS index: {e}")
                    raise
            
            # Update metadata
            self._update_metadata(file_path, result, start_index)
            # Track processed file for caching
            self.processed_files.add(file_path)
            
            logger.info(f"Added {len(result.chunks)} chunks from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding document {file_path}: {e}")
            return False
    
    def add_documents_from_directory(self, 
                                   directory: str, 
                                   recursive: bool = True,
                                   file_patterns: List[str] = None,
                                   force_reprocess: bool = False) -> Tuple[int, int]:
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
            # Skip already processed files unless reprocessing is forced
            if not force_reprocess and hasattr(self, 'processed_files') and file_path in self.processed_files:
                logger.info(f"Skipping already processed file: {file_path}")
                continue
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
        
        max_results = self.max_results if max_results is None else max_results
        # Preserve 0.0 values; only default when None
        min_score = self.similarity_threshold if min_score is None else float(min_score)
        
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
            # Ensure contiguous float32 for FAISS and normalize only for IP indices
            if isinstance(query_embedding, np.ndarray) and query_embedding.dtype != np.float32:
                query_embedding = query_embedding.astype(np.float32)
            try:
                if FAISS_AVAILABLE and isinstance(query_embedding, np.ndarray) and self.vector_db == 'faiss' and self._is_index_ip():
                    vec = query_embedding.reshape(1, -1).copy()
                    faiss.normalize_L2(vec)
                    query_embedding = vec.reshape(-1)
            except Exception:
                pass
            
            # Search vector index
            if self.vector_index is None or len(self.chunks) == 0:
                # Attempt to lazily load/rebuild once
                if self.vector_index is None and self.chunks:
                    try:
                        self._rebuild_vector_index()
                    except Exception:
                        pass
                if self.vector_index is None or len(self.chunks) == 0:
                    logger.warning("No documents in knowledge base")
                    return []
            
            # Perform vector search
            if self.vector_db == 'faiss':
                scores, indices = self.vector_index.search(
                    query_embedding.reshape(1, -1),
                    min(max_results * 2, len(self.chunks))  # Get more results for filtering
                )

                # Determine metric type (IP vs L2) for filtering semantics
                metric_is_ip = self._is_index_ip()

                # Convert to results
                results: List[SearchResult] = []
                id_map = getattr(self, 'faiss_id_to_chunk_index', None)

                for score, idx in zip(scores[0], indices[0]):
                    # Map FAISS IDMap label to chunk index when applicable
                    chunk_index = None
                    if id_map and isinstance(idx, (int, np.integer)):
                        chunk_index = id_map.get(int(idx))
                    else:
                        # Fall back to assuming contiguous indexing
                        if 0 <= idx < len(self.chunks):
                            chunk_index = int(idx)

                    if chunk_index is None or not (0 <= chunk_index < len(self.chunks)):
                        continue

                    # Score filtering: for IP higher is better, for L2 lower is better
                    passes_threshold = True
                    if metric_is_ip:
                        passes_threshold = (score >= min_score)
                    else:
                        # For L2 distance, do not apply min_score cutoff (incompatible scale)
                        # Keep top-k as returned by FAISS
                        passes_threshold = True

                    if not passes_threshold:
                        continue

                    chunk = self.chunks[chunk_index]

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
            # The build script creates a mapping.json, not metadata.json.
            # We will check for mapping.json for compatibility.
            metadata_path_to_check = self.metadata_path
            if not os.path.exists(metadata_path_to_check):
                logger.warning(f"{self.metadata_path} not found.")
                # The dataset builder saved mapping as metadata.json; also check mapping.json for future compatibility
                map_path = os.path.join(os.path.dirname(self.metadata_path), "mapping.json")
                if os.path.exists(map_path):
                    logger.info(f"Found {map_path} instead. Attempting to load.")
                    metadata_path_to_check = map_path
                else:
                    logger.info("No existing metadata or mapping file found, starting fresh.")
                    return True

            with open(metadata_path_to_check, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Handle both metadata and mapping file structures
            if 'chunks' in data:
                self.chunks = [DocumentChunk.from_dict(chunk_data) for chunk_data in data.get('chunks', [])]
                self.metadata = data.get('metadata', {})
                self.processed_files = set(self.metadata.keys())
                self.faiss_id_to_chunk_index = {}
            else: # This is a mapping file from the build script (id -> {chunk_id, text, metadata, hash})
                self.chunks = []
                self.faiss_id_to_chunk_index = {}
                for label_str, chunk_data in data.items():
                    # Remove unsupported keys and map fields
                    if 'hash' in chunk_data:
                        chunk_data = {k: v for k, v in chunk_data.items() if k != 'hash'}
                    # DocumentChunk.from_dict handles 'chunk_id' -> 'id' and 'text' -> 'content'
                    # Ensure minimal fields
                    if 'metadata' in chunk_data and isinstance(chunk_data['metadata'], dict):
                        meta = chunk_data['metadata']
                        if 'source' in meta and 'source_file' not in chunk_data:
                            chunk_data['source_file'] = meta.get('source', '')
                        if 'language' in meta and 'language' not in chunk_data:
                            chunk_data['language'] = meta.get('language', 'en')
                    doc_chunk = DocumentChunk.from_dict(chunk_data)
                    self.faiss_id_to_chunk_index[int(label_str)] = len(self.chunks)
                    self.chunks.append(doc_chunk)

            # Load vector index. Accept both file and directory inputs for index_path
            if self.vector_db == 'faiss':
                index_file_path = self.index_path
                if os.path.isdir(index_file_path):
                    index_file_path = os.path.join(index_file_path, 'faiss.index')
                if os.path.exists(index_file_path):
                    self.vector_index = faiss.read_index(index_file_path)
                    self.index_path = index_file_path
                    logger.info(f"Successfully loaded FAISS index from {index_file_path} with {self.vector_index.ntotal} vectors.")
                else:
                    if self.chunks:
                        logger.warning("FAISS index file not found. Rebuilding index from loaded chunks.")
                        self._rebuild_vector_index()
            
            logger.info(f"Loaded knowledge base with {len(self.chunks)} chunks.")
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
        
        embedding = self.embedding_model.encode(text, convert_to_numpy=True)
        # Ensure correct dtype for FAISS
        if isinstance(embedding, np.ndarray) and embedding.dtype != np.float32:
            embedding = embedding.astype(np.float32)
        # Normalize for cosine similarity when using Inner Product
        try:
            if FAISS_AVAILABLE and isinstance(embedding, np.ndarray):
                vec = embedding.reshape(1, -1).copy()
                faiss.normalize_L2(vec)
                embedding = vec.reshape(-1)
        except Exception:
            pass
        return embedding
    
    def _initialize_vector_index(self):
        """Initialize the vector index."""
        if self.vector_db == 'faiss' and FAISS_AVAILABLE:
            # Use inner product with normalized vectors to approximate cosine similarity
            self.vector_index = faiss.IndexFlatIP(self.embedding_dimension)
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
        if embeddings and self.vector_db == 'faiss' and FAISS_AVAILABLE:
            embeddings_array = np.ascontiguousarray(np.array(embeddings, dtype=np.float32))
            # Normalize to unit length for IP similarity
            try:
                faiss.normalize_L2(embeddings_array)
            except Exception:
                pass
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