"""
Base document processor class for the unified knowledge base.
Defines the interface for processing different document formats.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Iterator
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import logging

logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Represents a chunk of processed document content."""
    
    id: str = field(default_factory=lambda: "")
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_file: str = ""
    chunk_index: int = 0
    language: str = "en"
    embedding: Optional[List[float]] = None
    
    def __post_init__(self):
        """Generate ID if not provided."""
        if not self.id:
            self.id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate unique ID for the chunk."""
        content_hash = hashlib.md5(self.content.encode('utf-8')).hexdigest()
        return f"{self.source_file}:{self.chunk_index}:{content_hash[:8]}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary for storage."""
        return {
            'id': self.id,
            'content': self.content,
            'metadata': self.metadata,
            'source_file': self.source_file,
            'chunk_index': self.chunk_index,
            'language': self.language,
            'embedding': self.embedding
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentChunk':
        """Create chunk from dictionary."""
        # Handle the mapping file format
        if 'chunk_id' in data and 'id' not in data:
            data['id'] = data.pop('chunk_id')
        if 'text' in data and 'content' not in data:
            data['content'] = data.pop('text')

        return cls(**data)

@dataclass
class ProcessingResult:
    """Result of document processing."""
    
    chunks: List[DocumentChunk] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    
    def add_chunk(self, chunk: DocumentChunk):
        """Add a chunk to the result."""
        self.chunks.append(chunk)
    
    def get_total_content_length(self) -> int:
        """Get total content length across all chunks."""
        return sum(len(chunk.content) for chunk in self.chunks)

class BaseDocumentProcessor(ABC):
    """Base class for document processors."""
    
    def __init__(self, 
                 chunk_size: int = 512,
                 chunk_overlap: int = 50,
                 min_chunk_size: int = 50,
                 max_chunk_size: int = 1000):
        """
        Initialize processor with chunking parameters.
        
        Args:
            chunk_size: Target size for text chunks
            chunk_overlap: Overlap between consecutive chunks
            min_chunk_size: Minimum chunk size to keep
            max_chunk_size: Maximum chunk size allowed
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        
    @abstractmethod
    def can_process(self, file_path: str) -> bool:
        """Check if this processor can handle the given file."""
        pass
    
    @abstractmethod
    def extract_text(self, file_path: str) -> str:
        """Extract raw text from the document."""
        pass
    
    @abstractmethod
    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from the document."""
        pass
    
    def process_document(self, file_path: str) -> ProcessingResult:
        """
        Process a document into chunks.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            ProcessingResult with chunks and metadata
        """
        start_time = datetime.now()
        result = ProcessingResult()
        
        try:
            # Check if we can process this file
            if not self.can_process(file_path):
                result.success = False
                result.error_message = f"Cannot process file type: {file_path}"
                return result
            
            # Extract text and metadata
            text = self.extract_text(file_path)
            metadata = self.extract_metadata(file_path)
            
            # Detect language
            from utils.language_utils import detect_language
            language_detection = detect_language(text)
            
            # Create chunks
            chunks = self._create_chunks(
                text=text,
                source_file=file_path,
                language=language_detection.language,
                base_metadata=metadata
            )
            
            result.chunks = chunks
            result.metadata = metadata
            result.success = True
            
            logger.info(f"Processed {file_path}: {len(chunks)} chunks created")
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            result.success = False
            result.error_message = str(e)
        
        finally:
            end_time = datetime.now()
            result.processing_time = (end_time - start_time).total_seconds()
        
        return result
    
    def _create_chunks(self, 
                      text: str, 
                      source_file: str, 
                      language: str,
                      base_metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """
        Create chunks from text.
        
        Args:
            text: Source text to chunk
            source_file: Source file path
            language: Detected language
            base_metadata: Base metadata for all chunks
            
        Returns:
            List of DocumentChunk objects
        """
        if not text or len(text.strip()) < self.min_chunk_size:
            return []
        
        chunks = []
        
        # Split text into sentences first for better chunk boundaries
        sentences = self._split_into_sentences(text, language)
        
        # Group sentences into chunks
        current_chunk = ""
        current_chunk_sentences = []
        
        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            if (len(current_chunk) + len(sentence) > self.chunk_size and 
                len(current_chunk) >= self.min_chunk_size):
                
                # Create chunk from current content
                if current_chunk.strip():
                    chunk = self._create_chunk(
                        content=current_chunk.strip(),
                        source_file=source_file,
                        chunk_index=len(chunks),
                        language=language,
                        metadata=base_metadata.copy()
                    )
                    chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(
                    current_chunk_sentences, 
                    self.chunk_overlap
                )
                current_chunk = " ".join(overlap_sentences)
                current_chunk_sentences = overlap_sentences.copy()
            
            # Add sentence to current chunk
            current_chunk += " " + sentence if current_chunk else sentence
            current_chunk_sentences.append(sentence)
        
        # Add final chunk if it has content
        if current_chunk.strip() and len(current_chunk.strip()) >= self.min_chunk_size:
            chunk = self._create_chunk(
                content=current_chunk.strip(),
                source_file=source_file,
                chunk_index=len(chunks),
                language=language,
                metadata=base_metadata.copy()
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_into_sentences(self, text: str, language: str) -> List[str]:
        """Split text into sentences using NLTK."""
        try:
            import nltk
            from nltk.tokenize import sent_tokenize
            
            # Map language codes to NLTK languages
            nltk_lang_map = {
                'en': 'english',
                'es': 'spanish',
                'de': 'german',
                'fr': 'french',
                'it': 'italian',
                'pt': 'portuguese',
            }
            
            nltk_lang = nltk_lang_map.get(language, 'english')
            sentences = sent_tokenize(text, language=nltk_lang)
            
            # Filter out very short sentences
            sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
            
            return sentences
            
        except Exception as e:
            logger.warning(f"Sentence tokenization failed: {e}")
            # Fallback to simple sentence splitting
            sentences = text.split('. ')
            return [s.strip() + '.' for s in sentences if len(s.strip()) > 10]
    
    def _get_overlap_sentences(self, sentences: List[str], overlap_chars: int) -> List[str]:
        """Get sentences for chunk overlap."""
        if not sentences:
            return []
        
        overlap_sentences = []
        char_count = 0
        
        # Start from the end and work backwards
        for sentence in reversed(sentences):
            if char_count + len(sentence) <= overlap_chars:
                overlap_sentences.insert(0, sentence)
                char_count += len(sentence)
            else:
                break
        
        return overlap_sentences
    
    def _create_chunk(self, 
                     content: str, 
                     source_file: str, 
                     chunk_index: int,
                     language: str,
                     metadata: Dict[str, Any]) -> DocumentChunk:
        """Create a DocumentChunk object."""
        # Add chunk-specific metadata
        chunk_metadata = metadata.copy()
        chunk_metadata.update({
            'created_at': datetime.now().isoformat(),
            'content_length': len(content),
            'processor': self.__class__.__name__
        })
        
        return DocumentChunk(
            content=content,
            source_file=source_file,
            chunk_index=chunk_index,
            language=language,
            metadata=chunk_metadata
        )

class ProcessorRegistry:
    """Registry for document processors."""
    
    def __init__(self):
        self.processors: List[BaseDocumentProcessor] = []
    
    def register(self, processor: BaseDocumentProcessor):
        """Register a processor."""
        self.processors.append(processor)
        logger.info(f"Registered processor: {processor.__class__.__name__}")
    
    def get_processor(self, file_path: str) -> Optional[BaseDocumentProcessor]:
        """Get the appropriate processor for a file."""
        for processor in self.processors:
            if processor.can_process(file_path):
                return processor
        return None
    
    def get_supported_extensions(self) -> List[str]:
        """Get all supported file extensions."""
        extensions = set()
        for processor in self.processors:
            if hasattr(processor, 'supported_extensions'):
                extensions.update(processor.supported_extensions)
        return sorted(list(extensions))

# Global registry instance
processor_registry = ProcessorRegistry()