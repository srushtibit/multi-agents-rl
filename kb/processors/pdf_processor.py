"""
PDF document processor for the unified knowledge base.
Handles PDF files with text extraction and table detection.
"""

import os
from typing import Dict, Any, List, Optional
import logging
import re

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

from .base_processor import BaseDocumentProcessor

logger = logging.getLogger(__name__)

class PDFProcessor(BaseDocumentProcessor):
    """Processor for PDF files with multiple extraction backends."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supported_extensions = ['.pdf']
        
        # PDF-specific configuration
        self.extract_tables = True
        self.extract_images_text = False  # OCR would require additional setup
        self.preferred_backend = 'auto'  # 'pymupdf', 'pdfplumber', 'auto'
        self.min_text_length = 5
        self.merge_hyphenated_words = True
    
    def can_process(self, file_path: str) -> bool:
        """Check if file is a PDF and we have required libraries."""
        if not file_path.lower().endswith('.pdf'):
            return False
        
        if not (PYMUPDF_AVAILABLE or PDFPLUMBER_AVAILABLE):
            logger.error("No PDF processing library available. Install PyMuPDF or pdfplumber.")
            return False
        
        return True
    
    def extract_text(self, file_path: str) -> str:
        """Extract text content from PDF file."""
        try:
            backend = self._choose_backend()
            
            if backend == 'pymupdf':
                return self._extract_text_pymupdf(file_path)
            elif backend == 'pdfplumber':
                return self._extract_text_pdfplumber(file_path)
            else:
                raise ValueError(f"Unknown backend: {backend}")
                
        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {e}")
            # Try alternative backend if available
            if self.preferred_backend == 'auto':
                return self._extract_text_fallback(file_path)
            raise
    
    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from PDF file."""
        metadata = {
            'file_path': file_path,
            'file_name': os.path.basename(file_path),
            'file_size': os.path.getsize(file_path),
            'format': 'pdf'
        }
        
        try:
            backend = self._choose_backend()
            
            if backend == 'pymupdf' and PYMUPDF_AVAILABLE:
                metadata.update(self._extract_metadata_pymupdf(file_path))
            elif backend == 'pdfplumber' and PDFPLUMBER_AVAILABLE:
                metadata.update(self._extract_metadata_pdfplumber(file_path))
                
        except Exception as e:
            logger.warning(f"Could not extract detailed metadata from {file_path}: {e}")
            metadata['error'] = str(e)
        
        return metadata
    
    def _choose_backend(self) -> str:
        """Choose the best available backend."""
        if self.preferred_backend == 'pymupdf' and PYMUPDF_AVAILABLE:
            return 'pymupdf'
        elif self.preferred_backend == 'pdfplumber' and PDFPLUMBER_AVAILABLE:
            return 'pdfplumber'
        elif self.preferred_backend == 'auto':
            # Prefer PyMuPDF for speed, fall back to pdfplumber for better table extraction
            if PYMUPDF_AVAILABLE:
                return 'pymupdf'
            elif PDFPLUMBER_AVAILABLE:
                return 'pdfplumber'
        
        raise RuntimeError("No PDF processing backend available")
    
    def _extract_text_pymupdf(self, file_path: str) -> str:
        """Extract text using PyMuPDF."""
        if not PYMUPDF_AVAILABLE:
            raise RuntimeError("PyMuPDF not available")
        
        text_parts = []
        
        with fitz.open(file_path) as doc:
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Extract text
                page_text = page.get_text()
                
                if page_text.strip():
                    # Clean up the text
                    cleaned_text = self._clean_pdf_text(page_text)
                    
                    if len(cleaned_text) >= self.min_text_length:
                        text_parts.append(f"=== Page {page_num + 1} ===")
                        text_parts.append(cleaned_text)
                        text_parts.append("")
                
                # Extract tables if enabled
                if self.extract_tables:
                    table_text = self._extract_tables_pymupdf(page, page_num + 1)
                    if table_text:
                        text_parts.extend(table_text)
        
        return "\n".join(text_parts)
    
    def _extract_text_pdfplumber(self, file_path: str) -> str:
        """Extract text using pdfplumber."""
        if not PDFPLUMBER_AVAILABLE:
            raise RuntimeError("pdfplumber not available")
        
        text_parts = []
        
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # Extract text
                page_text = page.extract_text()
                
                if page_text and page_text.strip():
                    # Clean up the text
                    cleaned_text = self._clean_pdf_text(page_text)
                    
                    if len(cleaned_text) >= self.min_text_length:
                        text_parts.append(f"=== Page {page_num + 1} ===")
                        text_parts.append(cleaned_text)
                        text_parts.append("")
                
                # Extract tables if enabled
                if self.extract_tables:
                    table_text = self._extract_tables_pdfplumber(page, page_num + 1)
                    if table_text:
                        text_parts.extend(table_text)
        
        return "\n".join(text_parts)
    
    def _extract_text_fallback(self, file_path: str) -> str:
        """Try alternative backend as fallback."""
        try:
            if PDFPLUMBER_AVAILABLE:
                logger.info("Falling back to pdfplumber")
                return self._extract_text_pdfplumber(file_path)
            elif PYMUPDF_AVAILABLE:
                logger.info("Falling back to PyMuPDF")
                return self._extract_text_pymupdf(file_path)
        except Exception as e:
            logger.error(f"Fallback extraction also failed: {e}")
        
        return ""
    
    def _clean_pdf_text(self, text: str) -> str:
        """Clean extracted PDF text."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Merge hyphenated words if enabled
        if self.merge_hyphenated_words:
            text = re.sub(r'-\s+(\w)', r'\1', text)
        
        # Remove page numbers at the end of lines (simple heuristic)
        text = re.sub(r'\s+\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Remove excessive line breaks
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        
        return text.strip()
    
    def _extract_tables_pymupdf(self, page, page_num: int) -> List[str]:
        """Extract tables from a PyMuPDF page."""
        table_texts = []
        
        try:
            # PyMuPDF table extraction
            tables = page.find_tables()
            
            for table_index, table in enumerate(tables):
                table_data = table.extract()
                
                if table_data:
                    table_text = [f"Table {table_index + 1} (Page {page_num}):"]
                    
                    for row_index, row in enumerate(table_data):
                        if row and any(cell for cell in row if cell):  # Skip empty rows
                            clean_row = [str(cell).strip() if cell else "" for cell in row]
                            if any(clean_row):
                                table_text.append(f"Row {row_index + 1}: {' | '.join(clean_row)}")
                    
                    if len(table_text) > 1:
                        table_texts.extend(table_text)
                        table_texts.append("")
                        
        except Exception as e:
            logger.debug(f"Error extracting tables from page {page_num}: {e}")
        
        return table_texts
    
    def _extract_tables_pdfplumber(self, page, page_num: int) -> List[str]:
        """Extract tables from a pdfplumber page."""
        table_texts = []
        
        try:
            tables = page.extract_tables()
            
            for table_index, table in enumerate(tables):
                if table:
                    table_text = [f"Table {table_index + 1} (Page {page_num}):"]
                    
                    for row_index, row in enumerate(table):
                        if row and any(cell for cell in row if cell):  # Skip empty rows
                            clean_row = [str(cell).strip() if cell else "" for cell in row]
                            if any(clean_row):
                                table_text.append(f"Row {row_index + 1}: {' | '.join(clean_row)}")
                    
                    if len(table_text) > 1:
                        table_texts.extend(table_text)
                        table_texts.append("")
                        
        except Exception as e:
            logger.debug(f"Error extracting tables from page {page_num}: {e}")
        
        return table_texts
    
    def _extract_metadata_pymupdf(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata using PyMuPDF."""
        metadata = {}
        
        try:
            with fitz.open(file_path) as doc:
                # Document metadata
                doc_metadata = doc.metadata
                metadata.update({
                    'title': doc_metadata.get('title', ''),
                    'author': doc_metadata.get('author', ''),
                    'subject': doc_metadata.get('subject', ''),
                    'keywords': doc_metadata.get('keywords', ''),
                    'creator': doc_metadata.get('creator', ''),
                    'producer': doc_metadata.get('producer', ''),
                    'creation_date': doc_metadata.get('creationDate', ''),
                    'modification_date': doc_metadata.get('modDate', ''),
                })
                
                # Document statistics
                metadata.update({
                    'num_pages': len(doc),
                    'is_encrypted': doc.needs_pass,
                    'is_pdf': doc.is_pdf,
                })
                
                # Count words and characters (approximate)
                total_text = ""
                for page in doc:
                    total_text += page.get_text()
                
                metadata.update({
                    'word_count': len(total_text.split()),
                    'char_count': len(total_text),
                })
                
        except Exception as e:
            logger.warning(f"Error extracting PyMuPDF metadata: {e}")
            metadata['extraction_error'] = str(e)
        
        return metadata
    
    def _extract_metadata_pdfplumber(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata using pdfplumber."""
        metadata = {}
        
        try:
            with pdfplumber.open(file_path) as pdf:
                # Document metadata
                if hasattr(pdf, 'metadata') and pdf.metadata:
                    pdf_metadata = pdf.metadata
                    metadata.update({
                        'title': pdf_metadata.get('Title', ''),
                        'author': pdf_metadata.get('Author', ''),
                        'subject': pdf_metadata.get('Subject', ''),
                        'keywords': pdf_metadata.get('Keywords', ''),
                        'creator': pdf_metadata.get('Creator', ''),
                        'producer': pdf_metadata.get('Producer', ''),
                        'creation_date': pdf_metadata.get('CreationDate', ''),
                        'modification_date': pdf_metadata.get('ModDate', ''),
                    })
                
                # Document statistics
                metadata.update({
                    'num_pages': len(pdf.pages),
                })
                
                # Count words and characters (approximate)
                total_text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        total_text += page_text
                
                metadata.update({
                    'word_count': len(total_text.split()),
                    'char_count': len(total_text),
                })
                
        except Exception as e:
            logger.warning(f"Error extracting pdfplumber metadata: {e}")
            metadata['extraction_error'] = str(e)
        
        return metadata