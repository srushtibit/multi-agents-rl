"""
CSV document processor for the unified knowledge base.
Handles CSV files with intelligent column detection and content chunking.
"""

import pandas as pd
import os
from typing import Dict, Any, List
import logging
from .base_processor import BaseDocumentProcessor, DocumentChunk, ProcessingResult

logger = logging.getLogger(__name__)

class CSVProcessor(BaseDocumentProcessor):
    """Processor for CSV files."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supported_extensions = ['.csv']
        
        # CSV-specific configuration
        self.text_columns_threshold = 20  # Minimum chars to consider a column as text
        self.max_rows_per_chunk = 50  # Maximum rows per chunk
        self.include_headers = True
    
    def can_process(self, file_path: str) -> bool:
        """Check if file is a CSV."""
        return file_path.lower().endswith('.csv')
    
    def extract_text(self, file_path: str) -> str:
        """Extract text content from CSV file."""
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    logger.info(f"Successfully read CSV with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                raise ValueError("Could not read CSV with any supported encoding")
            
            # Identify text columns
            text_columns = self._identify_text_columns(df)
            
            # Convert relevant data to text
            text_parts = []
            
            # Add column information
            if self.include_headers:
                text_parts.append(f"CSV File: {os.path.basename(file_path)}")
                text_parts.append(f"Columns: {', '.join(df.columns.tolist())}")
                text_parts.append(f"Rows: {len(df)}")
                text_parts.append(f"Text columns: {', '.join(text_columns)}")
                text_parts.append("---")
            
            # Process rows
            for idx, row in df.iterrows():
                row_text = self._format_row_as_text(row, text_columns)
                if row_text:
                    text_parts.append(f"Row {idx + 1}: {row_text}")
            
            return "\n".join(text_parts)
            
        except Exception as e:
            logger.error(f"Error extracting text from CSV {file_path}: {e}")
            raise
    
    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from CSV file."""
        metadata = {
            'file_path': file_path,
            'file_name': os.path.basename(file_path),
            'file_size': os.path.getsize(file_path),
            'format': 'csv'
        }
        
        try:
            # Try to read CSV for metadata
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding, nrows=5)  # Read only first 5 rows for metadata
                    metadata['encoding'] = encoding
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is not None:
                metadata.update({
                    'num_columns': len(df.columns),
                    'columns': df.columns.tolist(),
                    'num_rows': len(pd.read_csv(file_path, encoding=metadata['encoding'])),
                    'text_columns': self._identify_text_columns(df),
                    'sample_data': df.head(3).to_dict('records')
                })
                
                # Detect data types
                metadata['column_types'] = df.dtypes.to_dict()
                
        except Exception as e:
            logger.warning(f"Could not extract detailed metadata from {file_path}: {e}")
            metadata['error'] = str(e)
        
        return metadata
    
    def _identify_text_columns(self, df: pd.DataFrame) -> List[str]:
        """Identify columns that contain significant text content."""
        text_columns = []
        
        for column in df.columns:
            # Check if column contains text data
            if df[column].dtype == 'object':
                # Calculate average string length for non-null values
                avg_length = df[column].dropna().astype(str).str.len().mean()
                
                if avg_length >= self.text_columns_threshold:
                    text_columns.append(column)
        
        return text_columns
    
    def _format_row_as_text(self, row: pd.Series, text_columns: List[str]) -> str:
        """Format a row as readable text focusing on text columns."""
        text_parts = []
        
        for column in text_columns:
            value = row.get(column)
            if pd.notna(value) and str(value).strip():
                # Clean and format the value
                clean_value = str(value).strip()
                if len(clean_value) > 10:  # Only include substantial text
                    text_parts.append(f"{column}: {clean_value}")
        
        # Also include other important columns (like IDs, categories)
        for column in row.index:
            if column not in text_columns:
                value = row.get(column)
                if pd.notna(value) and str(value).strip():
                    clean_value = str(value).strip()
                    # Include short categorical or ID fields
                    if len(clean_value) <= 50:
                        text_parts.append(f"{column}: {clean_value}")
        
        return " | ".join(text_parts)
    
    def process_document(self, file_path: str) -> ProcessingResult:
        """Process CSV file with intelligent chunking."""
        result = super().process_document(file_path)
        
        if not result.success:
            return result
        
        # For CSV files, we might want to create chunks based on rows rather than just text length
        # This allows for better semantic chunking of tabular data
        try:
            df = pd.read_csv(file_path, encoding=result.metadata.get('encoding', 'utf-8'))
            text_columns = self._identify_text_columns(df)
            
            # Create row-based chunks
            row_chunks = self._create_row_based_chunks(df, text_columns, file_path)
            
            # If row-based chunks are more sensible, use them instead
            if len(row_chunks) > 1 and len(row_chunks) < len(result.chunks):
                result.chunks = row_chunks
                logger.info(f"Using row-based chunking for {file_path}: {len(row_chunks)} chunks")
        
        except Exception as e:
            logger.warning(f"Could not create row-based chunks for {file_path}: {e}")
            # Fall back to text-based chunking
        
        return result
    
    def _create_row_based_chunks(self, 
                                df: pd.DataFrame, 
                                text_columns: List[str], 
                                source_file: str) -> List[DocumentChunk]:
        """Create chunks based on groups of rows."""
        chunks = []
        
        # Calculate rows per chunk based on content density
        avg_row_length = df[text_columns].astype(str).apply(lambda x: x.str.len().mean()).sum()
        
        if avg_row_length > 0:
            rows_per_chunk = max(1, min(self.max_rows_per_chunk, self.chunk_size // int(avg_row_length)))
        else:
            rows_per_chunk = self.max_rows_per_chunk
        
        # Create chunks from groups of rows
        for i in range(0, len(df), rows_per_chunk):
            chunk_df = df.iloc[i:i + rows_per_chunk]
            
            # Create text content for this chunk
            chunk_content = self._format_chunk_content(chunk_df, text_columns, source_file)
            
            if len(chunk_content) >= self.min_chunk_size:
                # Create metadata for this chunk
                chunk_metadata = {
                    'chunk_type': 'row_based',
                    'start_row': i + 1,
                    'end_row': min(i + rows_per_chunk, len(df)),
                    'num_rows': len(chunk_df),
                    'columns_included': text_columns
                }
                
                chunk = DocumentChunk(
                    content=chunk_content,
                    source_file=source_file,
                    chunk_index=len(chunks),
                    language='en',  # Default to English for CSV data
                    metadata=chunk_metadata
                )
                chunks.append(chunk)
        
        return chunks
    
    def _format_chunk_content(self, 
                             chunk_df: pd.DataFrame, 
                             text_columns: List[str], 
                             source_file: str) -> str:
        """Format a chunk of rows as text content."""
        content_parts = []
        
        # Add chunk header
        content_parts.append(f"Data from {os.path.basename(source_file)} (Rows {chunk_df.index[0] + 1}-{chunk_df.index[-1] + 1}):")
        content_parts.append("")
        
        # Add each row
        for idx, row in chunk_df.iterrows():
            row_text = self._format_row_as_text(row, text_columns)
            if row_text:
                content_parts.append(f"Entry {idx + 1}: {row_text}")
        
        return "\n".join(content_parts)