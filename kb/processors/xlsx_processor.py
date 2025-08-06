"""
XLSX document processor for the unified knowledge base.
Handles Excel files with multiple worksheets and intelligent content extraction.
"""

import pandas as pd
import os
from typing import Dict, Any, List, Optional
import logging
from .base_processor import BaseDocumentProcessor, DocumentChunk, ProcessingResult

logger = logging.getLogger(__name__)

class XLSXProcessor(BaseDocumentProcessor):
    """Processor for XLSX files."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supported_extensions = ['.xlsx', '.xls']
        
        # XLSX-specific configuration
        self.text_columns_threshold = 20  # Minimum chars to consider a column as text
        self.max_rows_per_chunk = 50  # Maximum rows per chunk
        self.include_sheet_names = True
        self.process_all_sheets = True
        self.include_formulas = False
    
    def can_process(self, file_path: str) -> bool:
        """Check if file is an Excel file."""
        return file_path.lower().endswith(('.xlsx', '.xls'))
    
    def extract_text(self, file_path: str) -> str:
        """Extract text content from Excel file."""
        try:
            text_parts = []
            
            # Get sheet names
            if self.process_all_sheets:
                sheet_names = pd.ExcelFile(file_path).sheet_names
            else:
                sheet_names = [0]  # Just the first sheet
            
            # Add file header
            text_parts.append(f"Excel File: {os.path.basename(file_path)}")
            text_parts.append(f"Sheets: {len(sheet_names)}")
            text_parts.append("---")
            
            # Process each sheet
            for sheet_name in sheet_names:
                try:
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                    
                    if self.include_sheet_names and sheet_name != 0:
                        text_parts.append(f"\n=== Sheet: {sheet_name} ===")
                    elif sheet_name == 0:
                        text_parts.append(f"\n=== Sheet 1 ===")
                    
                    # Identify text columns
                    text_columns = self._identify_text_columns(df)
                    
                    # Add sheet information
                    text_parts.append(f"Columns: {', '.join(df.columns.astype(str).tolist())}")
                    text_parts.append(f"Rows: {len(df)}")
                    text_parts.append(f"Text columns: {', '.join(text_columns)}")
                    text_parts.append("")
                    
                    # Process rows
                    for idx, row in df.iterrows():
                        row_text = self._format_row_as_text(row, text_columns)
                        if row_text:
                            text_parts.append(f"Row {idx + 1}: {row_text}")
                    
                    text_parts.append("")  # Empty line between sheets
                    
                except Exception as e:
                    logger.warning(f"Error processing sheet {sheet_name}: {e}")
                    text_parts.append(f"Error processing sheet {sheet_name}: {e}")
                    continue
            
            return "\n".join(text_parts)
            
        except Exception as e:
            logger.error(f"Error extracting text from Excel {file_path}: {e}")
            raise
    
    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from Excel file."""
        metadata = {
            'file_path': file_path,
            'file_name': os.path.basename(file_path),
            'file_size': os.path.getsize(file_path),
            'format': 'xlsx'
        }
        
        try:
            # Read Excel file info
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names
            
            metadata.update({
                'num_sheets': len(sheet_names),
                'sheet_names': sheet_names,
            })
            
            # Process each sheet for metadata
            sheets_info = []
            total_rows = 0
            total_cols = 0
            all_text_columns = []
            
            for sheet_name in sheet_names:
                try:
                    # Read just a few rows for metadata
                    df = pd.read_excel(file_path, sheet_name=sheet_name, nrows=5)
                    
                    # Full sheet for complete info
                    full_df = pd.read_excel(file_path, sheet_name=sheet_name)
                    
                    text_columns = self._identify_text_columns(df)
                    all_text_columns.extend(text_columns)
                    
                    sheet_info = {
                        'name': sheet_name,
                        'num_rows': len(full_df),
                        'num_columns': len(full_df.columns),
                        'columns': full_df.columns.tolist(),
                        'text_columns': text_columns,
                        'column_types': full_df.dtypes.to_dict(),
                        'sample_data': df.head(3).to_dict('records')
                    }
                    
                    sheets_info.append(sheet_info)
                    total_rows += len(full_df)
                    total_cols = max(total_cols, len(full_df.columns))
                    
                except Exception as e:
                    logger.warning(f"Could not extract metadata from sheet {sheet_name}: {e}")
                    continue
            
            metadata.update({
                'sheets': sheets_info,
                'total_rows': total_rows,
                'max_columns': total_cols,
                'all_text_columns': list(set(all_text_columns)),
            })
            
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
                non_null_values = df[column].dropna().astype(str)
                if len(non_null_values) > 0:
                    avg_length = non_null_values.str.len().mean()
                    
                    if avg_length >= self.text_columns_threshold:
                        text_columns.append(str(column))
        
        return text_columns
    
    def _format_row_as_text(self, row: pd.Series, text_columns: List[str]) -> str:
        """Format a row as readable text focusing on text columns."""
        text_parts = []
        
        # Process text columns first
        for column in text_columns:
            if column in row.index:
                value = row.get(column)
                if pd.notna(value) and str(value).strip():
                    clean_value = str(value).strip()
                    if len(clean_value) > 10:  # Only include substantial text
                        text_parts.append(f"{column}: {clean_value}")
        
        # Include other important columns (like IDs, categories, numbers)
        for column in row.index:
            if str(column) not in text_columns:
                value = row.get(column)
                if pd.notna(value) and str(value).strip():
                    clean_value = str(value).strip()
                    # Include short categorical, ID fields, or numeric values
                    if len(clean_value) <= 100:  # Allow longer values for Excel
                        text_parts.append(f"{column}: {clean_value}")
        
        return " | ".join(text_parts)
    
    def process_document(self, file_path: str) -> ProcessingResult:
        """Process Excel file with sheet-aware chunking."""
        result = super().process_document(file_path)
        
        if not result.success:
            return result
        
        # For Excel files, create sheet-based chunks for better organization
        try:
            sheet_names = pd.ExcelFile(file_path).sheet_names
            
            if len(sheet_names) > 1:
                # Create sheet-based chunks
                sheet_chunks = self._create_sheet_based_chunks(file_path, sheet_names)
                
                if sheet_chunks:
                    result.chunks = sheet_chunks
                    logger.info(f"Using sheet-based chunking for {file_path}: {len(sheet_chunks)} chunks")
        
        except Exception as e:
            logger.warning(f"Could not create sheet-based chunks for {file_path}: {e}")
            # Fall back to text-based chunking
        
        return result
    
    def _create_sheet_based_chunks(self, file_path: str, sheet_names: List[str]) -> List[DocumentChunk]:
        """Create chunks based on Excel sheets."""
        chunks = []
        
        for sheet_index, sheet_name in enumerate(sheet_names):
            try:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                text_columns = self._identify_text_columns(df)
                
                # If sheet is small enough, create one chunk per sheet
                sheet_content = self._format_sheet_content(df, sheet_name, text_columns)
                
                if len(sheet_content) >= self.min_chunk_size:
                    # If sheet content is too large, split it into multiple chunks
                    if len(sheet_content) > self.chunk_size:
                        sheet_chunks = self._create_row_based_chunks_for_sheet(
                            df, sheet_name, text_columns, file_path, len(chunks)
                        )
                        chunks.extend(sheet_chunks)
                    else:
                        # Create single chunk for the sheet
                        chunk_metadata = {
                            'chunk_type': 'sheet_based',
                            'sheet_name': sheet_name,
                            'sheet_index': sheet_index,
                            'num_rows': len(df),
                            'columns_included': text_columns
                        }
                        
                        chunk = DocumentChunk(
                            content=sheet_content,
                            source_file=file_path,
                            chunk_index=len(chunks),
                            language='en',  # Default to English for Excel data
                            metadata=chunk_metadata
                        )
                        chunks.append(chunk)
                
            except Exception as e:
                logger.warning(f"Error processing sheet {sheet_name}: {e}")
                continue
        
        return chunks
    
    def _format_sheet_content(self, df: pd.DataFrame, sheet_name: str, text_columns: List[str]) -> str:
        """Format sheet content as text."""
        content_parts = []
        
        # Add sheet header
        content_parts.append(f"Sheet: {sheet_name}")
        content_parts.append(f"Columns: {', '.join(df.columns.astype(str).tolist())}")
        content_parts.append(f"Rows: {len(df)}")
        content_parts.append("")
        
        # Add each row
        for idx, row in df.iterrows():
            row_text = self._format_row_as_text(row, text_columns)
            if row_text:
                content_parts.append(f"Row {idx + 1}: {row_text}")
        
        return "\n".join(content_parts)
    
    def _create_row_based_chunks_for_sheet(self, 
                                          df: pd.DataFrame, 
                                          sheet_name: str, 
                                          text_columns: List[str], 
                                          source_file: str,
                                          start_chunk_index: int) -> List[DocumentChunk]:
        """Create row-based chunks for a large sheet."""
        chunks = []
        
        # Calculate rows per chunk
        avg_row_length = df[text_columns].astype(str).apply(lambda x: x.str.len().mean()).sum()
        
        if avg_row_length > 0:
            rows_per_chunk = max(1, min(self.max_rows_per_chunk, self.chunk_size // int(avg_row_length)))
        else:
            rows_per_chunk = self.max_rows_per_chunk
        
        # Create chunks from groups of rows
        for i in range(0, len(df), rows_per_chunk):
            chunk_df = df.iloc[i:i + rows_per_chunk]
            
            # Create text content for this chunk
            chunk_content = self._format_chunk_content_for_sheet(
                chunk_df, sheet_name, text_columns, i + 1, min(i + rows_per_chunk, len(df))
            )
            
            if len(chunk_content) >= self.min_chunk_size:
                # Create metadata for this chunk
                chunk_metadata = {
                    'chunk_type': 'sheet_row_based',
                    'sheet_name': sheet_name,
                    'start_row': i + 1,
                    'end_row': min(i + rows_per_chunk, len(df)),
                    'num_rows': len(chunk_df),
                    'columns_included': text_columns
                }
                
                chunk = DocumentChunk(
                    content=chunk_content,
                    source_file=source_file,
                    chunk_index=start_chunk_index + len(chunks),
                    language='en',  # Default to English for Excel data
                    metadata=chunk_metadata
                )
                chunks.append(chunk)
        
        return chunks
    
    def _format_chunk_content_for_sheet(self, 
                                       chunk_df: pd.DataFrame, 
                                       sheet_name: str, 
                                       text_columns: List[str], 
                                       start_row: int, 
                                       end_row: int) -> str:
        """Format a chunk of rows from a sheet as text content."""
        content_parts = []
        
        # Add chunk header
        content_parts.append(f"Sheet: {sheet_name} (Rows {start_row}-{end_row})")
        content_parts.append("")
        
        # Add each row
        for idx, row in chunk_df.iterrows():
            row_text = self._format_row_as_text(row, text_columns)
            if row_text:
                content_parts.append(f"Row {idx + 1}: {row_text}")
        
        return "\n".join(content_parts)