"""
TXT document processor for the unified knowledge base.
Handles plain text files with intelligent chunking and encoding detection.
"""

import os
import chardet
from typing import Dict, Any, List, Optional
import logging
from .base_processor import BaseDocumentProcessor

logger = logging.getLogger(__name__)

class TXTProcessor(BaseDocumentProcessor):
    """Processor for plain text files."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supported_extensions = ['.txt', '.md', '.log', '.py', '.js', '.html', '.css', '.json', '.xml', '.yml', '.yaml']
        
        # TXT-specific configuration
        self.auto_detect_encoding = True
        self.preserve_line_structure = True
        self.include_line_numbers = False
        self.strip_empty_lines = True
    
    def can_process(self, file_path: str) -> bool:
        """Check if file is a supported text file."""
        return any(file_path.lower().endswith(ext) for ext in self.supported_extensions)
    
    def extract_text(self, file_path: str) -> str:
        """Extract text content from text file."""
        try:
            # Detect encoding
            encoding = self._detect_encoding(file_path)
            
            # Read file
            with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                content = f.read()
            
            # Process content based on settings
            if self.strip_empty_lines:
                lines = [line for line in content.splitlines() if line.strip()]
                content = '\n'.join(lines)
            
            # Add line numbers if requested
            if self.include_line_numbers:
                lines = content.splitlines()
                numbered_lines = [f"{i+1:4}: {line}" for i, line in enumerate(lines)]
                content = '\n'.join(numbered_lines)
            
            return content
            
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {e}")
            raise
    
    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from text file."""
        metadata = {
            'file_path': file_path,
            'file_name': os.path.basename(file_path),
            'file_size': os.path.getsize(file_path),
            'format': 'txt'
        }
        
        try:
            # Detect encoding
            encoding = self._detect_encoding(file_path)
            metadata['encoding'] = encoding
            
            # Read file for analysis
            with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                content = f.read()
            
            # Basic statistics
            lines = content.splitlines()
            metadata.update({
                'line_count': len(lines),
                'char_count': len(content),
                'word_count': len(content.split()),
                'non_empty_lines': len([line for line in lines if line.strip()]),
            })
            
            # File type detection based on extension and content
            file_ext = os.path.splitext(file_path)[1].lower()
            metadata['file_type'] = self._detect_file_type(file_ext, content)
            
            # Content analysis
            metadata.update(self._analyze_content(content, file_ext))
            
        except Exception as e:
            logger.warning(f"Could not extract detailed metadata from {file_path}: {e}")
            metadata['error'] = str(e)
        
        return metadata
    
    def _detect_encoding(self, file_path: str) -> str:
        """Detect file encoding."""
        if not self.auto_detect_encoding:
            return 'utf-8'
        
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)  # Read first 10KB for detection
            
            result = chardet.detect(raw_data)
            encoding = result.get('encoding', 'utf-8')
            confidence = result.get('confidence', 0)
            
            # Fall back to utf-8 if confidence is too low
            if confidence < 0.7:
                encoding = 'utf-8'
            
            logger.debug(f"Detected encoding {encoding} with confidence {confidence} for {file_path}")
            return encoding
            
        except Exception as e:
            logger.warning(f"Encoding detection failed for {file_path}: {e}")
            return 'utf-8'
    
    def _detect_file_type(self, file_ext: str, content: str) -> str:
        """Detect specific file type based on extension and content."""
        # Map extensions to types
        type_map = {
            '.txt': 'plain_text',
            '.md': 'markdown',
            '.log': 'log_file',
            '.py': 'python_code',
            '.js': 'javascript_code',
            '.html': 'html',
            '.css': 'css',
            '.json': 'json',
            '.xml': 'xml',
            '.yml': 'yaml',
            '.yaml': 'yaml',
        }
        
        detected_type = type_map.get(file_ext, 'unknown_text')
        
        # Additional content-based detection
        if detected_type == 'plain_text':
            # Check for specific patterns
            if '#!/bin/bash' in content[:100] or '#!/bin/sh' in content[:100]:
                detected_type = 'shell_script'
            elif 'import ' in content[:1000] and 'def ' in content:
                detected_type = 'python_code'
            elif 'function ' in content and '{' in content and '}' in content:
                detected_type = 'javascript_code'
        
        return detected_type
    
    def _analyze_content(self, content: str, file_ext: str) -> Dict[str, Any]:
        """Analyze content for additional metadata."""
        analysis = {}
        
        try:
            lines = content.splitlines()
            
            # Language detection for code files
            if file_ext in ['.py', '.js', '.html', '.css']:
                analysis['programming_language'] = file_ext[1:]  # Remove dot
            
            # Structure analysis for different file types
            if file_ext == '.md':
                analysis.update(self._analyze_markdown(content))
            elif file_ext == '.py':
                analysis.update(self._analyze_python_code(content))
            elif file_ext == '.log':
                analysis.update(self._analyze_log_file(lines))
            elif file_ext == '.json':
                analysis.update(self._analyze_json(content))
            
            # General text analysis
            analysis.update({
                'avg_line_length': sum(len(line) for line in lines) / len(lines) if lines else 0,
                'max_line_length': max(len(line) for line in lines) if lines else 0,
                'blank_line_ratio': len([line for line in lines if not line.strip()]) / len(lines) if lines else 0,
            })
            
        except Exception as e:
            logger.warning(f"Content analysis failed: {e}")
            analysis['analysis_error'] = str(e)
        
        return analysis
    
    def _analyze_markdown(self, content: str) -> Dict[str, Any]:
        """Analyze markdown file structure."""
        import re
        
        analysis = {}
        
        # Count headers
        header_pattern = r'^#+\s+'
        headers = re.findall(header_pattern, content, re.MULTILINE)
        analysis['num_headers'] = len(headers)
        
        # Count code blocks
        code_block_pattern = r'```.*?```'
        code_blocks = re.findall(code_block_pattern, content, re.DOTALL)
        analysis['num_code_blocks'] = len(code_blocks)
        
        # Count links
        link_pattern = r'\[.*?\]\(.*?\)'
        links = re.findall(link_pattern, content)
        analysis['num_links'] = len(links)
        
        return analysis
    
    def _analyze_python_code(self, content: str) -> Dict[str, Any]:
        """Analyze Python code structure."""
        import re
        
        analysis = {}
        
        # Count functions and classes
        function_pattern = r'^def\s+\w+'
        class_pattern = r'^class\s+\w+'
        import_pattern = r'^(import|from)\s+'
        
        functions = re.findall(function_pattern, content, re.MULTILINE)
        classes = re.findall(class_pattern, content, re.MULTILINE)
        imports = re.findall(import_pattern, content, re.MULTILINE)
        
        analysis.update({
            'num_functions': len(functions),
            'num_classes': len(classes),
            'num_imports': len(imports),
        })
        
        return analysis
    
    def _analyze_log_file(self, lines: List[str]) -> Dict[str, Any]:
        """Analyze log file patterns."""
        import re
        
        analysis = {}
        
        # Common log level patterns
        log_levels = {'ERROR': 0, 'WARN': 0, 'WARNING': 0, 'INFO': 0, 'DEBUG': 0}
        timestamp_pattern = r'\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}'
        
        timestamps = 0
        
        for line in lines:
            # Count log levels
            for level in log_levels:
                if level in line.upper():
                    log_levels[level] += 1
            
            # Count timestamps
            if re.search(timestamp_pattern, line):
                timestamps += 1
        
        analysis.update({
            'log_levels': log_levels,
            'lines_with_timestamps': timestamps,
            'timestamp_ratio': timestamps / len(lines) if lines else 0,
        })
        
        return analysis
    
    def _analyze_json(self, content: str) -> Dict[str, Any]:
        """Analyze JSON file structure."""
        import json
        
        analysis = {}
        
        try:
            data = json.loads(content)
            
            if isinstance(data, dict):
                analysis.update({
                    'json_type': 'object',
                    'num_keys': len(data.keys()),
                    'top_level_keys': list(data.keys())[:10],  # First 10 keys
                })
            elif isinstance(data, list):
                analysis.update({
                    'json_type': 'array',
                    'num_items': len(data),
                })
            else:
                analysis['json_type'] = 'primitive'
                
        except json.JSONDecodeError as e:
            analysis['json_error'] = str(e)
        
        return analysis