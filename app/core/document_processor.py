import os
import io
import magic
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from PyPDF2 import PdfReader
from docx import Document
import markdown
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles document processing including file type detection, text extraction, and chunking."""
    
    SUPPORTED_EXTENSIONS = {
        'application/pdf': '.pdf',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
        'text/plain': '.txt',
        'text/markdown': '.md',
        'text/html': '.html'
    }
    
    @staticmethod
    def detect_file_type(file_content: bytes, filename: str) -> str:
        """Detect file type using both magic and file extension."""
        try:
            # First try magic for content-based detection
            file_type = magic.Magic(mime=True).from_buffer(file_content)
            
            # Validate against supported types
            if file_type in DocumentProcessor.SUPPORTED_EXTENSIONS:
                return file_type
                
            # Fallback to extension-based detection
            ext = Path(filename).suffix.lower()
            for mime, extension in DocumentProcessor.SUPPORTED_EXTENSIONS.items():
                if ext == extension:
                    return mime
                    
            raise ValueError(f"Unsupported file type: {file_type} (extension: {ext})")
            
        except Exception as e:
            logger.error(f"Error detecting file type: {str(e)}")
            raise ValueError("Could not determine file type")
    
    @staticmethod
    def extract_text(file_content: bytes, content_type: str) -> str:
        """Extract text from file content based on detected type."""
        try:
            if content_type == 'application/pdf':
                return DocumentProcessor._extract_pdf_text(file_content)
            elif content_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                return DocumentProcessor._extract_docx_text(file_content)
            elif content_type in ['text/plain', 'text/markdown']:
                return file_content.decode('utf-8')
            elif content_type == 'text/html':
                return DocumentProcessor._extract_html_text(file_content)
            else:
                raise ValueError(f"Unsupported content type: {content_type}")
                
        except Exception as e:
            logger.error(f"Error extracting text: {str(e)}")
            raise ValueError(f"Could not extract text from file: {str(e)}")
    
    @staticmethod
    def _extract_pdf_text(content: bytes) -> str:
        """Extract text from PDF content."""
        try:
            pdf = PdfReader(io.BytesIO(content))
            return "\n\n".join([page.extract_text() for page in pdf.pages])
        except Exception as e:
            logger.error(f"PDF extraction failed: {str(e)}")
            raise ValueError("Failed to extract text from PDF")
    
    @staticmethod
    def _extract_docx_text(content: bytes) -> str:
        """Extract text from DOCX content."""
        try:
            doc = Document(io.BytesIO(content))
            return "\n\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            logger.error(f"DOCX extraction failed: {str(e)}")
            raise ValueError("Failed to extract text from Word document")
    
    @staticmethod
    def _extract_html_text(content: bytes) -> str:
        """Extract text from HTML content."""
        try:
            soup = BeautifulSoup(content, 'html.parser')
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            return soup.get_text(separator='\n', strip=True)
        except Exception as e:
            logger.error(f"HTML extraction failed: {str(e)}")
            # Fallback to raw text if HTML parsing fails
            return content.decode('utf-8', errors='ignore')
    
    @staticmethod
    def chunk_text(text: str, 
                  chunk_size: int = 1000, 
                  chunk_overlap: int = 200) -> List[Dict[str, any]]:
        """
        Split text into chunks with overlap using RecursiveCharacterTextSplitter.
        
        Args:
            text: Input text to chunk
            chunk_size: Maximum size of each chunk
            chunk_overlap: Number of characters to overlap between chunks
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text or not text.strip():
            return []
            
        try:
            # Initialize the splitter
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            
            # Split the text
            chunks = splitter.split_text(text)
            
            # Prepare chunk metadata
            chunk_objects = []
            for i, chunk in enumerate(chunks):
                chunk_objects.append({
                    'text': chunk,
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'chunk_size': len(chunk)
                })
                
            return chunk_objects
            
        except Exception as e:
            logger.error(f"Error chunking text: {str(e)}")
            # Fallback to simple chunking if recursive fails
            return [{'text': text, 'chunk_index': 0, 'total_chunks': 1, 'chunk_size': len(text)}]
