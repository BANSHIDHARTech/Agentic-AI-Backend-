"""Utility functions for knowledge base operations."""
from typing import Any, Dict, List, Optional, Union
import logging
from urllib.parse import urlparse
import re

logger = logging.getLogger(__name__)

def safe_data(resp: Any) -> Union[list, dict]:
    """Safely extract data from various response types.
    
    Args:
        resp: Response object from Supabase or similar
        
    Returns:
        Extracted data as list or dict, empty list if no data
    """
    if not resp:
        return []
    if hasattr(resp, "data"):
        return resp.data or []
    if isinstance(resp, dict):
        return resp.get("data", [])
    return []

def is_valid_url(url: str) -> bool:
    """Check if a string is a valid URL."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def clean_html(html: str) -> str:
    """Remove HTML tags from a string."""
    import re
    clean = re.compile('<.*?>')
    return re.sub(clean, '', html)

def extract_text_from_html(html: str) -> str:
    """Extract text content from HTML."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, 'html.parser')
    
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.extract()
    
    # Get text and clean up
    text = soup.get_text()
    
    # Break into lines and remove leading/trailing whitespace
    lines = (line.strip() for line in text.splitlines())
    # Break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # Drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)
    
    return text

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """Split text into chunks with overlap.
    
    Args:
        text: Input text to chunk
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of text chunks
    """
    if not text:
        return []
    
    # Simple chunking by character count with overlap
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]
        chunks.append(chunk)
        
        # Move start position, considering overlap
        if end == text_length:
            break
        start = end - min(chunk_overlap, start + chunk_overlap)
    
    return chunks

def get_file_extension(filename: str) -> str:
    """Get the file extension from a filename."""
    return filename.split('.')[-1].lower() if '.' in filename else ''

def is_supported_file_type(filename: str) -> bool:
    """Check if the file type is supported for processing."""
    supported_extensions = {'pdf', 'txt', 'md', 'markdown', 'docx', 'doc', 'html', 'htm'}
    return get_file_extension(filename) in supported_extensions

class DocumentProcessor:
    """Utility class for processing different document types."""
    
    @staticmethod
    def process_text(text: str, metadata: Optional[dict] = None) -> List[Dict[str, Any]]:
        """Process plain text into chunks with metadata."""
        if not text:
            return []
            
        chunks = chunk_text(text)
        return [{
            'content': chunk,
            'metadata': metadata or {}
        } for chunk in chunks]
    
    @staticmethod
    def process_html(html: str, metadata: Optional[dict] = None) -> List[Dict[str, Any]]:
        """Process HTML content into chunks with metadata."""
        text = extract_text_from_html(html)
        return DocumentProcessor.process_text(text, metadata)
    
    @staticmethod
    def process_pdf(file_content: bytes, metadata: Optional[dict] = None) -> List[Dict[str, Any]]:
        """Process PDF file content into chunks with metadata."""
        from PyPDF2 import PdfReader
        from io import BytesIO
        
        text = ""
        try:
            pdf_file = BytesIO(file_content)
            reader = PdfReader(pdf_file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise ValueError(f"Failed to process PDF: {str(e)}")
        
        return DocumentProcessor.process_text(text, metadata)
    
    @staticmethod
    def process_docx(file_content: bytes, metadata: Optional[dict] = None) -> List[Dict[str, Any]]:
        """Process DOCX file content into chunks with metadata."""
        from io import BytesIO
        from docx import Document
        
        text = ""
        try:
            doc_file = BytesIO(file_content)
            doc = Document(doc_file)
            for para in doc.paragraphs:
                text += para.text + "\n"
        except Exception as e:
            logger.error(f"Error processing DOCX: {str(e)}")
            raise ValueError(f"Failed to process DOCX: {str(e)}")
        
        return DocumentProcessor.process_text(text, metadata)
    
    @staticmethod
    def process_markdown(text: str, metadata: Optional[dict] = None) -> List[Dict[str, Any]]:
        """Process markdown text into chunks with metadata."""
        # For now, we'll just process it as plain text
        # In the future, we could parse the markdown structure
        return DocumentProcessor.process_text(text, metadata)
    
    @staticmethod
    def process_file(filename: str, file_content: bytes, metadata: Optional[dict] = None) -> List[Dict[str, Any]]:
        """Process a file based on its extension."""
        if not metadata:
            metadata = {}
            
        metadata['filename'] = filename
        ext = get_file_extension(filename)
        
        try:
            if ext in ['txt', 'md', 'markdown']:
                text = file_content.decode('utf-8')
                if ext in ['md', 'markdown']:
                    return DocumentProcessor.process_markdown(text, metadata)
                return DocumentProcessor.process_text(text, metadata)
            elif ext == 'pdf':
                return DocumentProcessor.process_pdf(file_content, metadata)
            elif ext in ['docx', 'doc']:
                return DocumentProcessor.process_docx(file_content, metadata)
            elif ext in ['html', 'htm']:
                text = file_content.decode('utf-8')
                return DocumentProcessor.process_html(text, metadata)
            else:
                raise ValueError(f"Unsupported file type: {ext}")
        except Exception as e:
            logger.error(f"Error processing file {filename}: {str(e)}")
            raise ValueError(f"Failed to process file {filename}: {str(e)}")
