import os
import hashlib
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import PyPDF2
# Import pdfminer components directly
from pdfminer.high_level import extract_text as pdfminer_extract_text
import spacy
from rag.chunking import TextChunker

# Configure logging
logger = logging.getLogger(__name__)

# Create a safer wrapper for pdfminer's extract_text
def safe_extract_text(pdf_path):
    """
    A wrapper around pdfminer's extract_text that handles logging issues.
    PDFMiner has a bug where it can crash when logging binary content.
    """
    # Temporarily disable the pdfminer logger to prevent crashes
    pdfminer_logger = logging.getLogger('pdfminer')
    original_level = pdfminer_logger.level
    pdfminer_logger.setLevel(logging.ERROR)  # Only show errors
    
    try:
        return pdfminer_extract_text(pdf_path)
    except Exception as e:
        logger.error(f"PDFMiner extraction error: {str(e)}")
        raise
    finally:
        # Restore original logging level
        pdfminer_logger.setLevel(original_level)

class DocumentProcessor:
    """
    Processes PDF documents for RAG applications.
    Handles document loading, text extraction, metadata extraction, and chunking.
    """
    
    def __init__(self, upload_folder: str):
        """
        Initialize the document processor.
        
        Args:
            upload_folder: Directory path where uploaded documents are stored
        """
        self.upload_folder = upload_folder
        self.chunker = TextChunker()
        
        # Initialize spaCy for text processing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Download the model if not available
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
            
    def process_pdf(self, filename: str, chunk_strategy: str = "paragraph", 
                   chunk_size: int = 1000, chunk_overlap: int = 200) -> Dict[str, Any]:
        """
        Process a PDF file to extract text, metadata, and create chunks.
        
        Args:
            filename: Name of the uploaded PDF file
            chunk_strategy: Strategy for chunking ('sentence', 'paragraph', 'sliding')
            chunk_size: Size of chunks in characters
            chunk_overlap: Overlap between chunks in characters
            
        Returns:
            Dictionary containing document data, metadata, and chunks
        """
        file_path = os.path.join(self.upload_folder, filename)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Extract text using our safer PDFMiner wrapper with error handling
        try:
            logger.info(f"Extracting text from: {file_path}")
            text = safe_extract_text(file_path)
            logger.info(f"Successfully extracted {len(text)} characters of text")
        except Exception as e:
            logger.error(f"Error extracting text with safe PDFMiner: {str(e)}")
            # Fallback to PyPDF2 if PDFMiner fails
            try:
                logger.info("Falling back to PyPDF2 extraction")
                text = self._extract_text_with_pypdf2(file_path)
                logger.info(f"PyPDF2 extraction successful, got {len(text)} characters")
            except Exception as e2:
                logger.error(f"Error with fallback PyPDF2 extraction: {str(e2)}")
                # Last resort - return empty text
                text = "Could not extract text from this PDF. The file may be corrupted or password-protected."
                logger.warning("Using empty text placeholder as last resort")
        
        # Extract metadata using PyPDF2
        metadata = self._extract_metadata(file_path)
        
        # Generate document ID
        doc_id = self._generate_document_id(file_path)
        
        # Create chunks with enhanced preprocessing
        cleaned_text = ' '.join(text.split())  # Normalize whitespace
        chunks = self.chunker.chunk_text(cleaned_text, strategy=chunk_strategy, 
                                       chunk_size=chunk_size, 
                                       chunk_overlap=chunk_overlap)
        
        # Filter out low-quality chunks
        chunks = [chunk for chunk in chunks if len(chunk.get('text', '').split()) > 10]
        
        # Create document record
        document = {
            "id": doc_id,
            "filename": filename,
            "upload_time": datetime.now().isoformat(),
            "metadata": metadata,
            "text_length": len(text),
            "chunk_count": len(chunks),
            "chunking_strategy": chunk_strategy,
            "chunks": chunks
        }
        
        logger.info(f"Processed document {filename} with {len(chunks)} chunks")
        return document
    
    def _extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from a PDF file."""
        metadata = {}
        
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                
                # Get basic PDF info
                info = reader.metadata
                if info:
                    for key in info:
                        # Convert PDF metadata keys to readable format
                        clean_key = key.strip('/').lower()
                        metadata[clean_key] = str(info[key])
                
                # Get page count
                metadata['page_count'] = len(reader.pages)
                
                # Extract text from first page for analysis
                if len(reader.pages) > 0:
                    first_page_text = reader.pages[0].extract_text()
                    # Extract potential title and summary using NLP
                    if first_page_text:
                        doc = self.nlp(first_page_text[:1000])  # Process first 1000 chars
                        # Find potential title (first sentence)
                        sents = list(doc.sents)  # Convert generator to list
                        if not metadata.get('title') and len(sents) > 0:
                            metadata['extracted_title'] = str(sents[0])
        except Exception as e:
            logger.error(f"Error extracting metadata: {str(e)}")
            
        return metadata
    
    def _extract_text_with_pypdf2(self, file_path: str) -> str:
        """Fallback text extraction using PyPDF2."""
        logger.info(f"Attempting to extract text with PyPDF2 from {file_path}")
        
        extracted_text = ""
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                # Check if the PDF is encrypted
                if reader.is_encrypted:
                    logger.warning(f"PDF is encrypted: {file_path}")
                    try:
                        # Try with empty password
                        reader.decrypt('')
                    except:
                        return "This PDF is encrypted and cannot be processed."
                
                # Extract text from each page
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text:
                        extracted_text += page_text + "\n\n"
                        
            if not extracted_text:
                logger.warning(f"No text could be extracted from {file_path}")
                return "No readable text could be extracted from this PDF."
                
            return extracted_text
                
        except Exception as e:
            logger.error(f"PyPDF2 extraction error: {str(e)}")
            raise
    
    def _generate_document_id(self, file_path: str) -> str:
        """Generate a unique document ID based on file content and name."""
        try:
            hasher = hashlib.md5()
            with open(file_path, 'rb') as file:
                # Read in chunks to handle large files
                for chunk in iter(lambda: file.read(4096), b""):
                    hasher.update(chunk)
            # Add filename to make it unique even for identical content
            hasher.update(os.path.basename(file_path).encode())
            return hasher.hexdigest()
        except Exception as e:
            logger.error(f"Error generating document ID: {str(e)}")
            # Fallback to timestamp-based ID
            return f"doc_{int(datetime.now().timestamp())}"
