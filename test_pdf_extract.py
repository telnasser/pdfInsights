#!/usr/bin/env python3
"""
Test script for PDF extraction with error handling
"""
import os
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our extraction functions
from rag.document_processor import safe_extract_text, DocumentProcessor

def test_pdf_extraction(pdf_path):
    """Test PDF extraction with various methods."""
    logger.info(f"Testing PDF extraction on: {pdf_path}")
    
    if not os.path.exists(pdf_path):
        logger.error(f"File not found: {pdf_path}")
        return
        
    # Try our safe PDFMiner wrapper
    logger.info("Method 1: Using safe_extract_text (PDFMiner wrapper)...")
    try:
        text1 = safe_extract_text(pdf_path)
        logger.info(f"  Success! Extracted {len(text1)} characters")
        logger.info(f"  Preview: {text1[:100].strip()}")
    except Exception as e:
        logger.error(f"  Error: {str(e)}")
        
    # Try PyPDF2 extraction
    logger.info("Method 2: Using PyPDF2...")
    try:
        # Create a document processor with a temp directory
        processor = DocumentProcessor("./")
        text2 = processor._extract_text_with_pypdf2(pdf_path)
        logger.info(f"  Success! Extracted {len(text2)} characters")
        logger.info(f"  Preview: {text2[:100].strip()}")
    except Exception as e:
        logger.error(f"  Error: {str(e)}")
        
    logger.info("Testing complete")
    
if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Use test.pdf if no file specified
        pdf_path = "test.pdf"
        if not os.path.exists(pdf_path):
            logger.error(f"Default file not found: {pdf_path}")
            logger.error("Please specify a PDF file path as an argument")
            sys.exit(1)
    else:
        pdf_path = sys.argv[1]
        
    test_pdf_extraction(pdf_path)