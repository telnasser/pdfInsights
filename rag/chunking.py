import re
from typing import List, Dict, Any
import spacy
import logging

logger = logging.getLogger(__name__)

class TextChunker:
    """
    Advanced text chunking implementation with multiple strategies:
    - Sentence-based: Split by sentences and group into chunks
    - Paragraph-based: Split by paragraphs and chunk
    - Sliding window: Create overlapping chunks of fixed size
    """
    
    def __init__(self):
        """Initialize the chunker with spaCy for text processing."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Download the model if not available
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
            
        # Disable unnecessary components for better performance
        self.nlp.disable_pipes(["ner", "lemmatizer", "tagger"])
    
    def chunk_text(self, text: str, strategy: str = "paragraph", 
                  chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
        """
        Chunk text using the specified strategy.
        
        Args:
            text: The text to chunk
            strategy: Chunking strategy ('sentence', 'paragraph', 'sliding')
            chunk_size: Target size of chunks in characters
            chunk_overlap: Overlap between chunks in characters
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text:
            return []
            
        # Clean text: only normalize horizontal whitespace, preserve line structure
        text = re.sub(r'[ \t]+', ' ', text)       # collapse spaces/tabs only
        text = re.sub(r'\n{3,}', '\n\n', text)    # cap consecutive blank lines at 2
        
        # Choose chunking strategy
        if strategy == "sentence":
            chunks = self._sentence_chunking(text, chunk_size, chunk_overlap)
        elif strategy == "paragraph":
            chunks = self._paragraph_chunking(text, chunk_size, chunk_overlap)
        elif strategy == "sliding":
            chunks = self._sliding_window_chunking(text, chunk_size, chunk_overlap)
        else:
            logger.warning(f"Unknown chunking strategy: {strategy}, using paragraph chunking")
            chunks = self._paragraph_chunking(text, chunk_size, chunk_overlap)
            
        # Add chunk metadata and index
        for i, chunk in enumerate(chunks):
            chunks[i] = {
                "id": f"chunk_{i}",
                "index": i,
                "text": chunk,
                "strategy": strategy,
                "length": len(chunk),
                "chunk_size_param": chunk_size,
                "chunk_overlap_param": chunk_overlap
            }
            
        return chunks
    
    def _sentence_chunking(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """
        Split text into sentences and group into chunks of approximately chunk_size characters.
        """
        # Process with spaCy to get sentence boundaries
        doc = self.nlp(text)
        sentences = [str(sent).strip() for sent in doc.sents]
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_len = len(sentence)
            
            # If adding this sentence exceeds chunk_size and we already have content,
            # finish the current chunk
            if current_length + sentence_len > chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                
                # Handle overlap: keep sentences that fit within the overlap window
                overlap_length = 0
                overlap_sentences = []
                
                # Start from the end and work backwards to find sentences for overlap
                for s in reversed(current_chunk):
                    s_len = len(s)
                    if overlap_length + s_len <= chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_length += s_len
                    else:
                        break
                        
                current_chunk = overlap_sentences
                current_length = overlap_length
            
            # Add the current sentence to the chunk
            current_chunk.append(sentence)
            current_length += sentence_len
            
        # Add the last chunk if it contains content
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks
    
    def _paragraph_chunking(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """
        Split text into paragraphs and group into chunks of approximately chunk_size characters.
        """
        # Split text into paragraphs
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for paragraph in paragraphs:
            paragraph_len = len(paragraph)
            
            # If this single paragraph exceeds chunk size, use sentence chunking for it
            if paragraph_len > chunk_size:
                # Process large paragraph with sentence chunking
                para_chunks = self._sentence_chunking(paragraph, chunk_size, chunk_overlap)
                chunks.extend(para_chunks)
                continue
                
            # If adding this paragraph exceeds chunk_size and we already have content,
            # finish the current chunk
            if current_length + paragraph_len > chunk_size and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                
                # For paragraph overlap, carry the last paragraph forward if it fits
                last_paragraph = current_chunk[-1]  # read BEFORE clearing
                current_chunk = []
                current_length = 0
                
                if last_paragraph and len(last_paragraph) <= chunk_overlap:
                    current_chunk = [last_paragraph]
                    current_length = len(last_paragraph)
            
            # Add the current paragraph to the chunk
            current_chunk.append(paragraph)
            current_length += paragraph_len
            
        # Add the last chunk if it contains content
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
            
        return chunks
    
    def _sliding_window_chunking(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """
        Create overlapping chunks of fixed size using a sliding window approach.
        """
        chunks = []
        
        # If text is smaller than chunk size, return it as single chunk
        if len(text) <= chunk_size:
            return [text]
            
        # Calculate stride (non-overlapping part)
        stride = chunk_size - chunk_overlap
        
        # Ensure stride is always positive
        if stride <= 0:
            stride = chunk_size // 2
            logger.warning(f"Overlap too large, adjusted stride to {stride}")
            
        # Create chunks with sliding window
        for i in range(0, len(text) - chunk_overlap, stride):
            end = min(i + chunk_size, len(text))
            chunks.append(text[i:end])
            
            # If we reached the end, break
            if end == len(text):
                break
                
        return chunks
        
    def analyze_chunk_quality(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze chunk quality based on various metrics.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Dictionary with quality metrics
        """
        if not chunks:
            return {"error": "No chunks to analyze"}
            
        # Extract just the text from chunks
        chunk_texts = [chunk["text"] for chunk in chunks]
        
        # Calculate basic statistics
        chunk_lengths = [len(text) for text in chunk_texts]
        avg_length = sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0
        
        # Check for sentence breaks at chunk boundaries
        broken_sentences = 0
        for text in chunk_texts:
            # Check if chunk starts with lowercase letter and doesn't start with common
            # lowercase starters like "the", "a", etc.
            if text and text[0].islower() and not re.match(r'^(the|a|an|in|of|to|for)\b', text.lower()):
                broken_sentences += 1
                
            # Check if chunk ends without proper punctuation
            if text and not re.search(r'[.!?;:]$', text):
                broken_sentences += 1
                
        quality_metrics = {
            "chunk_count": len(chunks),
            "avg_chunk_length": avg_length,
            "min_length": min(chunk_lengths) if chunk_lengths else 0,
            "max_length": max(chunk_lengths) if chunk_lengths else 0,
            "potentially_broken_boundaries": broken_sentences,
            "quality_score": 10 - (broken_sentences / (len(chunks) * 2) * 10) if chunks else 0
        }
        
        return quality_metrics
