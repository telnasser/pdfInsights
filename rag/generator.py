import os
import re
import logging
from typing import List, Dict, Any, Optional

import anthropic
from config import EMBEDDING_API_KEY, CLAUDE_LLM_MODEL, API_TIMEOUT

logger = logging.getLogger(__name__)

class Generator:
    """
    Generates responses using Anthropic Claude LLM based on retrieved chunks.
    """
    
    def __init__(self, model_name: str = None):
        """
        Initialize the generator.
        
        Args:
            model_name: Name of the Claude model to use
        """
        self.model_name = model_name or CLAUDE_LLM_MODEL
        self.api_key = EMBEDDING_API_KEY  # Using the same API key for embeddings and LLM
        self.client = None
        
        # Try to initialize the Anthropic client
        try:
            if self.api_key:
                self.client = anthropic.Anthropic(api_key=self.api_key)
                logger.info(f"Initialized Anthropic client with model {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic client: {str(e)}")
            self.client = None
        
    def generate_response(self, query: str, chunks: List[Dict[str, Any]], 
                         temperature: float = 0.7, max_tokens: int = 1000) -> Dict[str, Any]:
        """
        Generate a response for a query using retrieved chunks as context.
        
        Args:
            query: User query
            chunks: Retrieved chunks with text and metadata
            temperature: Generation temperature (0-1)
            max_tokens: Maximum tokens in response
            
        Returns:
            Dictionary with response text and metadata
        """
        # Check if any chunks were found
        if not chunks:
            # Return a response indicating no relevant information was found
            no_chunks_response = {
                "text": f"I don't have enough information to answer about '{query}' based on the provided context.",
                "sources": [],
                "metadata": {
                    "chunks_found": 0,
                    "query": query
                }
            }
            return no_chunks_response
            
        # Format chunks into context
        context = self._format_context(chunks)
        
        # Create the prompt
        prompt = self._create_prompt(query, context)
        
        # Generate response
        if self.is_api_available():
            response_text = self._call_api(prompt, temperature, max_tokens)
        else:
            # Use simple local response generation if API is unavailable
            response_text = self._generate_local_response(prompt)
            
        # Format the response with source citations if needed
        response = self._format_response(response_text, chunks)
        
        return response
        
    def _format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Format chunks into context string for the prompt."""
        context_parts = []
        
        for i, chunk in enumerate(chunks):
            # Get text from chunk
            text = chunk.get('text', '')
            
            # Skip empty chunks
            if not text.strip():
                continue
                
            # Add source metadata (like document_id, page_num, etc.)
            source_info = []
            
            if 'document_id' in chunk:
                source_info.append(f"Document: {chunk['document_id']}")
                
            if 'page_num' in chunk:
                source_info.append(f"Page: {chunk['page_num']}")
                
            if 'score' in chunk:
                score = chunk['score']
                if isinstance(score, (int, float)):
                    source_info.append(f"Relevance: {score:.2f}")
                    
            source_metadata = "; ".join(source_info)
            
            # Format context with source number for citation
            context_parts.append(f"[{i+1}] {text}")
            if source_metadata:
                context_parts.append(f"Source {i+1}: {source_metadata}")
                
            # Add separator between chunks
            context_parts.append("")
            
        return "\n".join(context_parts)
        
    def _create_prompt(self, query: str, context: str) -> str:
        """Create the prompt for the LLM."""
        return f"""You are a precise and accurate assistant. Your task is to answer the following query using ONLY the information provided in the context below. 

Key instructions:
1. If exact information is not found in the context, explicitly state this
2. Cite your sources using [X] format where X is the source number
3. Avoid making assumptions beyond what is directly stated in the context
4. Quote relevant portions of the text when appropriate

Context:
{context}

Query: {query}

Answer:"""
        
    def is_api_available(self) -> bool:
        """Check if the Anthropic Claude API is available and configured."""
        # Check if client is initialized
        if self.client is None:
            logger.warning("Anthropic Claude client is not initialized")
            return False
            
        # Check if API key is configured
        api_key = os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            logger.warning("ANTHROPIC_API_KEY environment variable not set")
            return False
        
        return True
    
    def _call_api(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """Call Anthropic Claude API to generate response or use fallback method."""
        try:
            # First check if API is available
            if not self.is_api_available():
                logger.warning("Anthropic Claude API is not available, using fallback response generation")
                return self._generate_local_response(prompt)
                
            # Parse the prompt to extract system and user messages
            system_content = prompt.split("Query:")[0].strip()
            query_part = prompt.split("Query:")[1].split("Answer:")[0].strip()
            
            logger.info(f"Generating response with Claude model: {self.model_name}, temperature={temperature}")
            
            # Make the API call using the Anthropic SDK
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_content,
                messages=[
                    {"role": "user", "content": f"Query: {query_part}"}
                ]
            )
            
            # Extract the response text
            if response and hasattr(response, 'content') and len(response.content) > 0:
                # Extract content from the first content block
                for content_block in response.content:
                    if hasattr(content_block, 'text') and content_block.text:
                        logger.info(f"Successfully generated response via Claude API ({len(content_block.text)} chars)")
                        return content_block.text
            
            logger.warning("Claude API returned empty or invalid response")
                
        except Exception as e:
            logger.error(f"Error using Anthropic Claude API: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
        # If we get here, something went wrong with the API call
        logger.warning("API generation failed, using fallback response method")
        return self._generate_local_response(prompt)
        
    def _generate_local_response(self, prompt: str) -> str:
        """
        Generate a simple response locally when API is unavailable.
        This is a basic extraction-based response without language model capabilities.
        """
        # Get the query from the prompt
        query_match = re.search(r"Query: (.*?)(?:\n\nAnswer:|\Z)", prompt, re.DOTALL)
        query = query_match.group(1).strip() if query_match else "unknown query"
        
        # Extract context sections
        context_match = re.search(r"Context:\n(.*?)(?:\n\nQuery:|\Z)", prompt, re.DOTALL)
        context = context_match.group(1) if context_match else ""
        
        # Split context into chunks by source markers
        chunks = re.split(r"\[\d+\]", context)
        chunks = [c.strip() for c in chunks if c.strip()]
        
        # Find chunk with highest term overlap with query
        query_terms = set(query.lower().split())
        best_chunk_score = 0
        best_chunk_idx = 0
        
        for i, chunk in enumerate(chunks):
            chunk_terms = set(chunk.lower().split())
            overlap = len(query_terms.intersection(chunk_terms))
            if overlap > best_chunk_score:
                best_chunk_score = overlap
                best_chunk_idx = i
                
        # If no good match, return a generic response
        if best_chunk_score <= 1:
            return f"I don't have enough information to answer about '{query}' based on the provided context."
            
        # Extract sentences from the best chunk that contain query terms
        sentences = re.split(r'(?<=[.!?]) +', chunks[best_chunk_idx])
        relevant_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Check if sentence contains any query term
            if any(term.lower() in sentence.lower() for term in query_terms):
                relevant_sentences.append(sentence)
                
        # Combine relevant sentences into a response
        if relevant_sentences:
            response = " ".join(relevant_sentences)
            return f"{response} [1]"  # Add citation to the first source
        else:
            # Fall back to the best chunk if no specific sentences match
            return f"{chunks[best_chunk_idx][:200]}... [1]"
            
    def _format_response(self, response_text: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Format the response with source citations."""
        # Extract citations from response
        citations = re.findall(r'\[(\d+)\]', response_text)
        sources = []
        
        for citation in citations:
            try:
                idx = int(citation) - 1
                if 0 <= idx < len(chunks):
                    # Add source if not already added
                    if idx not in [s.get('chunk_index') for s in sources]:
                        sources.append({
                            'chunk_index': idx,
                            'text': chunks[idx].get('text', ''),
                            'metadata': {
                                k: v for k, v in chunks[idx].items() 
                                if k not in ['text', 'embedding', 'score']
                            }
                        })
            except ValueError:
                pass
                
        return {
            'text': response_text,
            'sources': sources
        }
        
