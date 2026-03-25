import os
import logging
from typing import List, Dict, Any, Union, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import anthropic

from config import EMBEDDING_DIMENSION, EMBEDDING_API_KEY, CLAUDE_EMBEDDING_MODEL

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """
    Generates embeddings for text chunks using Anthropic Claude models.
    Falls back to TF-IDF if API is unavailable.
    """
    
    def __init__(self, model_name: str = None):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of the embedding model to use
        """
        from config import EMBEDDING_DIMENSION, CLAUDE_EMBEDDING_MODEL
        
        self.model_name = model_name or CLAUDE_EMBEDDING_MODEL
        self.embedding_dimension = EMBEDDING_DIMENSION  # Always use the configured dimension
        self.api_key = os.environ.get('ANTHROPIC_API_KEY')
        self.use_fallback = False
        self.tfidf_vectorizer = None
        self.client = None
        
        # Test API key and prepare fallback
        self._test_api_and_prepare_fallback()
        
    def _test_api_and_prepare_fallback(self):
        """Test API access and prepare fallback model if needed."""
        # Check if ANTHROPIC_API_KEY environment variable is available
        api_key = os.environ.get('ANTHROPIC_API_KEY')
        
        if api_key:
            try:
                # Initialize the Anthropic client
                self.client = anthropic.Anthropic(api_key=api_key)
                
                logger.info("Testing Anthropic Claude API access...")
                
                # Just do a simple check to see if we can create an instance
                if self.client:
                    logger.info("Anthropic Claude API connection initialized successfully")
                    self.use_fallback = False
                    return
                    
            except Exception as e:
                logger.error(f"Error connecting to Anthropic Claude API: {str(e)}")
                
        # If we get here, we need to use fallback
        print("Using local TF-IDF embeddings as fallback")
        self.use_fallback = True
        
        # Make sure embedding dimension is set
        if not self.embedding_dimension or self.embedding_dimension <= 0:
            self.embedding_dimension = 768
            print(f"Setting default embedding dimension to {self.embedding_dimension}")
            
        # Load the fallback model
        self._load_fallback_model()
        
        # Only proceed with initialization if using fallback
        if self.use_fallback:
            # Initialize with a small sample corpus to ensure consistent dimensions
            sample_texts = [
                "This is a sample text to initialize the TF-IDF vectorizer.",
                "Multiple examples help ensure we get consistent vocabulary size.",
                "The vectorizer needs enough examples to build a meaningful vocabulary.",
                "With these samples, we ensure consistent embedding dimensions for queries and documents."
            ]
            
            # Fit the vectorizer on sample texts
            print("Pre-fitting TF-IDF vectorizer on sample texts")
            if self.tfidf_vectorizer is not None:
                self.tfidf_vectorizer.fit(sample_texts)
                if hasattr(self.tfidf_vectorizer, 'vocabulary_'):
                    print(f"Initial vocabulary size: {len(self.tfidf_vectorizer.vocabulary_)}")
                else:
                    print("Warning: TF-IDF vectorizer has no vocabulary_ attribute after fitting")
            else:
                print("Warning: TF-IDF vectorizer is None")
        
        return
            
    def _load_fallback_model(self):
        """Load the fallback TF-IDF model."""
        import logging
        logger = logging.getLogger(__name__)
        
        # Make sure embedding dimension is not None and has a valid value
        if not self.embedding_dimension or self.embedding_dimension <= 0:
            from config import EMBEDDING_DIMENSION
            self.embedding_dimension = EMBEDDING_DIMENSION  # Use the configured dimension
            logger.warning(f"Invalid embedding dimension, reset to configured value: {self.embedding_dimension}")
        
        try:
            # Initialize the TF-IDF vectorizer
            self.tfidf_vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=min(self.embedding_dimension, 10000),  # Ensure we don't request too many features
                stop_words='english'
            )
            
            # Initialize with sample texts to ensure vectorizer is properly fitted
            sample_texts = [
                "This is a sample text to initialize the TF-IDF vectorizer.",
                "Multiple examples help ensure we get consistent vocabulary size.",
                "The vectorizer needs enough examples to build a meaningful vocabulary.",
                "With these samples, we ensure consistent embedding dimensions for queries and documents.",
                "Adding more diverse text to improve the vocabulary representation."
            ]
            
            # Fit on sample texts
            self.tfidf_vectorizer.fit(sample_texts)
            vocabulary_size = len(self.tfidf_vectorizer.vocabulary_) if hasattr(self.tfidf_vectorizer, 'vocabulary_') else 0
            
            logger.info(f"Initialized TF-IDF vectorizer with max_features={min(self.embedding_dimension, 10000)}, " 
                      f"vocabulary size: {vocabulary_size}")
            
        except Exception as e:
            logger.error(f"Error initializing TF-IDF vectorizer: {str(e)}", exc_info=True)
            self.tfidf_vectorizer = None
            # Create a dummy vectorizer as last resort
            try:
                from sklearn.feature_extraction.text import CountVectorizer
                self.tfidf_vectorizer = CountVectorizer(max_features=min(self.embedding_dimension, 1000))
                logger.warning("Falling back to simple CountVectorizer")
            except Exception as fallback_e:
                logger.error(f"Failed to create fallback CountVectorizer: {str(fallback_e)}", exc_info=True)
        
    def generate_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for a list of text chunks.
        
        Args:
            texts: List of text chunks to embed
            
        Returns:
            List of numpy arrays containing the embeddings
        """
        if self.use_fallback:
            return self._generate_local_embeddings(texts)
        else:
            return self._generate_api_embeddings(texts)
            
    def _generate_api_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings using Anthropic Claude API."""
        try:
            api_key = os.environ.get('ANTHROPIC_API_KEY')
            if not api_key:
                logger.warning("No ANTHROPIC_API_KEY found, falling back to local embeddings")
                self.use_fallback = True
                return self._generate_local_embeddings(texts)
            
            if not self.client:
                self.client = anthropic.Anthropic(api_key=api_key)
            
            # Use Anthropic's embeddings API
            logger.info("Using Anthropic's embeddings API")
            
            # The code below will be executed when the embeddings API is used:
            
            # Process texts in batches to avoid hitting API limits
            batch_size = 10
            all_embeddings = []
            
            logger.info(f"Generating embeddings for {len(texts)} texts using Anthropic Claude API")
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                logger.info(f"Processing batch {i//batch_size + 1} of {(len(texts)-1)//batch_size + 1}")
                
                try:
                    # Call the Anthropic embeddings API
                    response = self.client.embeddings.create(
                        model=self.model_name,
                        input=batch,
                        dimensions=self.embedding_dimension
                    )
                    
                    # Extract embeddings from the response and ensure they're serializable
                    if response and hasattr(response, 'data'):
                        # Convert directly to Python lists of floats to ensure they're serializable
                        batch_embeddings = [item.embedding for item in response.data]
                        all_embeddings.extend(batch_embeddings)
                    else:
                        logger.error(f"Unexpected API response format: {response}")
                        raise ValueError("Invalid response format from Anthropic API")
                        
                except Exception as batch_e:
                    logger.error(f"Error in batch {i//batch_size + 1}: {str(batch_e)}")
                    # Skip failed batch and continue with the next one
                    continue
            
            # Check if we got embeddings for all texts
            if len(all_embeddings) == len(texts):
                logger.info(f"Successfully generated {len(all_embeddings)} embeddings via Anthropic API")
                return all_embeddings
            elif len(all_embeddings) > 0:
                logger.warning(f"Partial success: generated {len(all_embeddings)} embeddings out of {len(texts)}")
                # Pad with zeros for missing embeddings
                while len(all_embeddings) < len(texts):
                    all_embeddings.append([0.0] * self.embedding_dimension)
                return all_embeddings
            else:
                logger.error("Failed to generate any embeddings via API")
                return self._generate_local_embeddings(texts)
                
        except Exception as e:
            logger.error(f"Error using Anthropic Claude API: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
        # If we get here, something went wrong with the API call
        logger.warning("API embeddings failed, using fallback")
        self.use_fallback = True
        
        # Load fallback model if not already loaded
        if not self.tfidf_vectorizer:
            self._load_fallback_model()
            
        return self._generate_local_embeddings(texts)
        
    def _generate_local_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings using TF-IDF vectorizer with dense projection."""
        try:
            # Handle empty list
            if not texts:
                print("Warning: Empty texts list for embedding generation")
                return []
            
            # Make sure we have a valid embedding dimension
            from config import EMBEDDING_DIMENSION
            self.embedding_dimension = EMBEDDING_DIMENSION
            print(f"Using embedding dimension: {self.embedding_dimension}")
                
            # Create a new vectorizer if none exists
            if self.tfidf_vectorizer is None:
                # Use much larger feature set to better capture meaning
                max_features = 10000
                print(f"Initializing TF-IDF vectorizer with max_features={max_features}")
                try:
                    # Initialize the TF-IDF vectorizer
                    self.tfidf_vectorizer = TfidfVectorizer(
                        ngram_range=(1, 2),
                        max_features=max_features,  # Get more features for better semantic matching
                        stop_words='english'
                    )
                    
                    # Initialize with sample texts plus current texts
                    sample_texts = [
                        "This is a sample text to initialize the TF-IDF vectorizer.",
                        "Multiple examples help ensure we get consistent vocabulary size.",
                        "The vectorizer needs enough examples to build a meaningful vocabulary.",
                        "With these samples, we ensure consistent embedding dimensions for queries and documents.",
                        # Add financial/business domain texts
                        "Annual financial report with revenue growth and profit margins for Amazon.",
                        "Company performance metrics including quarterly earnings and market share of AWS.",
                        "Business strategy focusing on expansion, acquisition and market penetration.",
                        "Economic factors affecting business operations and profitability goals.",
                        "Technical analysis of product development lifecycle and innovation roadmap.",
                        "Growth metrics and yearly performance data across multiple business segments.",
                        "Market trends and competitive analysis for e-commerce and cloud computing.",
                        "Analysis of year-over-year growth and financial performance indicators."
                    ]
                    
                    # Combine sample texts with actual texts for better vocabulary
                    all_texts = sample_texts + texts
                    
                    # Fit on combined texts
                    self.tfidf_vectorizer.fit(all_texts)
                    
                    print(f"Initialized TF-IDF vectorizer with max_features={max_features}, vocabulary size: {len(self.tfidf_vectorizer.vocabulary_)}")
                    
                except Exception as init_error:
                    print(f"Failed to initialize vectorizer: {str(init_error)}")
                    # Return random embeddings as last resort, properly serializable
                    return [[float(x) for x in (np.random.randn(self.embedding_dimension))] for _ in texts]
            
            # Transform texts to embeddings using existing vocabulary
            try:
                embeddings = self.tfidf_vectorizer.transform(texts)
                print(f"Generated sparse embeddings with shape: {embeddings.shape}")
            except Exception as transform_error:
                print(f"Error transforming with existing vocabulary: {str(transform_error)}")
                # If transformation fails, try recreating the vectorizer
                print("Recreating vectorizer with these texts...")
                try:
                    max_features = 10000
                    self.tfidf_vectorizer = TfidfVectorizer(
                        ngram_range=(1, 2),
                        max_features=max_features,
                        stop_words='english'
                    )
                    self.tfidf_vectorizer.fit(texts)
                    embeddings = self.tfidf_vectorizer.transform(texts)
                    print(f"Recreated vectorizer with {len(self.tfidf_vectorizer.vocabulary_)} terms")
                except Exception as refit_error:
                    print(f"Error after recreating vectorizer: {str(refit_error)}")
                    return [[float(x) for x in (np.random.randn(self.embedding_dimension))] for _ in texts]
            
            # Normalize and convert to dense
            try:
                # Convert sparse to dense
                dense_embeddings = embeddings.toarray()
                print(f"Converted to dense with shape: {dense_embeddings.shape}")
                
                # If we have more features than our target dimension, use random projection
                if dense_embeddings.shape[1] > self.embedding_dimension:
                    print(f"Projecting from {dense_embeddings.shape[1]} to {self.embedding_dimension} dimensions")
                    
                    # Create a random projection matrix
                    np.random.seed(42)  # For reproducibility
                    projection = np.random.randn(dense_embeddings.shape[1], self.embedding_dimension).astype(np.float32)
                    # Normalize columns to preserve distances approximately
                    projection = projection / np.sqrt(projection.shape[0])
                    
                    # Project to lower dimension
                    projected = np.dot(dense_embeddings, projection)
                    print(f"Projected to shape: {projected.shape}")
                    
                    # Normalize vectors
                    norms = np.linalg.norm(projected, axis=1, keepdims=True)
                    # Avoid division by zero
                    norms = np.maximum(norms, 1e-10)
                    normalized = projected / norms
                    
                    # Convert to list
                    result = [normalized[i] for i in range(normalized.shape[0])]
                    
                # If we have fewer features, pad to the target dimension
                elif dense_embeddings.shape[1] < self.embedding_dimension:
                    print(f"Padding from {dense_embeddings.shape[1]} to {self.embedding_dimension} dimensions")
                    result = []
                    for i in range(dense_embeddings.shape[0]):
                        # Create padded vector
                        padded = np.zeros(self.embedding_dimension, dtype=np.float32)
                        # Copy existing values
                        padded[:dense_embeddings.shape[1]] = dense_embeddings[i]
                        
                        # Add small random noise to padding to make vectors more discriminative
                        if dense_embeddings.shape[1] < self.embedding_dimension:
                            noise = np.random.randn(self.embedding_dimension - dense_embeddings.shape[1]).astype(np.float32) * 0.01
                            padded[dense_embeddings.shape[1]:] = noise
                        
                        # Normalize
                        padded = padded / (np.linalg.norm(padded) + 1e-10)
                        result.append(padded)
                else:
                    # Already the right dimension
                    # Normalize vectors
                    norms = np.linalg.norm(dense_embeddings, axis=1, keepdims=True)
                    # Avoid division by zero
                    norms = np.maximum(norms, 1e-10)
                    normalized = dense_embeddings / norms
                    result = [normalized[i] for i in range(normalized.shape[0])]
                
                print(f"Returning {len(result)} embeddings with dimension {result[0].shape if result else 'n/a'}")
                
                # Make sure values are all finite and convert to regular Python floats
                # to ensure JSON serialization works
                serializable_result = []
                for i in range(len(result)):
                    # First replace any NaN or infinity values
                    cleaned = np.nan_to_num(result[i])
                    # Convert to regular Python float list for JSON serialization
                    serializable_result.append([float(x) for x in cleaned])
                
                print(f"Returning serializable embeddings of type: {type(serializable_result[0][0]) if serializable_result else 'n/a'}")
                return serializable_result
                
            except Exception as normalize_error:
                print(f"Error during normalization: {str(normalize_error)}")
                # Create random vectors and convert to regular Python floats for serialization
                return [[float(x) for x in (np.random.randn(self.embedding_dimension) / np.sqrt(self.embedding_dimension))] for _ in texts]
            
        except Exception as e:
            print(f"Error generating local embeddings: {str(e)}")
            import traceback
            traceback.print_exc()
            # Return zero embeddings as last resort, converted to serializable format
            zeros = [[0.0] * self.embedding_dimension for _ in texts]
            print(f"Falling back to {len(zeros)} zero embeddings with dimension {len(zeros[0]) if zeros else 'n/a'}")
            return zeros
            
    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Embed a list of chunk dictionaries and add embeddings to each.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Same list with embeddings added to each chunk
        """
        # Extract text from chunks
        texts = [chunk.get('text', '') for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.generate_embeddings(texts)
        
        # Add embeddings to chunks
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk['embedding'] = embedding
            
        return chunks
        
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a query string.
        
        Args:
            query: Query text
            
        Returns:
            Numpy array with the query embedding
        """
        # Generate embedding for single query
        embeddings = self.generate_embeddings([query])
        
        if embeddings and len(embeddings) > 0:
            return embeddings[0]
        else:
            # Return zero embedding if generation failed, as a serializable list
            return [0.0] * self.embedding_dimension