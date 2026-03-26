import os
import logging
import pickle
from typing import List, Dict, Any, Union, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import anthropic

from config import EMBEDDING_DIMENSION, EMBEDDING_API_KEY, CLAUDE_EMBEDDING_MODEL

logger = logging.getLogger(__name__)

_VECTORIZER_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'vector_db', 'tfidf_vectorizer.pkl')
_PROJECTION_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'vector_db', 'tfidf_projection.pkl')


def _save_vectorizer(vectorizer, projection):
    """Persist the fitted TF-IDF vectorizer and projection matrix to disk."""
    try:
        os.makedirs(os.path.dirname(_VECTORIZER_PATH), exist_ok=True)
        with open(_VECTORIZER_PATH, 'wb') as f:
            pickle.dump(vectorizer, f)
        with open(_PROJECTION_PATH, 'wb') as f:
            pickle.dump(projection, f)
        logger.info("Saved TF-IDF vectorizer (vocab=%d) to disk", len(vectorizer.vocabulary_))
    except Exception as e:
        logger.error("Failed to save vectorizer: %s", e)


def _load_vectorizer():
    """Load the TF-IDF vectorizer and projection matrix from disk, or return (None, None)."""
    try:
        if os.path.exists(_VECTORIZER_PATH) and os.path.exists(_PROJECTION_PATH):
            with open(_VECTORIZER_PATH, 'rb') as f:
                vectorizer = pickle.load(f)
            with open(_PROJECTION_PATH, 'rb') as f:
                projection = pickle.load(f)
            logger.info("Loaded TF-IDF vectorizer (vocab=%d) from disk", len(vectorizer.vocabulary_))
            return vectorizer, projection
    except Exception as e:
        logger.error("Failed to load vectorizer from disk: %s", e)
    return None, None

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
        self._projection = None  # shared random-projection matrix (seed=42)
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
                while len(all_embeddings) < len(texts):
                    all_embeddings.append([0.0] * self.embedding_dimension)
                return all_embeddings
            else:
                logger.error("Failed to generate any embeddings via API — switching to local TF-IDF permanently")
                self.use_fallback = True
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
        
    def _ensure_vectorizer(self, fit_texts: List[str] = None, force_refit: bool = False):
        """
        Ensure self.tfidf_vectorizer and self._projection are ready.

        Priority:
          1. Already in memory and not forced refit → use as-is.
          2. force_refit=True → refit on fit_texts, save to disk.
          3. Not in memory → try loading from disk.
          4. Nothing on disk → fit on fit_texts (or empty corpus), save.
        """
        from config import EMBEDDING_DIMENSION
        self.embedding_dimension = EMBEDDING_DIMENSION

        if self.tfidf_vectorizer is not None and self._projection is not None and not force_refit:
            return

        if force_refit and fit_texts:
            self._refit_and_save(fit_texts)
            return

        # Try loading from disk first
        vectorizer, projection = _load_vectorizer()
        if vectorizer is not None and projection is not None:
            self.tfidf_vectorizer = vectorizer
            self._projection = projection
            return

        # Nothing on disk — fit from scratch using provided texts
        if fit_texts:
            self._refit_and_save(fit_texts)
        else:
            logger.warning("No texts and no saved vectorizer — creating minimal fallback vectorizer")
            self._refit_and_save([
                "document text placeholder for initialization purposes only"
            ])

    def _refit_and_save(self, texts: List[str]):
        """Fit vectorizer on texts + load all DB chunks, then save to disk."""
        from config import EMBEDDING_DIMENSION
        self.embedding_dimension = EMBEDDING_DIMENSION

        # Pull all existing chunks from DB to ensure vocabulary covers everything
        corpus = list(texts)
        try:
            from app import db
            from sqlalchemy import text as sa_text
            rows = db.session.execute(sa_text("SELECT text FROM chunk")).fetchall()
            for row in rows:
                if row[0]:
                    corpus.append(row[0])
        except Exception as e:
            logger.warning("Could not load existing chunks for vectorizer refit: %s", e)

        max_features = 10000
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=max_features,
            stop_words='english',
        )
        vectorizer.fit(corpus)
        vocab_size = len(vectorizer.vocabulary_)
        logger.info("Fitted TF-IDF vectorizer on %d texts, vocab=%d", len(corpus), vocab_size)

        # Build the random projection matrix with fixed seed so it's deterministic
        np.random.seed(42)
        projection = np.random.randn(vocab_size, self.embedding_dimension).astype(np.float32)
        projection /= np.sqrt(vocab_size)

        self.tfidf_vectorizer = vectorizer
        self._projection = projection
        _save_vectorizer(vectorizer, projection)

    def _generate_local_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings using TF-IDF vectorizer with dense projection."""
        try:
            if not texts:
                logger.warning("Empty texts list for embedding generation")
                return []

            from config import EMBEDDING_DIMENSION
            self.embedding_dimension = EMBEDDING_DIMENSION

            # Ensure vectorizer is ready.
            # When called with many texts it's a document batch → force refit so
            # the vocabulary is always built from the full corpus.
            # When called with a single text it's a query → load from disk.
            self._ensure_vectorizer(fit_texts=texts, force_refit=len(texts) > 1)

            # Project texts into embedding space
            try:
                sparse = self.tfidf_vectorizer.transform(texts)
            except Exception as e:
                logger.error("TF-IDF transform failed (%s); refitting on current texts", e)
                self._refit_and_save(texts)
                sparse = self.tfidf_vectorizer.transform(texts)

            dense = sparse.toarray().astype(np.float32)   # (N, vocab)

            vocab_size = dense.shape[1]
            proj = self._projection

            # Rebuild projection if shape doesn't match (e.g. vocab changed after refit)
            if proj is None or proj.shape[0] != vocab_size or proj.shape[1] != self.embedding_dimension:
                np.random.seed(42)
                proj = np.random.randn(vocab_size, self.embedding_dimension).astype(np.float32)
                proj /= np.sqrt(vocab_size)
                self._projection = proj
                # Persist updated projection
                _save_vectorizer(self.tfidf_vectorizer, proj)

            projected = dense @ proj                        # (N, 768)
            norms = np.linalg.norm(projected, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            normalized = projected / norms                  # (N, 768) unit vectors

            result = []
            for i in range(normalized.shape[0]):
                cleaned = np.nan_to_num(normalized[i])
                result.append([float(x) for x in cleaned])

            logger.info("Generated %d local embeddings (vocab=%d → dim=%d)",
                        len(result), vocab_size, self.embedding_dimension)
            return result

        except Exception as e:
            logger.error("Error generating local embeddings: %s", e, exc_info=True)
            zeros = [[0.0] * self.embedding_dimension for _ in texts]
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