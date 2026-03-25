import os
import json
import pickle
import shutil
import logging
from typing import List, Dict, Any, Optional

import numpy as np
import faiss

from config import VECTOR_DB_PATH

logger = logging.getLogger(__name__)

class VectorStore:
    """
    Vector database for storing and retrieving document chunk embeddings.
    Uses FAISS backend.
    """
    
    def __init__(self, db_path: str = VECTOR_DB_PATH):
        """
        Initialize the vector store.
        
        Args:
            db_path: Path to store vector database files
        """
        self.db_path = db_path
        self.index_dir = os.path.join(db_path, 'faiss_index')
        self.metadata_path = os.path.join(db_path, 'metadata')
        
        # Create directories if they don't exist
        os.makedirs(db_path, exist_ok=True)
        os.makedirs(self.index_dir, exist_ok=True)
        os.makedirs(self.metadata_path, exist_ok=True)
        
        # Initialize FAISS index
        self._init_faiss()
        
    def _init_faiss(self):
        """Initialize FAISS vector store."""
        # Placeholder for FAISS index (loaded on-demand)
        self.faiss_index = None
        self.embedding_size = None
        
    def add_document(self, doc_id: str, chunks: List[Dict[str, Any]]) -> bool:
        """
        Add document chunks with embeddings to the vector store.
        
        Args:
            doc_id: Unique document identifier
            chunks: List of chunk dictionaries with embeddings
            
        Returns:
            True if successful, False otherwise
        """
        return self._add_to_faiss(doc_id, chunks)
        
    def _add_to_faiss(self, doc_id: str, chunks: List[Dict[str, Any]]) -> bool:
        """Add document to FAISS index."""
        try:
            # Extract embeddings and metadata
            embeddings = []
            chunk_metadata = []
            
            logger.info(f"Processing {len(chunks)} chunks for document {doc_id}")
            
            for i, chunk in enumerate(chunks):
                if 'embedding' not in chunk:
                    logger.warning(f"Chunk {i} missing embedding for document {doc_id}, skipping")
                    continue
                
                # Validate embedding data type and shape
                embedding = chunk['embedding']
                if not isinstance(embedding, np.ndarray):
                    try:
                        embedding = np.array(embedding, dtype=np.float32)
                        logger.info(f"Converted embedding to numpy array for chunk {i}")
                    except Exception as e:
                        logger.error(f"Error converting embedding to numpy array for chunk {i}: {str(e)}")
                        continue
                
                # Check for NaN values
                if np.isnan(embedding).any():
                    logger.warning(f"Embedding for chunk {i} contains NaN values, replacing with zeros")
                    embedding = np.nan_to_num(embedding)
                    
                embeddings.append(embedding)
                
                # Create metadata without embedding (too large)
                metadata = {k: v for k, v in chunk.items() if k != 'embedding'}
                
                # Add identifiers if not present
                if 'chunk_id' not in metadata:
                    metadata['chunk_id'] = i
                if 'document_id' not in metadata:
                    metadata['document_id'] = doc_id
                    
                chunk_metadata.append(metadata)
                
            if not embeddings:
                logger.error(f"No valid embeddings found for document {doc_id}")
                return False
                
            # Convert to numpy array
            try:
                embeddings_array = np.array(embeddings).astype('float32')
                logger.info(f"Created embeddings array with shape {embeddings_array.shape} for document {doc_id}")
            except Exception as e:
                logger.error(f"Error creating embeddings array: {str(e)}")
                return False
            
            # Import the configured embedding dimension
            from config import EMBEDDING_DIMENSION
            
            # Set up embedding size based on configuration
            self.embedding_size = EMBEDDING_DIMENSION
            logger.info(f"Using configured embedding dimension: {self.embedding_size}")
            
            # Fix dimension mismatch if present
            if self.embedding_size != embeddings_array.shape[1]:
                error_msg = f"Embedding dimension mismatch: expected {self.embedding_size}, got {embeddings_array.shape[1]}"
                logger.warning(error_msg)
                
                # Try to fix dimension mismatch
                if embeddings_array.shape[1] > self.embedding_size:
                    logger.warning(f"Truncating embeddings from {embeddings_array.shape[1]} to {self.embedding_size}")
                    embeddings_array = embeddings_array[:, :self.embedding_size]
                else:
                    logger.warning(f"Padding embeddings from {embeddings_array.shape[1]} to {self.embedding_size}")
                    padding = np.zeros((embeddings_array.shape[0], self.embedding_size - embeddings_array.shape[1]), dtype=np.float32)
                    embeddings_array = np.hstack((embeddings_array, padding))
                
            # Load or create index
            index = self._load_or_create_index()
            
            # Get current index size
            current_index_size = index.ntotal
            
            # Normalize vectors and handle edge cases for better similarity matching
            embeddings_array = np.nan_to_num(embeddings_array)  # Replace NaN with zeros
            norms = np.linalg.norm(embeddings_array, axis=1)
            mask = norms > 0
            embeddings_array[mask] = embeddings_array[mask] / norms[mask, np.newaxis]
            embeddings_array[~mask] = 0  # Set zero vectors for zero norms
            
            # Add normalized embeddings to index
            index.add(embeddings_array)
            
            # Save metadata for the chunks
            metadata_file = os.path.join(self.metadata_path, f"{doc_id}.json")
            
            # Add index position to metadata
            for i, meta in enumerate(chunk_metadata):
                meta['index_position'] = current_index_size + i
                
            # Save metadata
            with open(metadata_file, 'w') as f:
                json.dump(chunk_metadata, f)
                
            # Save updated index
            self._save_index(index)
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding document to FAISS: {e}")
            return False
            
    def _load_or_create_index(self):
        """Load existing index or create a new one."""
        # Make sure the directory exists
        os.makedirs(self.index_dir, exist_ok=True)
        
        # Use a specific filename for the index
        index_file = os.path.join(self.index_dir, 'faiss.index')
        
        logger.info(f"Checking for index at: {index_file}")
        
        # Import the configured dimension
        from config import INDEX_DIMENSION, EMBEDDING_DIMENSION
        
        if os.path.exists(index_file) and self.faiss_index is None:
            try:
                # Try to load existing index
                logger.info(f"Loading existing FAISS index from {index_file}")
                self.faiss_index = faiss.read_index(index_file)
                logger.info(f"Successfully loaded index with {self.faiss_index.ntotal} vectors")
                
                # If the dimensions don't match, we need to rebuild the index
                if self.faiss_index.d != EMBEDDING_DIMENSION:
                    logger.warning(f"Index dimension ({self.faiss_index.d}) doesn't match configured dimension ({EMBEDDING_DIMENSION})")
                    logger.warning("Will use the existing index for now")
                
                # Always use the configured embedding dimension
                self.embedding_size = EMBEDDING_DIMENSION
                logger.info(f"Using configured embedding dimension: {self.embedding_size}")
                
                return self.faiss_index
            except Exception as e:
                logger.error(f"Error loading FAISS index: {e}")
                # Fall back to creating a new one
                pass
        else:
            if not os.path.exists(index_file):
                logger.info(f"Index file does not exist at {index_file}, will create new index")
            elif self.faiss_index is not None:
                logger.info(f"Index already loaded in memory with {self.faiss_index.ntotal} vectors")
                
        # Create new index if it doesn't exist or couldn't be loaded
        if self.faiss_index is None:
            if self.embedding_size is None:
                self.embedding_size = EMBEDDING_DIMENSION  # Use the configured dimension
            logger.info(f"Creating new FAISS index with dimension {self.embedding_size}")
            # Using IndexFlatIP for cosine similarity instead of IndexFlatL2 for L2 distance
            # IP = Inner Product, which is cosine similarity when vectors are normalized
            self.faiss_index = faiss.IndexFlatIP(self.embedding_size)
            
        return self.faiss_index
        
    def _save_index(self, index):
        """Save FAISS index to disk."""
        # Make sure directory exists
        os.makedirs(self.index_dir, exist_ok=True)
        
        # Use a specific filename for the index
        index_file = os.path.join(self.index_dir, 'faiss.index')
        
        logger.info(f"Saving FAISS index with {index.ntotal} vectors to {index_file}")
        faiss.write_index(index, index_file)
        
    def search(self, query_embedding: np.ndarray, doc_id: Optional[str] = None, 
               top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using a query embedding.
        
        Args:
            query_embedding: Query embedding vector
            doc_id: Optional document ID to limit search scope
            top_k: Number of results to return
            
        Returns:
            List of chunk dictionaries with similarity scores
        """
        return self._search_faiss(query_embedding, doc_id, top_k)
        
    def _search_faiss(self, query_embedding: np.ndarray, doc_id: Optional[str], top_k: int) -> List[Dict[str, Any]]:
        """Search using FAISS index."""
        try:
            # Make sure we have valid data
            if query_embedding is None:
                logger.error("Query embedding is None")
                return []
                
            # Print shape before conversion for debugging
            try:
                logger.info(f"Original query embedding shape: {query_embedding.shape}, type: {type(query_embedding)}")
            except:
                logger.info(f"Original query embedding type: {type(query_embedding)}")
                
            # Prepare query embedding - ensure it's a 2D array of float32
            if isinstance(query_embedding, list):
                query_embedding = np.array(query_embedding)
            
            # Ensure it's a proper 2D array with the correct shape
            if len(query_embedding.shape) == 1:
                query_embedding = np.array([query_embedding]).astype('float32')
            else:
                query_embedding = query_embedding.astype('float32')
            
            # Load index
            index = self._load_or_create_index()
            
            # Check if index is empty or None
            if index is None:
                logger.error("FAISS index is None")
                return []
                
            try:
                if index.ntotal == 0:
                    logger.warning("FAISS index is empty. No documents have been indexed.")
                    return []
            except Exception as e:
                logger.error(f"Error checking index.ntotal: {str(e)}")
                return []
                
            # Print dimensions for debugging
            logger.info(f"Query embedding shape: {query_embedding.shape}, index dimension: {index.d}")
            
            # Normalize query vector for cosine similarity with IndexFlatIP
            faiss.normalize_L2(query_embedding)
            
            # Check if dimensions match
            if query_embedding.shape[1] != index.d:
                logger.error(f"Dimension mismatch: query embedding dimension {query_embedding.shape[1]} does not match index dimension {index.d}")
                # Try to resize the embedding
                logger.info(f"Attempting to resize query embedding from {query_embedding.shape[1]} to {index.d}")
                
                # Pad or truncate to match the index dimension
                if query_embedding.shape[1] < index.d:
                    # Pad with zeros
                    padding = np.zeros((1, index.d - query_embedding.shape[1])).astype('float32')
                    query_embedding = np.hstack((query_embedding, padding))
                else:
                    # Truncate
                    query_embedding = query_embedding[:, :index.d]
                    
                logger.info(f"Resized query embedding shape: {query_embedding.shape}")
            
            # Set search factors
            search_factor = 4  # Search 4x top_k to allow for filtering
            actual_k = min(top_k * search_factor, index.ntotal)
            
            logger.info(f"Searching for {actual_k} nearest neighbors in index with {index.ntotal} vectors")
            
            # Search
            distances, indices = index.search(query_embedding, actual_k)
            
            logger.info(f"Found {len(indices[0])} vectors with distances: {distances[0][:5]}...")
            
            # Collect all documents metadata
            all_metadata = {}
            doc_ids = []
            
            if doc_id:
                # Only load specific document metadata
                doc_ids = [doc_id]
                logger.info(f"Searching only within document: {doc_id}")
            else:
                # Load all document metadata
                doc_ids = self.get_all_documents()
                logger.info(f"Searching across all documents: {len(doc_ids)} documents found")
            
            # Verify metadata directory exists
            if not os.path.exists(self.metadata_path):
                os.makedirs(self.metadata_path, exist_ok=True)
                logger.warning(f"Metadata directory did not exist, created: {self.metadata_path}")
                
            for did in doc_ids:
                metadata_file = os.path.join(self.metadata_path, f"{did}.json")
                if os.path.exists(metadata_file):
                    with open(metadata_file, 'r') as f:
                        try:
                            doc_metadata = json.load(f)
                            if not doc_metadata:
                                logger.warning(f"Empty metadata for document: {did}")
                                continue
                                
                            for meta in doc_metadata:
                                if 'index_position' in meta:
                                    all_metadata[meta['index_position']] = meta
                        except json.JSONDecodeError:
                            logger.error(f"Invalid JSON in metadata file: {metadata_file}")
                else:
                    logger.warning(f"Metadata file not found: {metadata_file}")
                                
            # Filter and process results
            results = []
            
            # Create a mapping of indices to distances for easy lookup
            distance_map = {idx: distance for idx, distance in zip(indices[0], distances[0])}
            
            # First, collect all metadata matching our criteria
            matching_metadata = []
            for idx in indices[0]:
                # Skip if index not found in metadata
                if idx not in all_metadata:
                    continue
                    
                # Skip if filtering by doc_id and not matching
                metadata = all_metadata[idx].copy()  # Use copy to avoid modifying original
                if doc_id and metadata.get('document_id') != doc_id:
                    continue
                    
                # With IndexFlatIP, the 'distance' is actually the inner product similarity score
                # Higher values are better (unlike L2 distance where lower is better)
                # For normalized vectors, inner product equals cosine similarity
                score = float(distance_map[idx])
                
                # Convert inner product to range [0,1] for consistency
                # Inner product of normalized vectors is in range [-1,1]
                # Scale to [0,1] range: (score + 1) / 2
                # This makes it easier to apply threshold filtering consistently
                normalized_score = (score + 1) / 2
                
                metadata['score'] = normalized_score
                metadata['raw_similarity'] = score  # Keep the original score too
                
                matching_metadata.append(metadata)
                
            # Sort by score (highest first) and limit to top_k
            results = sorted(matching_metadata, key=lambda x: x['score'], reverse=True)[:top_k]
                    
            return results
            
        except Exception as e:
            logger.error(f"Error searching in FAISS: {str(e)}")
            if doc_id:
                logger.info(f"Checking if document {doc_id} has been properly indexed...")
                metadata_file = os.path.join(self.metadata_path, f"{doc_id}.json")
                if not os.path.exists(metadata_file):
                    logger.error(f"Document {doc_id} has not been indexed properly. Metadata file not found.")
                else:
                    logger.info(f"Metadata file exists for {doc_id}, but search failed.")
            else:
                doc_ids = self.get_all_documents() 
                logger.info(f"Found {len(doc_ids)} indexed documents: {doc_ids}")
                
            # Try creating a simple array of dummy vectors for testing
            try:
                logger.info("Attempting to create a test vector...")
                test_vector = np.random.rand(1, self.embedding_size).astype('float32')
                logger.info(f"Test vector created with shape: {test_vector.shape}")
            except Exception as test_e:
                logger.error(f"Failed to create test vector: {str(test_e)}")
                
            return []
            
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document and its chunks from the vector store.
        
        Args:
            doc_id: Document ID to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get document metadata
            metadata_file = os.path.join(self.metadata_path, f"{doc_id}.json")
            if not os.path.exists(metadata_file):
                return False
                
            # Read metadata to get index positions
            with open(metadata_file, 'r') as f:
                doc_metadata = json.load(f)
                
            # Remove metadata file
            os.remove(metadata_file)
            
            # Rebuild index without deleted document
            self._rebuild_index_without_document(doc_id, doc_metadata)
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            return False
            
    def _rebuild_index_without_document(self, doc_id: str, doc_metadata: List[Dict[str, Any]]):
        """Rebuild index without the specified document."""
        # Get all documents
        doc_ids = self.get_all_documents()
        
        if not doc_ids and doc_id not in doc_ids:
            return
            
        # Get index positions to remove
        positions_to_remove = set(meta.get('index_position') for meta in doc_metadata if 'index_position' in meta)
        
        # Create a new index
        old_index = self._load_or_create_index()
        # Using IndexFlatIP for cosine similarity instead of IndexFlatL2 for L2 distance
        new_index = faiss.IndexFlatIP(self.embedding_size)
        
        # Get all vectors and rebuild without the deleted document
        if old_index.ntotal > 0:
            # Get all vectors
            all_vectors = faiss.rev_swig_ptr(old_index.get_xb(), old_index.ntotal * old_index.d).reshape(old_index.ntotal, old_index.d)
            
            # Filter out vectors from the document to delete
            keep_indices = [i for i in range(old_index.ntotal) if i not in positions_to_remove]
            if keep_indices:
                new_vectors = all_vectors[keep_indices]
                
                # Normalize vectors for cosine similarity
                faiss.normalize_L2(new_vectors)
                
                # Add the normalized vectors to the new index
                new_index.add(new_vectors)
                
        # Update metadata for all remaining documents
        self._update_metadata_after_rebuild(positions_to_remove)
        
        # Save the new index
        self.faiss_index = new_index
        self._save_index(new_index)
        
    def _update_metadata_after_rebuild(self, removed_positions: set):
        """Update metadata after index rebuild."""
        # Get all documents
        doc_ids = self.get_all_documents()
        
        # Create a mapping of old positions to new positions
        position_map = {}
        new_position = 0
        
        # Make sure we have a valid index before trying to access ntotal
        if self.faiss_index is None:
            logger.error("FAISS index is None during metadata rebuild")
            return
            
        total_positions = self.faiss_index.ntotal + len(removed_positions)
        for old_position in range(total_positions):
            if old_position not in removed_positions:
                position_map[old_position] = new_position
                new_position += 1
                
        # Update each document's metadata
        for doc_id in doc_ids:
            metadata_file = os.path.join(self.metadata_path, f"{doc_id}.json")
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    doc_metadata = json.load(f)
                    
                # Update index positions
                for meta in doc_metadata:
                    if 'index_position' in meta and meta['index_position'] in position_map:
                        meta['index_position'] = position_map[meta['index_position']]
                        
                # Save updated metadata
                with open(metadata_file, 'w') as f:
                    json.dump(doc_metadata, f)
                    
    def get_all_documents(self) -> List[str]:
        """
        Get a list of all document IDs in the store.
        
        Returns:
            List of document IDs
        """
        try:
            # Get all metadata files
            metadata_files = os.listdir(self.metadata_path)
            
            # Extract document IDs from file names
            doc_ids = [f.split('.')[0] for f in metadata_files if f.endswith('.json')]
            
            return doc_ids
            
        except Exception as e:
            logger.error(f"Error getting all documents: {e}")
            return []
            
    def get_document_chunks(self, doc_id: str) -> List[Dict[str, Any]]:
        """
        Get all chunks for a specific document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            List of chunk dictionaries without embeddings
        """
        try:
            # Get document metadata
            metadata_file = os.path.join(self.metadata_path, f"{doc_id}.json")
            if not os.path.exists(metadata_file):
                return []
                
            # Read metadata
            with open(metadata_file, 'r') as f:
                doc_metadata = json.load(f)
                
            return doc_metadata
            
        except Exception as e:
            logger.error(f"Error getting document chunks: {e}")
            return []
            
    def clear(self) -> bool:
        """
        Clear all data from the vector store.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Remove all metadata files
            for filename in os.listdir(self.metadata_path):
                file_path = os.path.join(self.metadata_path, filename)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                    
            # Create new empty index
            # Using IndexFlatIP for cosine similarity instead of IndexFlatL2 for L2 distance
            self.faiss_index = faiss.IndexFlatIP(self.embedding_size or 768)
            self._save_index(self.faiss_index)
            
            return True
            
        except Exception as e:
            logger.error(f"Error clearing vector store: {e}")
            return False