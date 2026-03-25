import os
import sys
import logging
import numpy as np
import faiss
import json
from typing import List, Dict, Any

from config import VECTOR_DB_PATH, EMBEDDING_DIMENSION

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def rebuild_index():
    """
    Rebuild the FAISS index with the correct embedding dimension.
    This will extract all embeddings from the metadata files and recreate the index.
    """
    # Paths
    index_dir = os.path.join(VECTOR_DB_PATH, 'faiss_index')
    metadata_path = os.path.join(VECTOR_DB_PATH, 'metadata')
    index_file = os.path.join(index_dir, 'faiss.index')
    
    # Create directories if they don't exist
    os.makedirs(index_dir, exist_ok=True)
    os.makedirs(metadata_path, exist_ok=True)
    
    # Check if any metadata files exist
    if not os.path.exists(metadata_path) or not os.listdir(metadata_path):
        logger.warning("No metadata files found. Nothing to rebuild.")
        return
    
    # Create a new index with the correct dimension
    logger.info(f"Creating new FAISS index with dimension {EMBEDDING_DIMENSION}")
    new_index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
    
    # Get all document IDs
    doc_ids = [f.split('.')[0] for f in os.listdir(metadata_path) if f.endswith('.json')]
    logger.info(f"Found {len(doc_ids)} documents")
    
    # Process each document
    index_position_map = {}  # Map to track new index positions
    all_metadata = []
    
    for doc_id in doc_ids:
        metadata_file = os.path.join(metadata_path, f"{doc_id}.json")
        logger.info(f"Processing document: {doc_id}")
        
        try:
            # Load metadata
            with open(metadata_file, 'r') as f:
                doc_metadata = json.load(f)
            
            if not doc_metadata:
                logger.warning(f"Empty metadata for document: {doc_id}")
                continue
                
            # Extract and process embeddings
            embeddings = []
            doc_metadata_copy = []
            
            for meta in doc_metadata:
                # We need to have the embedding data to rebuild
                if 'embedding' not in meta:
                    logger.warning(f"Chunk without embedding in document {doc_id}, skipping")
                    continue
                
                # Get the embedding
                embedding = meta['embedding']
                
                # Convert to numpy array if needed
                if isinstance(embedding, list):
                    embedding = np.array(embedding)
                
                # Ensure it's a 1D array of float32
                if len(embedding.shape) > 1:
                    embedding = embedding.flatten()
                embedding = embedding.astype('float32')
                
                # Resize to match the configured dimension
                if embedding.shape[0] != EMBEDDING_DIMENSION:
                    logger.warning(f"Resizing embedding from {embedding.shape[0]} to {EMBEDDING_DIMENSION}")
                    if embedding.shape[0] < EMBEDDING_DIMENSION:
                        # Pad with zeros
                        padding = np.zeros(EMBEDDING_DIMENSION - embedding.shape[0], dtype=np.float32)
                        embedding = np.concatenate([embedding, padding])
                    else:
                        # Truncate
                        embedding = embedding[:EMBEDDING_DIMENSION]
                
                embeddings.append(embedding)
                
                # Make a copy of metadata without the embedding (too large)
                meta_copy = {k: v for k, v in meta.items() if k != 'embedding'}
                meta_copy['document_id'] = doc_id
                doc_metadata_copy.append(meta_copy)
            
            if not embeddings:
                logger.warning(f"No valid embeddings for document {doc_id}")
                continue
            
            # Convert to 2D array
            embeddings_array = np.array(embeddings).astype('float32')
            
            # Normalize vectors for cosine similarity
            norms = np.linalg.norm(embeddings_array, axis=1)
            mask = norms > 0
            embeddings_array[mask] = embeddings_array[mask] / norms[mask, np.newaxis]
            embeddings_array[~mask] = 0  # Set zero vectors for zero norms
            
            # Get current index size
            current_index_size = new_index.ntotal
            
            # Add to index
            new_index.add(embeddings_array)
            
            # Update index positions in metadata
            for i, meta in enumerate(doc_metadata_copy):
                meta['index_position'] = current_index_size + i
                index_position_map[meta.get('chunk_id', i)] = current_index_size + i
                all_metadata.append(meta)
            
            # Save updated metadata
            with open(metadata_file, 'w') as f:
                json.dump(doc_metadata_copy, f)
                
            logger.info(f"Added {len(embeddings)} vectors from document {doc_id}")
            
        except Exception as e:
            logger.error(f"Error processing document {doc_id}: {e}")
    
    # Save the new index
    logger.info(f"Saving new index with {new_index.ntotal} vectors")
    if new_index.ntotal > 0:
        faiss.write_index(new_index, index_file)
        logger.info(f"Successfully rebuilt index with {new_index.ntotal} vectors")
        return True
    else:
        logger.warning("No vectors added to index. Not saving empty index.")
        return False

if __name__ == "__main__":
    logger.info("Starting index rebuild process")
    success = rebuild_index()
    if success:
        logger.info("Index rebuild completed successfully")
    else:
        logger.warning("Index rebuild completed with warnings")