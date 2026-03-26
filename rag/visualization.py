import numpy as np
from typing import Dict, List, Any, Optional
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class Visualizer:
    """
    Creates visualizations for document chunks and their relationships.
    Implements dimensionality reduction for embedding visualization.
    """
    
    def __init__(self):
        """Initialize the visualizer."""
        pass
        
    def create_chunk_visualization(self, doc_id: str, vector_store) -> Dict[str, Any]:
        """
        Create visualization data for document chunks.
        
        Args:
            doc_id: Document ID
            vector_store: Vector store instance
            
        Returns:
            Dictionary with visualization data
        """
        # Get document chunks from vector store (metadata only – no embeddings stored)
        chunks = vector_store.get_document_chunks(doc_id)
        
        if not chunks:
            return {'error': 'No chunks found for document'}
        
        # Embeddings are not persisted in metadata; build lightweight positional layout
        n = len(chunks)

        # Simple positional layout arranged in a grid
        cols = max(1, int(n ** 0.5))
        projection_data = []
        for i, chunk in enumerate(chunks):
            row = i // cols
            col = i % cols
            projection_data.append({
                'x': float(col * 10),
                'y': float(row * 10),
                'chunk': self._prepare_chunk_for_json(chunk)
            })

        # Build sequential similarity links (each chunk connected to the next)
        similarity_data = []
        for i in range(n - 1):
            similarity_data.append({'source': i, 'target': i + 1, 'score': 0.8})

        return {
            'projections': projection_data,
            'similarities': similarity_data
        }
            
    def _calculate_chunk_similarities(self, embeddings: np.ndarray) -> List[tuple]:
        """
        Calculate similarities between chunk embeddings.
        
        Args:
            embeddings: Numpy array of embeddings
            
        Returns:
            List of (chunk_i, chunk_j, similarity_score) tuples
        """
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / norms
        
        # Calculate cosine similarity (dot product of normalized vectors)
        similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
        
        # Extract non-diagonal elements (chunk pairs)
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                similarities.append((i, j, similarity_matrix[i, j]))
                
        # Sort by similarity score (descending)
        similarities.sort(key=lambda x: x[2], reverse=True)
        
        return similarities[:50]  # Return top 50 similarities
        
    def _prepare_chunk_for_json(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare chunk data for JSON serialization.
        
        Args:
            chunk: Chunk dictionary
            
        Returns:
            JSON-serializable chunk dictionary
        """
        # Create a copy without embedding (too large for visualization)
        result = {k: v for k, v in chunk.items() if k != 'embedding'}
        
        # Add ID field if not present
        if 'id' not in result and 'chunk_id' in result:
            result['id'] = result['chunk_id']
        elif 'id' not in result:
            result['id'] = hash(result.get('text', ''))
            
        return result
        
    def analyze_chunk_distribution(self, doc_id: str, vector_store) -> Dict[str, Any]:
        """
        Analyze chunk size distribution for a document.
        
        Args:
            doc_id: Document ID
            vector_store: Vector store instance
            
        Returns:
            Dictionary with analysis data
        """
        # Get document chunks from vector store
        chunks = vector_store.get_document_chunks(doc_id)
        
        if not chunks:
            return {'error': 'No chunks found for document'}
            
        # Get chunk sizes
        chunk_sizes = [len(chunk.get('text', '')) for chunk in chunks]
        
        # Calculate statistics
        avg_size = sum(chunk_sizes) / len(chunk_sizes)
        max_size = max(chunk_sizes)
        min_size = min(chunk_sizes)
        
        return {
            'chunkSizes': chunk_sizes,
            'stats': {
                'avgSize': avg_size,
                'maxSize': max_size,
                'minSize': min_size,
                'totalChunks': len(chunks)
            }
        }