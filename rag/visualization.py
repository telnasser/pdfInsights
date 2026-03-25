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
        self.pca = PCA(n_components=50)  # For initial dimensionality reduction
        self.tsne = TSNE(n_components=2, random_state=42)  # For 2D projection
        
    def create_chunk_visualization(self, doc_id: str, vector_store) -> Dict[str, Any]:
        """
        Create visualization data for document chunks.
        
        Args:
            doc_id: Document ID
            vector_store: Vector store instance
            
        Returns:
            Dictionary with visualization data
        """
        # Get document chunks from vector store
        chunks = vector_store.get_document_chunks(doc_id)
        
        if not chunks:
            return {'error': 'No chunks found for document'}
        
        # Extract embeddings
        embeddings = [chunk['embedding'] for chunk in chunks if 'embedding' in chunk]
        
        if not embeddings:
            return {'error': 'No embeddings found in chunks'}
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings)
        
        # Reduce dimensionality
        try:
            if embeddings_array.shape[1] > 50:
                # First reduce with PCA to 50 dimensions
                reduced_embeddings = self.pca.fit_transform(embeddings_array)
            else:
                reduced_embeddings = embeddings_array
                
            # Then use t-SNE for 2D visualization
            projections = self.tsne.fit_transform(reduced_embeddings)
            
            # Calculate similarities between chunks
            similarities = self._calculate_chunk_similarities(embeddings_array)
            
            # Create projection data
            projection_data = []
            for i, (x, y) in enumerate(projections):
                projection_data.append({
                    'x': float(x),
                    'y': float(y),
                    'chunk': self._prepare_chunk_for_json(chunks[i])
                })
                
            # Create similarity data (connections between chunks)
            similarity_data = []
            for i, j, score in similarities:
                if score > 0.7:  # Only include strong connections
                    similarity_data.append({
                        'source': i,
                        'target': j,
                        'score': float(score)
                    })
                    
            return {
                'projections': projection_data,
                'similarities': similarity_data
            }
            
        except Exception as e:
            return {'error': str(e)}
            
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