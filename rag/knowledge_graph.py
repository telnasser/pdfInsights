"""
Knowledge Graph module for document analysis.
This module extracts entities and relationships from chunks and builds a knowledge graph.
"""
import os
import json
import spacy
import networkx as nx
from typing import List, Dict, Any, Tuple, Set
from collections import defaultdict
from pyvis.network import Network

# Load spaCy model for entity recognition
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    # Fallback if model isn't available
    print("Warning: spaCy model not found. Using basic model.")
    nlp = spacy.blank('en')


class KnowledgeGraph:
    """
    Knowledge Graph for document analysis.
    
    Extracts entities and relationships from document chunks,
    builds a graph representation, and allows querying the graph.
    """

    # ── LLM extraction prompt ─────────────────────────────────────────────────
    _LLM_PROMPT = """Extract all important named entities from the resume/document text below.
Return ONLY a valid JSON array — no explanation, no markdown fences.

Entity types to use:
- PERSON       : person names
- COMPANY      : employer or client company names (e.g. "Infoblox", "MBC Group", "STC")
- UNIVERSITY   : academic institutions (e.g. "German Jordanian University")
- ROLE         : job titles (e.g. "Director of Software Engineering")
- DATE_RANGE   : employment or study periods (e.g. "March 2025 - Present", "2018–2022")
- TECHNOLOGY   : tools, languages, frameworks, cloud services (e.g. "AWS", "Kubernetes", "Python")
- LOCATION     : cities, countries, regions
- SKILL        : competencies (e.g. "Agile", "Team Leadership")
- PRODUCT      : products or services launched (e.g. "Shahid", "GraphRAG platform")
- DEGREE       : academic qualifications (e.g. "Master of Business Administration")

Rules:
- Use the clean canonical name (no trailing punctuation, no "the", no "a/an").
- If a company name appears with ".com" or similar suffix strip it: "Infoblox.com" → "Infoblox".
- Include every employer, every technology mentioned, every date range.
- Maximum 60 entities total.

Text:
{text}

JSON array:"""

    def __init__(self, db_path: str = "./knowledge_graph"):
        """
        Initialize the knowledge graph.
        
        Args:
            db_path: Path to store knowledge graph data
        """
        self.db_path = db_path
        self.graph = nx.Graph()
        self.entity_counts = defaultdict(int)
        self.entity_contexts = defaultdict(set)
        self.document_entities = defaultdict(set)

        # Simple in-memory cache: sha256(text) → entity list
        self._llm_entity_cache: Dict[str, List[Tuple[str, str]]] = {}

        # Initialise the Anthropic client once
        self._anthropic_client = None
        try:
            import anthropic as _anthropic
            _api_key = os.environ.get('ANTHROPIC_API_KEY')
            if _api_key:
                self._anthropic_client = _anthropic.Anthropic(api_key=_api_key)
        except Exception as _e:
            print(f"KnowledgeGraph: Anthropic unavailable — will use spaCy NER ({_e})")
        
        # Create directory if it doesn't exist
        os.makedirs(db_path, exist_ok=True)
        
        # Try to load existing graph if available
        self._load_graph()
    
    def add_document(self, doc_id: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process document chunks and add entities and relationships to the graph.
        
        Args:
            doc_id: Document ID
            chunks: List of document chunks with text and metadata
            
        Returns:
            Statistics about extracted entities and relationships
        """
        # Clear previous entities for this document if it's being reprocessed
        if doc_id in self.document_entities:
            self._remove_document_entities(doc_id)
        
        # Extract entities and relationships from chunks
        all_entities = set()
        entity_pairs = []
        
        for chunk in chunks:
            chunk_text = chunk.get('text', '')
            chunk_idx = chunk.get('chunk_index', 0)
            
            if not chunk_text:
                continue
                
            # Extract entities — use LLM during ingestion for high-quality results
            entities = self._extract_entities_llm(chunk_text)

            # Add entities to the graph
            for entity, entity_type in entities:
                if self.graph.has_node(entity):
                    # Update existing node — do NOT overwrite, just extend sets
                    node_data = self.graph.nodes[entity]
                    node_data.setdefault('doc_ids', set()).add(doc_id)
                    node_data.setdefault('chunk_indices', set()).add(chunk_idx)
                else:
                    # Brand-new node
                    self.graph.add_node(entity,
                                        type=entity_type,
                                        doc_ids=set([doc_id]),
                                        chunk_indices=set([chunk_idx]))

                # Update entity metadata
                self.entity_counts[entity] += 1
                self.entity_contexts[entity].add(chunk_idx)
                self.document_entities[doc_id].add(entity)
                all_entities.add(entity)
            
            # Extract relationships between entities in the same chunk
            if len(entities) > 1:
                for i in range(len(entities)):
                    for j in range(i+1, len(entities)):
                        entity1 = entities[i][0]
                        entity2 = entities[j][0]
                        
                        # Add edge if it doesn't exist
                        if not self.graph.has_edge(entity1, entity2):
                            self.graph.add_edge(entity1, entity2, 
                                               weight=1,
                                               chunks=set([chunk_idx]),
                                               doc_ids=set([doc_id]))
                        else:
                            # Update edge weight and metadata
                            self.graph[entity1][entity2]['weight'] += 1
                            self.graph[entity1][entity2]['chunks'].add(chunk_idx)
                            self.graph[entity1][entity2]['doc_ids'].add(doc_id)
                        
                        entity_pairs.append((entity1, entity2))
        
        # Save the updated graph
        self._save_graph()
        
        # Return statistics
        return {
            'doc_id': doc_id,
            'entity_count': len(all_entities),
            'relationship_count': len(entity_pairs),
            'top_entities': self._get_top_entities(doc_id, limit=10)
        }
    
    # ── LLM-powered entity extraction (used during document ingestion) ─────────

    def _extract_entities_llm(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract entities using Claude Haiku for context-aware, high-quality
        extraction of companies, roles, dates, technologies, etc.

        Falls back to spaCy if the API is unavailable or the call fails.
        Results are cached in-memory by SHA-256 of the text.
        """
        import hashlib, json, re as _re

        cache_key = hashlib.sha256(text.encode()).hexdigest()
        if cache_key in self._llm_entity_cache:
            return self._llm_entity_cache[cache_key]

        if self._anthropic_client is None:
            result = self._extract_entities(text)
            self._llm_entity_cache[cache_key] = result
            return result

        try:
            prompt = self._LLM_PROMPT.format(text=text[:4000])
            response = self._anthropic_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()

            # Robustly extract the full JSON array even if the model wraps it in text.
            # Use greedy match from first '[' to last ']'.
            json_match = _re.search(r'\[.*\]', raw, _re.DOTALL)
            if not json_match:
                raise ValueError(f"No JSON array found in LLM response: {raw[:200]}")

            entities_raw = json.loads(json_match.group())
            entities: List[Tuple[str, str]] = []
            seen: set = set()
            for item in entities_raw:
                # Handle both {"name":..,"type":..} dicts AND plain strings
                if isinstance(item, dict):
                    name = str(item.get('name', '') or '').strip().strip('.,;:')
                    etype = str(item.get('type', 'CONCEPT') or 'CONCEPT').strip().upper()
                elif isinstance(item, str):
                    name = item.strip().strip('.,;:')
                    etype = 'CONCEPT'
                else:
                    continue
                # Normalise name: strip domain suffixes
                name = _re.sub(
                    r'\.(com|net|org|io|co|ae|sa|jo|ca|uk|us|gov|edu)$', '', name, flags=_re.I
                ).strip()
                if not name or len(name) < 2 or len(name) > 120:
                    continue
                key = name.lower()
                if key in seen:
                    continue
                seen.add(key)
                entities.append((name, etype))

            print(f"LLM extracted {len(entities)} entities from chunk "
                  f"(first: {entities[:3]})")
            self._llm_entity_cache[cache_key] = entities
            return entities

        except Exception as ex:
            print(f"LLM entity extraction failed ({ex}); falling back to spaCy")
            result = self._extract_entities(text)
            self._llm_entity_cache[cache_key] = result
            return result

    # ── spaCy-based extraction (used for query-time, or as fallback) ──────────

    def _extract_entities(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract named entities from text using spaCy.
        Used at query time (fast, no API call) and as an ingestion fallback.

        Returns:
            List of (entity_text, entity_type) tuples
        """
        try:
            import re as _re

            max_text_length = 10000
            if len(text) > max_text_length:
                text = text[:max_text_length]

            # Strip domain suffixes before spaCy so "Infoblox.com" → "Infoblox"
            text = _re.sub(
                r'\b(\w{3,})\.(com|net|org|io|co|ae|sa|jo|ca|uk|us|gov|edu)\b',
                r'\1', text, flags=_re.I
            )

            doc = nlp(text)
            entities = []
            _DOMAIN_SUFFIX = _re.compile(
                r'\.(com|net|org|io|co|uk|gov|edu|ca|ae|sa|jo|us)$', _re.I
            )

            try:
                for ent in doc.ents:
                    if len(ent.text) > 1 and ent.text.lower() not in ['the', 'a', 'an']:
                        entity_text = _DOMAIN_SUFFIX.sub('', ent.text.strip()).strip('.').title()
                        if 1 < len(entity_text) <= 100:
                            entities.append((entity_text, ent.label_))
            except Exception as e:
                print(f"Error extracting named entities: {e}")

            try:
                for chunk in doc.noun_chunks:
                    if (len(chunk.text.split()) > 1 and
                            5 < len(chunk.text) < 100 and
                            not any(chunk.text.lower() == ent[0].lower() for ent in entities)):
                        chunk_text = chunk.text.strip().title()
                        if len(chunk_text) <= 100:
                            entities.append((chunk_text, 'CONCEPT'))
            except Exception as e:
                print(f"Error extracting noun chunks: {e}")

            return entities[:100]

        except Exception as e:
            print(f"Error in spaCy entity extraction: {e}")
            return []
    
    def search(self, query: str, doc_id: str = None, limit: int = 10) -> Dict[str, Any]:
        """
        Search the knowledge graph for entities related to the query.
        
        Args:
            query: Search query
            doc_id: Optional document ID to limit scope
            limit: Maximum number of results to return
            
        Returns:
            Dictionary with search results
        """
        # Extract entities from the query
        query_entities = self._extract_entities(query)
        
        if not query_entities:
            # If no entities found, try to match query text against nodes
            query_doc = nlp(query)
            
            # Extract important keywords from query
            keywords = []
            for token in query_doc:
                if token.pos_ in ('NOUN', 'PROPN', 'ADJ') and len(token.text) > 3:
                    keywords.append(token.text.lower())
            
            if not keywords:
                return {'entities': [], 'relationships': []}
            
            # Find nodes that match query keywords
            matching_nodes = []
            for node in self.graph.nodes:
                node_lower = node.lower()
                if any(keyword in node_lower for keyword in keywords):
                    matching_nodes.append(node)
            
            query_entities = [(node, self.graph.nodes[node].get('type', 'UNKNOWN')) 
                              for node in matching_nodes[:5]]
        
        # Initialize results
        entity_results = []
        relationship_results = []
        
        # For each entity in the query, find related entities in the graph
        for entity, _ in query_entities:
            # Check if entity exists in graph
            if entity in self.graph:
                entity_data = {
                    'entity': entity,
                    'type': self.graph.nodes[entity].get('type', 'UNKNOWN'),
                    'count': self.entity_counts.get(entity, 0)
                }
                entity_results.append(entity_data)
                
                # Get related entities (neighbors in the graph)
                neighbors = list(self.graph.neighbors(entity))
                
                # Filter by document if specified
                if doc_id:
                    neighbors = [n for n in neighbors 
                                if doc_id in self.graph.nodes[n].get('doc_ids', set())]
                
                # Sort by edge weight (relationship strength)
                neighbors.sort(key=lambda n: self.graph[entity][n].get('weight', 0), 
                               reverse=True)
                
                # Add top relationships
                for neighbor in neighbors[:limit]:
                    relationship = {
                        'source': entity,
                        'target': neighbor,
                        'weight': self.graph[entity][neighbor].get('weight', 1),
                        'type': self.graph.nodes[neighbor].get('type', 'UNKNOWN')
                    }
                    relationship_results.append(relationship)
        
        return {
            'entities': entity_results,
            'relationships': relationship_results
        }
    
    def get_entity_context(self, entity: str, doc_id: str = None) -> List[int]:
        """
        Get chunk indices where an entity appears.
        
        Args:
            entity: Entity to get context for
            doc_id: Optional document ID to limit scope
            
        Returns:
            List of chunk indices
        """
        if entity not in self.graph:
            return []
        
        chunk_indices = self.graph.nodes[entity].get('chunk_indices', set())
        
        if doc_id:
            # Filter chunk indices by document
            doc_entities = self.document_entities.get(doc_id, set())
            if entity not in doc_entities:
                return []
        
        return list(chunk_indices)
    
    def visualize(self, query: str = None, doc_id: str = None, 
                 output_path: str = 'static/graph.html', limit: int = 30) -> str:
        """
        Create an interactive visualization of the knowledge graph.
        
        Args:
            query: Optional query to filter the graph
            doc_id: Optional document ID to filter the graph
            output_path: Path to save the visualization HTML
            limit: Maximum number of nodes to include
            
        Returns:
            Path to the generated HTML visualization
        """
        # Create a new pyvis network (cdn_resources='remote' avoids broken local lib paths)
        net = Network(height='600px', width='100%', notebook=False, cdn_resources='remote')
        
        # If query is provided, get subgraph based on query
        if query:
            result = self.search(query, doc_id=doc_id, limit=limit)
            entities = [item['entity'] for item in result['entities']]
            
            # Add related entities
            related_entities = set()
            for rel in result['relationships']:
                related_entities.add(rel['source'])
                related_entities.add(rel['target'])
            
            # Combine all entities
            all_entities = set(entities) | related_entities
            
            # Create subgraph
            if all_entities:
                subgraph = self.graph.subgraph(all_entities)
            else:
                # If no entities found, use empty graph
                subgraph = nx.Graph()
        
        # If doc_id is provided without query, filter by document
        elif doc_id:
            doc_entities = self.document_entities.get(doc_id, set())
            # Limit to top entities by count to avoid overwhelming visualization
            sorted_entities = sorted(doc_entities, 
                                    key=lambda e: self.entity_counts.get(e, 0), 
                                    reverse=True)
            top_entities = sorted_entities[:limit]
            subgraph = self.graph.subgraph(top_entities)
        
        # Otherwise use whole graph (limited to top entities)
        else:
            # Sort entities by count and take top ones
            sorted_entities = sorted(self.entity_counts.items(), 
                                    key=lambda x: x[1], 
                                    reverse=True)
            top_entities = [e[0] for e in sorted_entities[:limit]]
            subgraph = self.graph.subgraph(top_entities)
        
        # Convert NetworkX graph to Pyvis
        for node, node_attrs in subgraph.nodes(data=True):
            entity_type = node_attrs.get('type', 'UNKNOWN')
            count = self.entity_counts.get(node, 1)
            
            # Choose color based on entity type
            color_map = {
                'PERSON': '#FF5733',      # Red-orange
                'ORG': '#33A8FF',         # Blue
                'GPE': '#33FF57',         # Green
                'LOC': '#33FF57',         # Green
                'DATE': '#F3FF33',        # Yellow
                'TIME': '#F3FF33',        # Yellow
                'MONEY': '#C133FF',       # Purple
                'PERCENT': '#FF33E9',     # Pink
                'QUANTITY': '#FF9033',    # Orange
                'CONCEPT': '#33FFF6',     # Cyan
                'UNKNOWN': '#CCCCCC',     # Gray
            }
            
            # Set node color and size based on entity type and count
            color = color_map.get(entity_type, '#CCCCCC')
            size = min(30 + (count * 2), 80)  # Scale size with count
            
            # Add node to visualization
            net.add_node(node, label=node, title=f"{node} ({entity_type})", 
                         color=color, size=size)
        
        # Add edges to visualization
        for u, v, edge_attrs in subgraph.edges(data=True):
            weight = edge_attrs.get('weight', 1)
            # Scale width with weight
            width = min(1 + (weight * 0.5), 8)
            net.add_edge(u, v, value=weight, title=f"Weight: {weight}", width=width)
        
        # Set physics and interaction options
        net.set_options("""
        {
          "physics": {
            "forceAtlas2Based": {
              "gravitationalConstant": -50,
              "centralGravity": 0.01,
              "springLength": 100,
              "springConstant": 0.08
            },
            "maxVelocity": 50,
            "solver": "forceAtlas2Based",
            "timestep": 0.35,
            "stabilization": {
              "enabled": true,
              "iterations": 300
            }
          },
          "interaction": {
            "hover": true,
            "navigationButtons": true,
            "keyboard": {
              "enabled": true
            }
          }
        }
        """)
        
        # Create directory if it doesn't exist
        dir_name = os.path.dirname(output_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        
        # Save visualization
        net.save_graph(output_path)
        
        return output_path
    
    def _get_top_entities(self, doc_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top entities for a document by occurrence count."""
        doc_entities = self.document_entities.get(doc_id, set())
        
        # Sort entities by count
        sorted_entities = sorted(doc_entities, 
                                key=lambda e: self.entity_counts.get(e, 0), 
                                reverse=True)
        
        top_entities = []
        for entity in sorted_entities[:limit]:
            if entity in self.graph:
                entity_type = self.graph.nodes[entity].get('type', 'UNKNOWN')
                top_entities.append({
                    'entity': entity,
                    'count': self.entity_counts.get(entity, 0),
                    'type': entity_type
                })
        
        return top_entities
    
    def _remove_document_entities(self, doc_id: str):
        """Remove entities and relationships associated with a document."""
        # Get entities for this document
        doc_entities = self.document_entities.get(doc_id, set())
        
        for entity in doc_entities:
            # Update document lists in nodes
            if entity in self.graph:
                node_data = self.graph.nodes[entity]
                if 'doc_ids' in node_data:
                    node_data['doc_ids'].discard(doc_id)
                
                # Remove node if it's no longer associated with any document
                if not node_data.get('doc_ids'):
                    self.graph.remove_node(entity)
                    # Clean up entity metadata
                    self.entity_counts.pop(entity, None)
                    self.entity_contexts.pop(entity, None)
        
        # Clear document entity list
        self.document_entities.pop(doc_id, None)
    
    def _save_graph(self):
        """Save the graph to disk."""
        graph_path = os.path.join(self.db_path, 'graph.json')
        
        # Convert graph to serializable format
        graph_data = {
            'nodes': [],
            'edges': []
        }
        
        # Add nodes
        for node, attrs in self.graph.nodes(data=True):
            node_data = {'id': node}
            
            # Convert sets to lists for serialization
            for key, value in attrs.items():
                if isinstance(value, set):
                    node_data[key] = list(value)
                else:
                    node_data[key] = value
            
            graph_data['nodes'].append(node_data)
        
        # Add edges
        for u, v, attrs in self.graph.edges(data=True):
            edge_data = {'source': u, 'target': v}
            
            # Convert sets to lists for serialization
            for key, value in attrs.items():
                if isinstance(value, set):
                    edge_data[key] = list(value)
                else:
                    edge_data[key] = value
            
            graph_data['edges'].append(edge_data)
        
        # Save graph data
        with open(graph_path, 'w') as f:
            json.dump(graph_data, f)
        
        # Save entity metadata
        entity_data = {
            'entity_counts': self.entity_counts,
            'entity_contexts': {k: list(v) for k, v in self.entity_contexts.items()},
            'document_entities': {k: list(v) for k, v in self.document_entities.items()}
        }
        
        entity_path = os.path.join(self.db_path, 'entities.json')
        with open(entity_path, 'w') as f:
            json.dump(entity_data, f)
    
    def _load_graph(self):
        """Load the graph from disk."""
        graph_path = os.path.join(self.db_path, 'graph.json')
        entity_path = os.path.join(self.db_path, 'entities.json')
        
        if os.path.exists(graph_path) and os.path.exists(entity_path):
            try:
                # Load graph data
                with open(graph_path, 'r') as f:
                    graph_data = json.load(f)
                
                # Create new graph
                graph = nx.Graph()
                
                # Add nodes
                for node_data in graph_data['nodes']:
                    node_id = node_data.pop('id')
                    
                    # Convert lists back to sets
                    for key, value in node_data.items():
                        if isinstance(value, list):
                            node_data[key] = set(value)
                    
                    graph.add_node(node_id, **node_data)
                
                # Add edges
                for edge_data in graph_data['edges']:
                    source = edge_data.pop('source')
                    target = edge_data.pop('target')
                    
                    # Convert lists back to sets
                    for key, value in edge_data.items():
                        if isinstance(value, list):
                            edge_data[key] = set(value)
                    
                    graph.add_edge(source, target, **edge_data)
                
                # Replace current graph
                self.graph = graph
                
                # Load entity metadata
                with open(entity_path, 'r') as f:
                    entity_data = json.load(f)
                
                self.entity_counts = defaultdict(int, entity_data.get('entity_counts', {}))
                
                # Convert lists back to sets
                self.entity_contexts = defaultdict(set)
                for k, v in entity_data.get('entity_contexts', {}).items():
                    self.entity_contexts[k] = set(v)
                
                self.document_entities = defaultdict(set)
                for k, v in entity_data.get('document_entities', {}).items():
                    self.document_entities[k] = set(v)
                
                print(f"Loaded knowledge graph with {len(self.graph.nodes)} entities and {len(self.graph.edges)} relationships")
                
            except Exception as e:
                print(f"Error loading knowledge graph: {str(e)}")
                # Start with empty graph if loading fails
                self.graph = nx.Graph()
                self.entity_counts = defaultdict(int)
                self.entity_contexts = defaultdict(set)
                self.document_entities = defaultdict(set)
    
    def get_all_entities(self) -> List[Dict[str, Any]]:
        """Get all entities in the graph sorted by count."""
        entities = []
        for entity, count in sorted(self.entity_counts.items(), key=lambda x: x[1], reverse=True):
            if entity in self.graph:
                entity_type = self.graph.nodes[entity].get('type', 'UNKNOWN')
                entities.append({
                    'entity': entity,
                    'count': count,
                    'type': entity_type
                })
        return entities
    
    def get_entity_relationships(self, entity: str) -> List[Dict[str, Any]]:
        """Get all relationships for an entity."""
        if entity not in self.graph:
            return []
        
        relationships = []
        for neighbor in self.graph.neighbors(entity):
            weight = self.graph[entity][neighbor].get('weight', 1)
            neighbor_type = self.graph.nodes[neighbor].get('type', 'UNKNOWN')
            
            relationships.append({
                'source': entity,
                'target': neighbor,
                'weight': weight,
                'type': neighbor_type
            })
        
        # Sort by weight
        relationships.sort(key=lambda x: x['weight'], reverse=True)
        
        return relationships

    def enhance_query_with_entities(self, query: str) -> str:
        """Enhance a query by adding relevant entities from the knowledge graph."""
        # Extract entities from the query
        query_entities = self._extract_entities(query)
        
        if not query_entities:
            return query
        
        expanded_terms = []
        
        # For each entity in the query, find related entities
        for entity, _ in query_entities:
            if entity in self.graph:
                # Get top neighbors by edge weight
                neighbors = list(self.graph.neighbors(entity))
                if neighbors:
                    # Sort by edge weight
                    neighbors.sort(key=lambda n: self.graph[entity][n].get('weight', 0), 
                                  reverse=True)
                    # Add top related entities to expansion terms
                    expanded_terms.extend(neighbors[:3])
        
        # If we found related terms, add them to the query
        if expanded_terms:
            enhanced_query = f"{query} OR " + " OR ".join(expanded_terms)
            return enhanced_query
        
        return query

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph."""
        entity_types = defaultdict(int)
        for node, attrs in self.graph.nodes(data=True):
            entity_type = attrs.get('type', 'UNKNOWN')
            entity_types[entity_type] += 1
        document_coverage = len(self.document_entities)
        sorted_types = sorted(entity_types.items(), key=lambda x: x[1], reverse=True)
        return {
            'node_count': len(self.graph.nodes),
            'edge_count': len(self.graph.edges),
            'document_count': document_coverage,
            'entity_types': dict(sorted_types)
        }

    # ------------------------------------------------------------------
    # Multi-hop graph traversal
    # ------------------------------------------------------------------

    def multi_hop_search(
        self,
        query: str,
        hops: int = 2,
        doc_id: str = None,
        max_neighbors_per_node: int = 6,
        limit_chunk_indices: int = 60,
    ) -> Dict[str, Any]:
        """
        Perform a breadth-first multi-hop traversal starting from entities
        mentioned in *query*, collecting chunk indices at each hop.

        Args:
            query:                  User query text.
            hops:                   Maximum number of graph hops to follow.
            doc_id:                 When given, restrict entity/chunk lookup to
                                    this document.
            max_neighbors_per_node: How many neighbors to follow per node at
                                    each hop (sorted by edge weight).
            limit_chunk_indices:    Cap on total chunk indices returned (keeps
                                    downstream retrieval tractable).

        Returns::

            {
                'query_entities': ['Amazon', 'AWS', ...],
                'all_entities':   [{'entity': str, 'type': str, 'hop': int}, ...],
                'chunk_indices':  [3, 7, 12, ...],     # chunk_index values
                'hop_summary':    [{'hop': 1, 'entities': [...]}, ...],
            }
        """
        # ── 1. seed entities ───────────────────────────────────────────
        raw_entities = self._extract_entities(query)
        seed_names: Set[str] = set()

        for entity_text, _ in raw_entities:
            title = entity_text.title()
            if title in self.graph:
                seed_names.add(title)
            else:
                # fuzzy fall-back: substring match on graph nodes
                q_lower = entity_text.lower()
                for node in self.graph.nodes:
                    if q_lower in node.lower() or node.lower() in q_lower:
                        seed_names.add(node)
                        break

        # If spaCy found nothing, fall back to keyword matching on nodes
        if not seed_names:
            keywords = [
                tok.text.lower()
                for tok in nlp(query)
                if tok.pos_ in ('NOUN', 'PROPN') and len(tok.text) > 3
            ]
            for node in self.graph.nodes:
                if any(kw in node.lower() for kw in keywords):
                    seed_names.add(node)
                if len(seed_names) >= 10:
                    break

        # ── 2. BFS traversal ──────────────────────────────────────────
        visited: Set[str] = set()
        frontier: Set[str] = seed_names.copy()
        all_chunk_indices: List[int] = []
        hop_summary: List[Dict[str, Any]] = []
        all_entities_info: List[Dict[str, Any]] = []

        for hop in range(1, hops + 1):
            if not frontier:
                break

            next_frontier: Set[str] = set()
            hop_entities: List[Dict[str, Any]] = []

            for entity in frontier:
                if entity in visited or entity not in self.graph:
                    continue
                visited.add(entity)

                # Collect chunk indices
                node_data = self.graph.nodes[entity]
                raw_indices: Set[int] = node_data.get('chunk_indices', set())

                if doc_id:
                    # Only include if this entity belongs to the document
                    entity_docs: Set[str] = node_data.get('doc_ids', set())
                    if doc_id not in entity_docs:
                        continue
                    # chunk_indices stored per-entity are global across docs,
                    # so we accept them all (they were set when doc was added)

                hop_entities.append({
                    'entity': entity,
                    'type': node_data.get('type', 'UNKNOWN'),
                    'hop': hop,
                    'chunk_indices': list(raw_indices),
                })
                all_entities_info.append({
                    'entity': entity,
                    'type': node_data.get('type', 'UNKNOWN'),
                    'hop': hop,
                })
                all_chunk_indices.extend(raw_indices)

                # Queue neighbors for next hop, sorted by edge weight desc
                neighbors = sorted(
                    self.graph.neighbors(entity),
                    key=lambda n: self.graph[entity][n].get('weight', 0),
                    reverse=True,
                )
                next_frontier.update(neighbors[:max_neighbors_per_node])

            hop_summary.append({'hop': hop, 'entities': hop_entities})
            frontier = next_frontier - visited

        # Deduplicate & cap chunk indices
        seen: Set[int] = set()
        deduped: List[int] = []
        for idx in all_chunk_indices:
            if idx not in seen:
                seen.add(idx)
                deduped.append(idx)
            if len(deduped) >= limit_chunk_indices:
                break

        return {
            'query_entities': list(seed_names),
            'all_entities': all_entities_info,
            'chunk_indices': deduped,
            'hop_summary': hop_summary,
        }

    # ------------------------------------------------------------------
    # PostgreSQL graph persistence
    # ------------------------------------------------------------------

    def sync_to_db(self, doc_id: str = None) -> Dict[str, Any]:
        """
        Upsert the in-memory NetworkX graph into the PostgreSQL
        ``graph_entity`` and ``graph_relationship`` tables.

        When *doc_id* is provided only the entities that belong to that
        document are synced (faster incremental updates).  Pass ``None``
        to do a full sync (useful for initial migration).

        Returns counts of entities/relationships upserted.
        """
        try:
            from app import db
            from models import GraphEntity, GraphRelationship

            nodes_to_sync = (
                [n for n in self.graph.nodes
                 if doc_id in self.graph.nodes[n].get('doc_ids', set())]
                if doc_id else list(self.graph.nodes)
            )

            # ── If re-syncing a specific doc, purge its stale entities first ──
            # This prevents old chunk_indices from polluting future traversals.
            if doc_id:
                from sqlalchemy import text as _sa_text
                stale_ids_rows = db.session.execute(
                    _sa_text(
                        "SELECT id FROM graph_entity "
                        "WHERE doc_ids::text LIKE :pat"
                    ),
                    {"pat": f"%{doc_id}%"},
                ).fetchall()
                stale_ids = [r[0] for r in stale_ids_rows]
                if stale_ids:
                    db.session.execute(
                        _sa_text(
                            "DELETE FROM graph_relationship "
                            "WHERE source_id = ANY(:ids) OR target_id = ANY(:ids)"
                        ),
                        {"ids": stale_ids},
                    )
                    db.session.execute(
                        _sa_text("DELETE FROM graph_entity WHERE id = ANY(:ids)"),
                        {"ids": stale_ids},
                    )
                    db.session.flush()

            upserted_entities = 0
            entity_id_map: Dict[str, int] = {}  # name → db id

            for name in nodes_to_sync:
                node_data = self.graph.nodes[name]
                entity_type = node_data.get('type', 'UNKNOWN')
                doc_ids_list = list(node_data.get('doc_ids', set()))
                chunk_idx_list = sorted(node_data.get('chunk_indices', set()))
                occ = self.entity_counts.get(name, 1)

                # After purge, existing rows for this doc are gone — always insert
                existing = GraphEntity.query.filter_by(name=name).first()
                if existing:
                    # Entity belongs to another doc too — merge without duplicates
                    merged_docs = list(set((existing.doc_ids or []) + doc_ids_list))
                    # Keep per-doc chunk ranges separate to avoid stale cross-doc indices
                    existing_chunks = existing.chunk_indices or []
                    merged_chunks = sorted(set(existing_chunks) | set(chunk_idx_list))
                    existing.doc_ids = merged_docs
                    existing.chunk_indices = merged_chunks
                    existing.occurrence_count = max(existing.occurrence_count, occ)
                    entity_id_map[name] = existing.id
                else:
                    ent = GraphEntity(
                        name=name,
                        entity_type=entity_type,
                        doc_ids=doc_ids_list,
                        chunk_indices=chunk_idx_list,
                        occurrence_count=occ,
                    )
                    db.session.add(ent)
                    db.session.flush()  # get id
                    entity_id_map[name] = ent.id

                upserted_entities += 1

            db.session.flush()

            # Re-fetch any entities we didn't create (pre-existing)
            missing = [n for n in nodes_to_sync if n not in entity_id_map]
            if missing:
                rows = GraphEntity.query.filter(GraphEntity.name.in_(missing)).all()
                for r in rows:
                    entity_id_map[r.name] = r.id

            # ── Relationships ─────────────────────────────────────────
            upserted_rels = 0
            for u, v, edge_data in self.graph.edges(data=True):
                if u not in entity_id_map or v not in entity_id_map:
                    continue
                src_id = entity_id_map[u]
                tgt_id = entity_id_map[v]
                weight = edge_data.get('weight', 1)
                doc_ids_list = list(edge_data.get('doc_ids', set()))
                chunk_idx_list = list(edge_data.get('chunks', set()))

                existing_rel = GraphRelationship.query.filter_by(
                    source_id=src_id, target_id=tgt_id
                ).first()
                if existing_rel:
                    existing_rel.weight = existing_rel.weight + weight
                    merged_d = list(set((existing_rel.doc_ids or []) + doc_ids_list))
                    merged_c = list(set((existing_rel.chunk_indices or []) + chunk_idx_list))
                    existing_rel.doc_ids = merged_d
                    existing_rel.chunk_indices = merged_c
                else:
                    rel = GraphRelationship(
                        source_id=src_id,
                        target_id=tgt_id,
                        weight=weight,
                        doc_ids=doc_ids_list,
                        chunk_indices=chunk_idx_list,
                    )
                    db.session.add(rel)
                upserted_rels += 1

            db.session.commit()
            print(f"sync_to_db: upserted {upserted_entities} entities, "
                  f"{upserted_rels} relationships for doc_id={doc_id}")
            return {
                'entities_upserted': upserted_entities,
                'relationships_upserted': upserted_rels,
            }

        except Exception as exc:
            import traceback
            print(f"sync_to_db error: {exc}\n{traceback.format_exc()}")
            try:
                from app import db
                db.session.rollback()
            except Exception:
                pass
            return {'entities_upserted': 0, 'relationships_upserted': 0}

    def multi_hop_search_db(
        self,
        query: str,
        hops: int = 2,
        doc_id: str = None,
        max_neighbors: int = 6,
        limit_chunk_indices: int = 60,
    ) -> Dict[str, Any]:
        """
        SQL-backed multi-hop traversal using the PostgreSQL graph tables.
        Falls back to the NetworkX version if the DB tables are empty.

        This is the preferred path when the tables are populated because
        it avoids loading the full NetworkX graph into memory for each query
        and enables proper SQL filtering.
        """
        try:
            from app import db
            from models import GraphEntity, GraphRelationship

            # Check if DB has graph data
            if GraphEntity.query.count() == 0:
                return self.multi_hop_search(query, hops, doc_id,
                                              max_neighbors, limit_chunk_indices)

            # ── Common abbreviation → expansion map ───────────────────────────────
            _ABBREV = {
                'mba': ['master', 'business', 'administration'],
                'ms': ['master', 'science'],
                'bs': ['bachelor', 'science'],
                'phd': ['doctor', 'philosophy'],
                'aws': ['amazon', 'web', 'services'],
                'gcp': ['google', 'cloud'],
                'ai': ['artificial', 'intelligence'],
                'ml': ['machine', 'learning'],
                'nlp': ['natural', 'language', 'processing'],
                'vod': ['video', 'demand'],
                'rag': ['retrieval', 'augmented'],
                'llm': ['language', 'model'],
                'ddi': ['dns', 'dhcp', 'ipam'],
                'saas': ['software', 'service'],
                'ci': ['continuous', 'integration'],
                'cd': ['continuous', 'delivery'],
                'k8s': ['kubernetes'],
                'pmp': ['project', 'management', 'professional'],
            }

            # Extract seed entity names from query
            raw_entities = self._extract_entities(query)
            exact_names = [pair[0].title() for pair in raw_entities]

            # Also pull significant nouns/proper-nouns from the query as keywords
            kw_tokens = [
                tok.text.lower()
                for tok in nlp(query)
                if tok.pos_ in ('NOUN', 'PROPN') and len(tok.text) > 2
                and tok.text.lower() not in ('did', 'does', 'has', 'was', 'were',
                                              'have', 'been', 'being', 'where',
                                              'when', 'what', 'who', 'how')
            ]

            # Expand abbreviations — e.g. "MBA" → ["master","business","administration"]
            expanded: List[str] = []
            for tok in nlp(query):
                abbrev = _ABBREV.get(tok.text.lower())
                if abbrev:
                    expanded.extend(abbrev)
            kw_tokens = list(set(kw_tokens + expanded))

            from sqlalchemy import or_

            # Resolve seed entity IDs via ILIKE fuzzy matching on ALL candidate names
            candidates = list(set(exact_names + kw_tokens))
            seed_entities_db: List = []
            if candidates:
                ilike_filter = GraphEntity.name.ilike(f'%{candidates[0]}%')
                for cand in candidates[1:]:
                    ilike_filter = or_(ilike_filter, GraphEntity.name.ilike(f'%{cand}%'))
                seed_entities_db = GraphEntity.query.filter(ilike_filter).limit(10).all()

            if not seed_entities_db:
                return {'query_entities': [], 'all_entities': [],
                        'chunk_indices': [], 'hop_summary': []}

            seed_names = [e.name for e in seed_entities_db]

            # BFS using DB
            visited_ids: Set[int] = set()
            frontier_names = set(seed_names)
            all_chunk_indices: List[int] = []
            all_entities_info: List[Dict[str, Any]] = []
            hop_summary: List[Dict[str, Any]] = []

            for hop in range(1, hops + 1):
                if not frontier_names:
                    break

                # Fetch entities matching frontier names
                entities = GraphEntity.query.filter(
                    GraphEntity.name.in_(list(frontier_names))
                ).all()

                hop_entities = []
                next_frontier_ids: Set[int] = set()

                for ent in entities:
                    if ent.id in visited_ids:
                        continue
                    if doc_id and doc_id not in (ent.doc_ids or []):
                        continue
                    visited_ids.add(ent.id)

                    indices = ent.chunk_indices or []
                    all_chunk_indices.extend(indices)
                    hop_entities.append({
                        'entity': ent.name,
                        'type': ent.entity_type,
                        'hop': hop,
                        'chunk_indices': indices,
                    })
                    all_entities_info.append({
                        'entity': ent.name,
                        'type': ent.entity_type,
                        'hop': hop,
                    })

                    # Collect neighbor IDs via outgoing + incoming rels
                    outgoing = (
                        GraphRelationship.query
                        .filter_by(source_id=ent.id)
                        .order_by(GraphRelationship.weight.desc())
                        .limit(max_neighbors)
                        .all()
                    )
                    incoming = (
                        GraphRelationship.query
                        .filter_by(target_id=ent.id)
                        .order_by(GraphRelationship.weight.desc())
                        .limit(max_neighbors)
                        .all()
                    )
                    neighbor_ids = (
                        {r.target_id for r in outgoing} |
                        {r.source_id for r in incoming}
                    ) - visited_ids
                    next_frontier_ids.update(neighbor_ids)

                hop_summary.append({'hop': hop, 'entities': hop_entities})

                # Fetch next frontier entity names
                if next_frontier_ids:
                    next_ents = GraphEntity.query.filter(
                        GraphEntity.id.in_(list(next_frontier_ids))
                    ).all()
                    frontier_names = {e.name for e in next_ents}
                else:
                    frontier_names = set()

            # Deduplicate & cap
            seen: Set[int] = set()
            deduped: List[int] = []
            for idx in all_chunk_indices:
                if idx not in seen:
                    seen.add(idx)
                    deduped.append(idx)
                if len(deduped) >= limit_chunk_indices:
                    break

            return {
                'query_entities': seed_names,
                'all_entities': all_entities_info,
                'chunk_indices': deduped,
                'hop_summary': hop_summary,
            }

        except Exception as exc:
            import traceback
            print(f"multi_hop_search_db error: {exc}\n{traceback.format_exc()}")
            return self.multi_hop_search(query, hops, doc_id,
                                          max_neighbors, limit_chunk_indices)