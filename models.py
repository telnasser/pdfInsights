import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Document(db.Model):
    """Document model for storing metadata about uploaded PDFs."""
    id = db.Column(db.String(64), primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    title = db.Column(db.String(255), nullable=True)
    upload_date = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    num_pages = db.Column(db.Integer, nullable=True)
    num_chunks = db.Column(db.Integer, nullable=True)
    file_size = db.Column(db.Integer, nullable=True)  # Size in bytes
    chunk_strategy = db.Column(db.String(50), nullable=True)
    chunk_size = db.Column(db.Integer, nullable=True)
    chunk_overlap = db.Column(db.Integer, nullable=True)
    
    # Knowledge graph related fields
    kg_processed = db.Column(db.Boolean, default=False)
    kg_entity_count = db.Column(db.Integer, nullable=True)
    kg_relationship_count = db.Column(db.Integer, nullable=True)
    kg_processing_time = db.Column(db.Float, nullable=True)  # Time in seconds
    
    # Relationships
    chunks = db.relationship('Chunk', backref='document', lazy=True, cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Document {self.filename}>"
    
    def to_dict(self):
        """Convert document to dictionary."""
        return {
            'id': self.id,
            'filename': self.filename,
            'title': self.title,
            'upload_date': self.upload_date.isoformat() if self.upload_date else None,
            'num_pages': self.num_pages,
            'num_chunks': self.num_chunks,
            'file_size': self.file_size,
            'chunk_strategy': self.chunk_strategy,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'kg_processed': self.kg_processed,
            'kg_entity_count': self.kg_entity_count,
            'kg_relationship_count': self.kg_relationship_count,
            'kg_processing_time': self.kg_processing_time
        }


class Chunk(db.Model):
    """Chunk model for storing document chunks."""
    id = db.Column(db.Integer, primary_key=True)
    document_id = db.Column(db.String(64), db.ForeignKey('document.id'), nullable=False)
    chunk_index = db.Column(db.Integer, nullable=False)
    text = db.Column(db.Text, nullable=False)
    page_num = db.Column(db.Integer, nullable=True)
    chunk_metadata = db.Column(db.JSON, nullable=True)
    
    def __repr__(self):
        return f"<Chunk {self.id} of Document {self.document_id}>"
    
    def to_dict(self, include_text=True):
        """Convert chunk to dictionary."""
        result = {
            'id': self.id,
            'document_id': self.document_id,
            'chunk_index': self.chunk_index,
            'page_num': self.page_num
        }
        
        if include_text:
            result['text'] = self.text
            
        if self.chunk_metadata:
            result.update(self.chunk_metadata)
            
        return result


class Query(db.Model):
    """Query model for storing user queries and responses."""
    id = db.Column(db.Integer, primary_key=True)
    query_text = db.Column(db.Text, nullable=False)
    response_text = db.Column(db.Text, nullable=True)
    document_id = db.Column(db.String(64), db.ForeignKey('document.id'), nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    top_k = db.Column(db.Integer, nullable=True)
    temperature = db.Column(db.Float, nullable=True)
    
    # Relationships
    chunks = db.relationship('QueryChunk', backref='query', lazy=True, cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Query {self.id}: {self.query_text[:30]}...>"
    
    def to_dict(self):
        """Convert query to dictionary."""
        return {
            'id': self.id,
            'query_text': self.query_text,
            'response_text': self.response_text,
            'document_id': self.document_id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'top_k': self.top_k,
            'temperature': self.temperature
        }


class QueryChunk(db.Model):
    """Association table for queries and their relevant chunks."""
    id = db.Column(db.Integer, primary_key=True)
    query_id = db.Column(db.Integer, db.ForeignKey('query.id'), nullable=False)
    chunk_id = db.Column(db.Integer, db.ForeignKey('chunk.id'), nullable=False)
    relevance_score = db.Column(db.Float, nullable=True)
    
    # Relationships
    chunk = db.relationship('Chunk', backref='queries')
    
    def __repr__(self):
        return f"<QueryChunk {self.id}: Query {self.query_id}, Chunk {self.chunk_id}>"


class GraphEntity(db.Model):
    """
    Persists a knowledge-graph entity (node) in PostgreSQL.
    Replaces the JSON-file approach so the graph survives restarts and
    can be queried with SQL for multi-hop traversal.
    """
    __tablename__ = 'graph_entity'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False, unique=True, index=True)
    entity_type = db.Column(db.String(50), nullable=True)
    doc_ids = db.Column(db.JSON, default=list)
    chunk_indices = db.Column(db.JSON, default=list)
    occurrence_count = db.Column(db.Integer, default=1)

    outgoing_rels = db.relationship(
        'GraphRelationship',
        foreign_keys='GraphRelationship.source_id',
        backref='source_entity',
        lazy='dynamic',
        cascade='all, delete-orphan'
    )
    incoming_rels = db.relationship(
        'GraphRelationship',
        foreign_keys='GraphRelationship.target_id',
        backref='target_entity',
        lazy='dynamic',
        cascade='all, delete-orphan'
    )

    def __repr__(self):
        return f"<GraphEntity {self.name} ({self.entity_type})>"

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'entity_type': self.entity_type,
            'doc_ids': self.doc_ids or [],
            'chunk_indices': self.chunk_indices or [],
            'occurrence_count': self.occurrence_count,
        }


class GraphRelationship(db.Model):
    """
    Persists a knowledge-graph relationship (edge) in PostgreSQL.
    """
    __tablename__ = 'graph_relationship'

    id = db.Column(db.Integer, primary_key=True)
    source_id = db.Column(
        db.Integer, db.ForeignKey('graph_entity.id', ondelete='CASCADE'),
        nullable=False, index=True
    )
    target_id = db.Column(
        db.Integer, db.ForeignKey('graph_entity.id', ondelete='CASCADE'),
        nullable=False, index=True
    )
    weight = db.Column(db.Integer, default=1)
    doc_ids = db.Column(db.JSON, default=list)
    chunk_indices = db.Column(db.JSON, default=list)

    __table_args__ = (
        db.UniqueConstraint('source_id', 'target_id', name='uq_graph_relationship'),
    )

    def __repr__(self):
        return f"<GraphRelationship {self.source_id} -> {self.target_id} (w={self.weight})>"