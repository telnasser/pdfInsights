import os
import logging
import datetime
import json
import numpy as np
from flask import Flask, render_template, jsonify
from flask_bootstrap import Bootstrap
from models import db

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key")
app.json_encoder = NumpyEncoder  # Use the custom JSON encoder

# Configure database
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get("DATABASE_URL")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_recycle': 300,
    'pool_pre_ping': True,
}

# Initialize extensions
bootstrap = Bootstrap(app)
db.init_app(app)

# Create database tables if they don't exist
with app.app_context():
    db.create_all()

def _rebuild_index_from_db():
    """
    Called once at startup (per worker).  If the persistent TF-IDF vectorizer
    is missing on disk but DB chunks exist, re-generate embeddings from scratch
    so that FAISS and the vectorizer are always in sync.
    """
    import os, logging, pickle
    from rag.embeddings import _VECTORIZER_PATH, _PROJECTION_PATH

    log = logging.getLogger(__name__)

    vectorizer_missing = (
        not os.path.exists(_VECTORIZER_PATH) or
        not os.path.exists(_PROJECTION_PATH)
    )
    if not vectorizer_missing:
        log.info("TF-IDF vectorizer found on disk — skipping re-index")
        return

    with app.app_context():
        from sqlalchemy import text as sa_text
        rows = db.session.execute(sa_text(
            "SELECT id, document_id, chunk_index, text, page_num FROM chunk"
        )).fetchall()

        if not rows:
            log.info("No chunks in DB — nothing to re-index")
            return

        log.info("Re-building TF-IDF vectorizer + FAISS index for %d chunks ...", len(rows))

        try:
            from rag.embeddings import EmbeddingGenerator
            from rag.vector_store import VectorStore

            emb_gen = EmbeddingGenerator()
            texts = [r[3] for r in rows]

            # Force-refit on all chunk texts → saves vectorizer + projection to disk
            emb_gen._refit_and_save(texts)

            # Generate embeddings using the newly saved vectorizer
            embeddings = emb_gen._generate_local_embeddings(texts)

            # Rebuild FAISS index: delete old index first
            import shutil
            index_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vector_db', 'faiss_index')
            if os.path.exists(index_dir):
                shutil.rmtree(index_dir)
            os.makedirs(index_dir, exist_ok=True)

            vector_store = VectorStore()

            # Group by document_id
            from collections import defaultdict
            doc_chunks: dict = defaultdict(list)
            for i, row in enumerate(rows):
                doc_chunks[row[1]].append({
                    'chunk_index': row[2],
                    'text': row[3],
                    'page_num': row[4],
                    'document_id': row[1],
                    'embedding': embeddings[i],
                })

            for doc_id_key, chunks in doc_chunks.items():
                vector_store.add_document(doc_id_key, chunks)
                log.info("Re-indexed %d chunks for doc %s", len(chunks), doc_id_key)

            log.info("Re-index complete — %d total chunks indexed", len(rows))
        except Exception as exc:
            log.error("Re-index failed: %s", exc, exc_info=True)

_rebuild_index_from_db()


def _rebuild_graph_from_db():
    """
    Called once at startup (per worker).  If the knowledge-graph JSON files are
    absent but DB graph tables are also empty (e.g. first boot or after cleanup),
    re-build the in-memory + DB graph from the chunk table so queries always
    have graph data available.
    """
    import logging
    log = logging.getLogger(__name__)

    kg_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'knowledge_graph')
    graph_json = os.path.join(kg_dir, 'graph.json')

    with app.app_context():
        from models import GraphEntity, Chunk, Document
        db_empty = GraphEntity.query.count() == 0
        json_missing = not os.path.exists(graph_json)

        if not db_empty:
            log.info("Graph tables populated (%d entities) — skipping rebuild",
                     GraphEntity.query.count())
            return

        if db_empty and not json_missing:
            # JSON exists but DB is empty — sync from in-memory graph on first query
            log.info("Graph JSON exists but DB empty — will sync on first query")
            return

        # Both DB and JSON are missing — rebuild from chunks
        chunk_count = Chunk.query.count()
        if chunk_count == 0:
            log.info("No chunks in DB — skipping graph rebuild")
            return

        log.info("Rebuilding knowledge graph from %d DB chunks …", chunk_count)
        try:
            from rag.knowledge_graph import KnowledgeGraph
            kg = KnowledgeGraph()

            docs = Document.query.all()
            for doc in docs:
                chunks_db = Chunk.query.filter_by(document_id=doc.id).order_by(Chunk.chunk_index).all()
                chunks_list = [{'chunk_index': c.chunk_index, 'text': c.text} for c in chunks_db]
                stats = kg.add_document(doc.id, chunks_list)
                log.info("Graph built for doc %s: %d entities, %d rels",
                         doc.id[:12], stats['entity_count'], stats['relationship_count'])
                kg.sync_to_db(doc_id=doc.id)

            log.info("Graph rebuild complete")
        except Exception as exc:
            log.error("Graph rebuild failed: %s", exc, exc_info=True)


_rebuild_graph_from_db()

# Add datetime to all templates
@app.context_processor
def inject_now():
    return {'now': datetime.datetime.now()}

# Configure upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 35 * 1024 * 1024  # 35MB max upload size

# Import routes after app is created to avoid circular imports
from routes import document_routes, query_routes

# Register blueprints
app.register_blueprint(document_routes.bp)
app.register_blueprint(query_routes.bp)

# Define error handlers
@app.errorhandler(413)
def too_large(e):
    """Handle request entity too large error."""
    return render_template('error.html', error="File too large, maximum size is 35MB"), 413

@app.errorhandler(404)
def not_found(e):
    """Handle page not found error."""
    return render_template('error.html', error="Page not found"), 404

@app.errorhandler(500)
def server_error(e):
    """Handle internal server error."""
    app.logger.error(f"500 error: {str(e)}")
    
    # Get more detailed error information
    import traceback
    error_traceback = traceback.format_exc()
    app.logger.error(f"Traceback: {error_traceback}")
    
    # In development mode, show the traceback to assist debugging
    if app.debug:
        error_message = f"Internal server error: {str(e)}<br><pre>{error_traceback}</pre>"
    else:
        error_message = "An unexpected error occurred. Please try again later."
    
    return render_template('error.html', error=error_message), 500

# Root route
@app.route('/')
def index():
    """Render the home page."""
    return render_template('index.html')

# If this file is run directly, start the development server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)