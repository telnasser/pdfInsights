import os
import json
import time
from uuid import uuid4
from werkzeug.utils import secure_filename
from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, send_from_directory

from config import ALLOWED_EXTENSIONS, UPLOAD_FOLDER, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_STRATEGY, KNOWLEDGE_GRAPH_PATH
from rag.document_processor import DocumentProcessor
from rag.embeddings import EmbeddingGenerator
from rag.vector_store import VectorStore
from rag.visualization import Visualizer
from rag.knowledge_graph import KnowledgeGraph
from models import db, Document, Chunk

# Initialize the blueprint
bp = Blueprint('document', __name__, url_prefix='/document')

# Initialize components
document_processor = DocumentProcessor(UPLOAD_FOLDER)
embedding_generator = EmbeddingGenerator()
vector_store = VectorStore()
visualizer = Visualizer()
knowledge_graph = KnowledgeGraph(db_path=KNOWLEDGE_GRAPH_PATH)

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@bp.route('/upload', methods=['GET', 'POST'])
def upload():
    """Handle document upload."""
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
            
        file = request.files['file']
        
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)
            
        # Get chunking parameters
        chunk_size = int(request.form.get('chunk_size', DEFAULT_CHUNK_SIZE))
        chunk_overlap = int(request.form.get('chunk_overlap', DEFAULT_CHUNK_OVERLAP))
        chunk_strategy = request.form.get('chunk_strategy', DEFAULT_CHUNK_STRATEGY)
        
        if file and allowed_file(file.filename):
            # Generate a UUID for the file
            file_uuid = str(uuid4())
            filename = secure_filename(file.filename)
            
            # Add UUID as prefix to filename
            filename = f"{file_uuid}_{filename}"
            
            # Save the file
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            
            try:
                # Process the document
                processed_doc = document_processor.process_pdf(
                    filename,
                    chunk_strategy=chunk_strategy,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                
                # Check if we have chunks
                if not processed_doc['chunks'] or len(processed_doc['chunks']) == 0:
                    raise ValueError(f"No text chunks were extracted from the document: {filename}")
                
                # Log chunking information
                print(f"Processing document: {filename}")
                print(f"Document ID: {processed_doc['id']}")
                print(f"Extracted {len(processed_doc['chunks'])} chunks using {chunk_strategy} strategy")
                
                # Generate embeddings for chunks
                print(f"Generating embeddings for {len(processed_doc['chunks'])} chunks")
                chunks_with_embeddings = embedding_generator.embed_chunks(processed_doc['chunks'])
                print(f"Successfully generated embeddings for {len(chunks_with_embeddings)} chunks")
                
                # Store in vector database
                print(f"Adding document to vector store: {processed_doc['id']}")
                vector_store_result = vector_store.add_document(processed_doc['id'], chunks_with_embeddings)
                
                if not vector_store_result:
                    print(f"Warning: Vector store reported failure for document: {processed_doc['id']}")
                    # Continue anyway since we'll still save to the database
                
                # Save document to database
                doc = Document(
                    id=processed_doc['id'],
                    filename=filename,
                    title=processed_doc['metadata'].get('title') or processed_doc['metadata'].get('extracted_title'),
                    num_pages=processed_doc['metadata'].get('page_count'),
                    num_chunks=len(processed_doc['chunks']),
                    file_size=os.path.getsize(file_path),
                    chunk_strategy=chunk_strategy,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    kg_processed=False,
                    kg_entity_count=None,
                    kg_relationship_count=None,
                    kg_processing_time=None
                )
                
                # Save chunks to database
                for i, chunk in enumerate(processed_doc['chunks']):
                    db_chunk = Chunk(
                        document_id=processed_doc['id'],
                        chunk_index=i,
                        text=chunk['text'],
                        page_num=chunk.get('page_num'),
                        chunk_metadata=chunk.get('metadata', {})
                    )
                    doc.chunks.append(db_chunk)
                
                # Commit to database
                db.session.add(doc)
                db.session.commit()
                
                # We'll add the document to the knowledge graph in a separate process after returning
                # to avoid timeout issues, but we'll still track it in the database
                
                # Add background processing job flag if we support it later
                
                # Redirect to document view
                return redirect(url_for('document.view', doc_id=processed_doc['id']))
                
            except Exception as e:
                db.session.rollback()
                flash(f'Error processing document: {str(e)}', 'danger')
                return redirect(request.url)
                
        else:
            flash('File type not allowed. Please upload a PDF.', 'danger')
            return redirect(request.url)
            
    # If GET request, show upload form
    return render_template('upload.html')

@bp.route('/view/<doc_id>')
def view(doc_id):
    """View a processed document."""
    try:
        # Get document from database
        document = Document.query.get(doc_id)
        
        if not document:
            flash('Document not found', 'danger')
            return redirect(url_for('index'))
            
        # For PDF viewing in browser
        pdf_url = url_for('document.serve_pdf', filename=document.filename)
        
        # Get visualization data
        visualization_data = visualizer.create_chunk_visualization(doc_id, vector_store)
        
        return render_template(
            'document_view.html',
            document=document,
            doc_id=doc_id,
            filename=document.filename,
            pdf_url=pdf_url,
            visualization_data=json.dumps(visualization_data)
        )
        
    except Exception as e:
        flash(f'Error viewing document: {str(e)}', 'danger')
        return redirect(url_for('index'))

@bp.route('/pdf/<filename>')
def serve_pdf(filename):
    """Serve the PDF file."""
    return send_from_directory(UPLOAD_FOLDER, filename)

@bp.route('/chunks/<doc_id>')
def get_chunks(doc_id):
    """Get chunks for a document."""
    try:
        # Get chunks from database
        chunks = Chunk.query.filter_by(document_id=doc_id).order_by(Chunk.chunk_index).all()
        
        # Format chunks for response
        chunk_list = [chunk.to_dict() for chunk in chunks]
        
        return jsonify({'chunks': chunk_list})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/delete/<doc_id>', methods=['POST'])
def delete_document(doc_id):
    """Delete a document and its chunks."""
    try:
        # Get document from database
        document = Document.query.get(doc_id)
        
        if not document:
            flash('Document not found', 'danger')
            return redirect(url_for('index'))
        
        # Delete from vector store
        vector_store.delete_document(doc_id)
        
        # Delete from knowledge graph
        try:
            print(f"Removing document from knowledge graph: {doc_id}")
            # Create a new document with empty chunks to remove it
            knowledge_graph.add_document(doc_id, [])
            print(f"Document removed from knowledge graph")
        except Exception as kg_error:
            print(f"Warning: Failed to remove document from knowledge graph: {str(kg_error)}")
            # Continue anyway
        
        # Delete the file
        if os.path.exists(os.path.join(UPLOAD_FOLDER, document.filename)):
            os.remove(os.path.join(UPLOAD_FOLDER, document.filename))
        
        # Delete from database
        db.session.delete(document)  # This will also delete associated chunks due to cascade
        db.session.commit()
            
        flash('Document deleted successfully', 'success')
        return redirect(url_for('index'))
        
    except Exception as e:
        db.session.rollback()
        flash(f'Error deleting document: {str(e)}', 'danger')
        return redirect(url_for('document.view', doc_id=doc_id))

@bp.route('/visualization/<doc_id>')
def visualization(doc_id):
    """Get visualization data for a document."""
    try:
        visualization_data = visualizer.create_chunk_visualization(doc_id, vector_store)
        return jsonify(visualization_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/graph/<doc_id>')
def knowledge_graph_view(doc_id):
    """View knowledge graph for a document."""
    try:
        # Get document from database
        document = Document.query.get(doc_id)
        
        if not document:
            flash('Document not found', 'danger')
            return redirect(url_for('index'))
        
        # Check if document has been processed by knowledge graph already
        # If not, process it now
        if not document.kg_processed:
            try:
                print(f"Processing document for knowledge graph: {doc_id}")
                start_time = time.time()
                
                # Get chunks from database
                chunks = Chunk.query.filter_by(document_id=doc_id).order_by(Chunk.chunk_index).all()
                
                # Convert DB chunks to dictionary format for knowledge graph
                kg_chunks = [
                    {
                        'text': chunk.text,
                        'chunk_index': chunk.chunk_index,
                        'page_num': chunk.page_num,
                    }
                    for chunk in chunks
                ]
                
                # Process document with knowledge graph
                kg_result = knowledge_graph.add_document(doc_id, kg_chunks)
                
                # Update document with knowledge graph stats
                processing_time = time.time() - start_time
                document.kg_processed = True
                document.kg_entity_count = kg_result['entity_count']
                document.kg_relationship_count = kg_result['relationship_count']
                document.kg_processing_time = processing_time
                
                # Save to database
                db.session.commit()
                
                # Sync extracted entities/relationships to PostgreSQL graph tables
                try:
                    sync_result = knowledge_graph.sync_to_db(doc_id=doc_id)
                    print(f"Graph sync: {sync_result['entities_upserted']} entities, "
                          f"{sync_result['relationships_upserted']} relationships")
                except Exception as sync_err:
                    print(f"Warning: graph sync to DB failed: {sync_err}")
                
                print(f"Knowledge graph processing complete in {processing_time:.2f} seconds: "
                      f"{kg_result['entity_count']} entities and {kg_result['relationship_count']} relationships found")
            except Exception as kg_error:
                print(f"Warning: Knowledge graph processing failed: {str(kg_error)}")
                # Continue anyway to show existing graph data if available
        
        # Generate graph visualization
        graph_path = knowledge_graph.visualize(doc_id=doc_id, output_path=f'static/graph_{doc_id}.html')
        
        # Get top entities for the document
        entity_stats = knowledge_graph._get_top_entities(doc_id, limit=20)
        
        # Get graph stats
        graph_stats = knowledge_graph.get_stats()
        
        return render_template(
            'knowledge_graph.html',
            document=document,
            doc_id=doc_id,
            graph_embed_path=f'/static/graph_{doc_id}.html',
            entity_stats=entity_stats,
            graph_stats=graph_stats
        )
        
    except Exception as e:
        flash(f'Error generating knowledge graph: {str(e)}', 'danger')
        return redirect(url_for('document.view', doc_id=doc_id))

@bp.route('/graph/api/entities/<doc_id>')
def get_document_entities(doc_id):
    """Get entities for a document."""
    try:
        # Get top entities for the document
        entity_stats = knowledge_graph._get_top_entities(doc_id, limit=50)
        return jsonify({'entities': entity_stats})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/graph/api/search')
def search_knowledge_graph():
    """Search the knowledge graph."""
    try:
        query = request.args.get('q', '')
        doc_id = request.args.get('doc_id', None)
        
        if not query:
            return jsonify({'entities': [], 'relationships': []})
            
        # Search the knowledge graph
        results = knowledge_graph.search(query, doc_id=doc_id)
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/graph/api/migrate', methods=['POST'])
def migrate_graph_to_db():
    """
    Migrate the in-memory NetworkX knowledge graph to PostgreSQL.

    This is a one-time (or incremental) operation. For large graphs it may
    take tens of seconds; call it from a background job or admin UI.
    Pass an optional ``doc_id`` in the JSON body to migrate only that
    document's entities; omit for a full migration.
    """
    try:
        data = request.json or {}
        doc_id = data.get('doc_id', None)

        result = knowledge_graph.sync_to_db(doc_id=doc_id)

        return jsonify({
            'status': 'ok',
            'entities_upserted': result['entities_upserted'],
            'relationships_upserted': result['relationships_upserted'],
            'doc_id': doc_id,
        })
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500