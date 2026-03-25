import json
import numpy as np
from flask import Blueprint, render_template, request, jsonify

from config import KNOWLEDGE_GRAPH_PATH
from models import db, Document, Query, Chunk, QueryChunk
from rag.embeddings import EmbeddingGenerator
from rag.vector_store import VectorStore
from rag.retriever import Retriever
from rag.generator import Generator
from rag.knowledge_graph import KnowledgeGraph
from rag.query_router import QueryRouter

bp = Blueprint('query', __name__, url_prefix='/query')

embedding_generator = EmbeddingGenerator()
vector_store = VectorStore()
knowledge_graph = KnowledgeGraph(db_path=KNOWLEDGE_GRAPH_PATH)
retriever = Retriever(vector_store, embedding_generator, knowledge_graph=knowledge_graph)
generator = Generator()
query_router = QueryRouter()


def _clean_chunks(chunks):
    """Strip embeddings and convert numpy scalars to native Python types."""
    clean = []
    for chunk in chunks:
        c = {}
        for k, v in chunk.items():
            if k == 'embedding':
                continue
            elif isinstance(v, (np.float32, np.float64)):
                c[k] = float(v)
            elif isinstance(v, (np.int32, np.int64)):
                c[k] = int(v)
            elif isinstance(v, dict):
                # Recursively clean nested dicts (e.g. routing)
                c[k] = {
                    dk: (float(dv) if isinstance(dv, (np.float32, np.float64))
                         else int(dv) if isinstance(dv, (np.int32, np.int64))
                         else dv)
                    for dk, dv in v.items()
                }
            else:
                c[k] = v
        clean.append(c)
    return clean


@bp.route('/', methods=['GET'])
def query_page():
    documents = Document.query.all()
    return render_template('query.html', documents=documents)


@bp.route('/ask', methods=['POST'])
def ask():
    """
    Process a query and return a generated answer.

    Request JSON::
        {
            "query": "...",
            "doc_id": "optional",
            "top_k": 5,
            "temperature": 0.7
        }

    Response JSON::
        {
            "response": {"text": "...", "sources": [...]},
            "chunks": [...],
            "query_id": 1,
            "routing": {"mode": "hybrid", ...},
            "graph_info": {"query_entities": [...], "all_entities": [...]}
        }
    """
    try:
        data = request.json
        query_text = data.get('query')
        doc_id = data.get('doc_id') or None
        top_k = int(data.get('top_k', 5))
        temperature = float(data.get('temperature', 0.7))

        if not query_text:
            return jsonify({'error': 'Query is required'}), 400

        # ── Route the query first (for response metadata) ──────────────
        routing_info = query_router.route(query_text)

        # ── Retrieve chunks (full pipeline: route → graph → search) ───
        try:
            retrieved_chunks = retriever.retrieve(
                query_text, doc_id=doc_id, top_k=top_k
            )
            print(f"Retrieved {len(retrieved_chunks)} chunks "
                  f"(mode={routing_info['mode_label']})")
        except Exception as exc:
            import traceback
            print(f"Retrieval error: {exc}\n{traceback.format_exc()}")
            retrieved_chunks = []

        # ── Generate response ──────────────────────────────────────────
        if not retrieved_chunks:
            response_data = {
                'text': (
                    "I couldn't find any relevant information to answer your question. "
                    "Please try rephrasing or selecting a different document."
                ),
                'sources': [],
            }
        else:
            try:
                response_data = generator.generate_response(
                    query=query_text,
                    chunks=retrieved_chunks,
                    temperature=temperature,
                )
            except Exception as gen_exc:
                import traceback
                print(f"Generator error: {gen_exc}\n{traceback.format_exc()}")
                text = f"Here is what I found about '{query_text}':\n\n"
                for i, chunk in enumerate(retrieved_chunks[:3]):
                    text += f"[{i+1}] {chunk.get('text', '')[:300]}...\n\n"
                response_data = {
                    'text': text,
                    'sources': [
                        {
                            'chunk_index': i,
                            'text': c.get('text', ''),
                            'metadata': {
                                k: v for k, v in c.items()
                                if k not in ('text', 'embedding', 'score')
                            },
                        }
                        for i, c in enumerate(retrieved_chunks[:3])
                    ],
                }

        # ── Persist query to database ──────────────────────────────────
        query_record = Query(
            query_text=query_text,
            response_text=response_data['text'],
            document_id=doc_id,
            top_k=top_k,
            temperature=temperature,
        )

        if retrieved_chunks and doc_id:
            for chunk_data in retrieved_chunks:
                chunk_index = chunk_data.get('chunk_index', 0)
                db_chunk = Chunk.query.filter_by(
                    document_id=doc_id, chunk_index=chunk_index
                ).first()
                if db_chunk:
                    query_record.chunks.append(
                        QueryChunk(chunk=db_chunk,
                                   relevance_score=float(chunk_data.get('score', 0.0)))
                    )

        db.session.add(query_record)
        db.session.commit()

        # ── Build response ─────────────────────────────────────────────
        routing_payload = retrieved_chunks[0].get('routing', {}) if retrieved_chunks else {}
        graph_info = {
            'query_entities': routing_payload.get('graph_entities', []),
            'mode': routing_info['mode_label'],
        }

        return jsonify({
            'response': response_data,
            'chunks': _clean_chunks(retrieved_chunks),
            'query_id': query_record.id,
            'routing': {
                'mode': routing_info['mode_label'],
                'keyword_score': routing_info['keyword_score'],
                'vector_score': routing_info['vector_score'],
                'reasons': routing_info.get('reasons', []),
            },
            'graph_info': graph_info,
        })

    except Exception as exc:
        db.session.rollback()
        import traceback
        print(f"ask() error: {exc}\n{traceback.format_exc()}")
        return jsonify({'error': str(exc)}), 500


@bp.route('/search', methods=['POST'])
def search():
    """
    Search for relevant chunks without generating an answer.

    Request JSON::
        {
            "query": "...",
            "doc_id": "optional",
            "top_k": 5
        }
    """
    try:
        data = request.json
        query_text = data.get('query')
        doc_id = data.get('doc_id') or None
        top_k = int(data.get('top_k', 5))

        if not query_text:
            return jsonify({'error': 'Query is required'}), 400

        routing_info = query_router.route(query_text)

        try:
            chunks = retriever.retrieve(query_text, doc_id=doc_id, top_k=top_k)
            print(f"Search returned {len(chunks)} chunks "
                  f"(mode={routing_info['mode_label']})")
        except Exception as exc:
            import traceback
            print(f"Search error: {exc}\n{traceback.format_exc()}")
            chunks = []

        if not chunks:
            return jsonify({
                'chunks': [],
                'count': 0,
                'routing': routing_info['mode_label'],
                'message': (
                    "No relevant chunks found. "
                    "Try rephrasing or selecting a different document."
                ),
            })

        return jsonify({
            'chunks': _clean_chunks(chunks),
            'count': len(chunks),
            'routing': {
                'mode': routing_info['mode_label'],
                'keyword_score': routing_info['keyword_score'],
                'vector_score': routing_info['vector_score'],
            },
        })

    except Exception as exc:
        import traceback
        print(f"search() error: {exc}\n{traceback.format_exc()}")
        return jsonify({'error': str(exc)}), 500


@bp.route('/documents', methods=['GET'])
def get_documents():
    try:
        docs = Document.query.all()
        return jsonify({'documents': [d.to_dict() for d in docs]})
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@bp.route('/history', methods=['GET'])
def query_history():
    try:
        queries = Query.query.order_by(Query.timestamp.desc()).limit(20).all()
        return jsonify({'queries': [q.to_dict() for q in queries]})
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@bp.route('/graph/multihop', methods=['POST'])
def graph_multihop():
    """
    Dedicated endpoint to inspect multi-hop graph traversal for a query.
    Useful for debugging / UI exploration.

    Request JSON::  {"query": "...", "doc_id": "optional", "hops": 2}
    """
    try:
        data = request.json
        query_text = data.get('query')
        doc_id = data.get('doc_id') or None
        hops = int(data.get('hops', 2))

        if not query_text:
            return jsonify({'error': 'Query is required'}), 400

        result = knowledge_graph.multi_hop_search_db(
            query_text, hops=hops, doc_id=doc_id
        )
        return jsonify(result)

    except Exception as exc:
        return jsonify({'error': str(exc)}), 500
