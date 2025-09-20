"""
Simple embeddings manager for Legal Eagle MVP
"""
from typing import Dict, List, Any, Optional
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class HybridEmbeddingsManager:
    """Simple embeddings manager for demo"""
    
    def __init__(self):
        # Simple in-memory storage for demo
        self.documents = []
        logger.info("Simple embeddings manager initialized")
    
    def add_document(self, doc_id: str, text: str, metadata: Dict = None):
        """Add document to simple storage"""
        try:
            self.documents.append({
                'id': doc_id,
                'text': text[:2000],  # Store only first 2000 chars for demo
                'metadata': metadata or {},
                'added_at': datetime.now().isoformat()
            })
            logger.info(f"Added document: {doc_id}")
        except Exception as e:
            logger.error(f"Failed to add document: {e}")
    
    def search_similar_documents(self, query: str, top_k: int = 3) -> List[Dict]:
        """Simple keyword-based search for demo"""
        try:
            if not self.documents:
                return []
            
            # Simple keyword matching
            query_words = set(query.lower().split())
            results = []
            
            for doc in self.documents:
                doc_words = set(doc['text'].lower().split())
                overlap = len(query_words.intersection(doc_words))
                
                if overlap > 0:
                    results.append({
                        'title': doc.get('metadata', {}).get('filename', 'Document'),
                        'relevance': overlap,
                        'content': doc['text'][:200]
                    })
            
            # Sort by relevance and return top results
            results.sort(key=lambda x: x['relevance'], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
