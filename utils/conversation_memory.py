from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import streamlit as st

class ConversationMemory:
    """Manages conversation history and context for Legal RAG chatbot"""
    
    def __init__(self, max_history: int = 50):
        self.max_history = max_history
        self.conversation_history = []
        self.current_document_context = None
        self.session_metadata = {}
        
        # Initialize session state for Streamlit
        if 'conversation_memory' not in st.session_state:
            st.session_state.conversation_memory = {
                'history': [],
                'document_context': None,
                'metadata': {}
            }
        
        self._load_from_session()
    
    def _load_from_session(self):
        """Load conversation data from Streamlit session state"""
        session_data = st.session_state.conversation_memory
        self.conversation_history = session_data.get('history', [])
        self.current_document_context = session_data.get('document_context')
        self.session_metadata = session_data.get('metadata', {})
    
    def _save_to_session(self):
        """Save conversation data to Streamlit session state"""
        st.session_state.conversation_memory = {
            'history': self.conversation_history,
            'document_context': self.current_document_context,
            'metadata': self.session_metadata
        }
    
    def add_interaction(self, user_query: str, ai_response: str, 
                       context_type: str = "general", metadata: Dict[str, Any] = None):
        """Add user-AI interaction to conversation history"""
        
        interaction = {
            'id': len(self.conversation_history) + 1,
            'timestamp': datetime.now().isoformat(),
            'user_query': user_query,
            'ai_response': ai_response,
            'context_type': context_type,
            'metadata': metadata or {}
        }
        
        self.conversation_history.append(interaction)
        
        # Maintain max history limit
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
        
        self._save_to_session()
    
    def set_document_context(self, document_id: str, document_data: Dict[str, Any]):
        """Set current document context for conversation"""
        self.current_document_context = {
            'document_id': document_id,
            'document_data': document_data,
            'set_at': datetime.now().isoformat()
        }
        self._save_to_session()
    
    def get_relevant_context(self, query: str, max_interactions: int = 5) -> List[Dict[str, Any]]:
        """Get relevant conversation context for current query"""
        if not self.conversation_history:
            return []
        
        # Simple relevance scoring based on keyword overlap
        query_words = set(query.lower().split())
        scored_interactions = []
        
        for interaction in self.conversation_history[-max_interactions*2:]:  # Look at recent interactions
            # Score based on query similarity
            interaction_words = set(interaction['user_query'].lower().split())
            overlap_score = len(query_words.intersection(interaction_words))
            
            # Boost score for recent interactions
            recency_score = 1.0 / (len(self.conversation_history) - self.conversation_history.index(interaction) + 1)
            
            total_score = overlap_score + recency_score
            
            if total_score > 0:
                scored_interactions.append((total_score, interaction))
        
        # Sort by score and return top interactions
        scored_interactions.sort(key=lambda x: x[0], reverse=True)
        return [interaction for _, interaction in scored_interactions[:max_interactions]]
    
    def get_conversation_summary(self) -> str:
        """Generate summary of current conversation"""
        if not self.conversation_history:
            return "No previous conversation"
        
        recent_topics = []
        for interaction in self.conversation_history[-5:]:  # Last 5 interactions
            query = interaction['user_query']
            if len(query) > 100:
                query = query[:100] + "..."
            recent_topics.append(f"- {query}")
        
        summary = f"Recent conversation topics ({len(self.conversation_history)} total interactions):\n"
        summary += "\n".join(recent_topics)
        
        if self.current_document_context:
            summary += f"\n\nCurrent document context: {self.current_document_context['document_id']}"
        
        return summary
    
    def clear_conversation(self):
        """Clear conversation history"""
        self.conversation_history = []
        self.current_document_context = None
        self.session_metadata = {}
        self._save_to_session()
    
    def export_conversation(self) -> str:
        """Export conversation history as JSON"""
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'total_interactions': len(self.conversation_history),
            'document_context': self.current_document_context,
            'conversation_history': self.conversation_history,
            'session_metadata': self.session_metadata
        }
        
        return json.dumps(export_data, indent=2)
    
    def get_conversation_statistics(self) -> Dict[str, Any]:
        """Get conversation statistics"""
        if not self.conversation_history:
            return {
                'total_interactions': 0,
                'session_duration': '0 minutes',
                'topics_discussed': [],
                'avg_response_length': 0
            }
        
        # Calculate session duration
        first_interaction = datetime.fromisoformat(self.conversation_history[0]['timestamp'])
        last_interaction = datetime.fromisoformat(self.conversation_history[-1]['timestamp'])
        duration_minutes = int((last_interaction - first_interaction).total_seconds() / 60)
        
        # Extract topics/keywords
        all_queries = " ".join([interaction['user_query'] for interaction in self.conversation_history])
        words = all_queries.lower().split()
        
        # Simple keyword extraction (legal terms)
        legal_keywords = ['contract', 'agreement', 'liability', 'clause', 'section', 'act', 'law', 'legal', 'court', 'judgment']
        topics_discussed = list(set([word for word in words if word in legal_keywords]))
        
        # Average response length
        avg_response_length = sum(len(interaction['ai_response']) for interaction in self.conversation_history) // len(self.conversation_history)
        
        return {
            'total_interactions': len(self.conversation_history),
            'session_duration': f'{duration_minutes} minutes',
            'topics_discussed': topics_discussed[:10],  # Top 10 topics
            'avg_response_length': avg_response_length,
            'document_context_set': bool(self.current_document_context)
        }
    
    def format_context_for_llm(self, max_context_length: int = 2000) -> str:
        """Format conversation context for LLM input"""
        if not self.conversation_history:
            context = "No previous conversation context."
        else:
            context_parts = []
            
            # Add document context if available
            if self.current_document_context:
                doc_info = self.current_document_context['document_data']
                context_parts.append(f"Current document: {doc_info.get('filename', 'Unknown')} ({doc_info.get('type', 'Unknown type')})")
            
            # Add recent interactions
            recent_interactions = self.conversation_history[-3:]  # Last 3 interactions
            for interaction in recent_interactions:
                context_parts.append(f"User: {interaction['user_query']}")
                context_parts.append(f"Assistant: {interaction['ai_response'][:200]}...")  # Truncate long responses
            
            context = "\n".join(context_parts)
        
        # Truncate if too long
        if len(context) > max_context_length:
            context = context[:max_context_length] + "... [truncated]"
        
        return context
