import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import streamlit as st
import google.generativeai as genai
import threading

logger = logging.getLogger(__name__)

class LegalConversationalRAG:
    """Simple Legal Conversational RAG with PROPER document context passing"""
    
    def __init__(self, gemini_api_key: str):
        # Configure Gemini
        genai.configure(api_key=gemini_api_key)
        self.gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Simple in-memory conversation history
        self.conversation_history = []
        self.current_document_context = None
        self._lock = threading.Lock()
        
        logger.info("Legal Conversational RAG initialized with proper context handling")
    
    def process_legal_conversation(self, question: str, document_context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        FIXED: Process conversation with PROPER document text in prompt
        """
        try:
            with self._lock:
                # Store the conversation
                self.conversation_history.append({
                    'user': question,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Keep only last 10 interactions
                if len(self.conversation_history) > 10:
                    self.conversation_history = self.conversation_history[-10:]
                
                # Generate response with FIXED document context
                response = self._generate_response_with_document_context(question, document_context)
                
                # Store AI response
                self.conversation_history.append({
                    'ai': response['answer'],
                    'timestamp': datetime.now().isoformat()
                })
                
                return response
                
        except Exception as e:
            logger.error(f"Chat processing failed: {e}")
            return {
                'answer': f"I apologize, but I encountered an error: {str(e)}. Please try rephrasing your question.",
                'sources': [],
                'confidence': 0.3,
                'timestamp': datetime.now().isoformat()
            }
    
    def _generate_response_with_document_context(self, question: str, document_context: Optional[Dict] = None) -> Dict[str, Any]:
        """FIXED: Generate response with ACTUAL document text in prompt"""
        
        try:
            # Build prompt with ACTUAL document text
            prompt = self._build_prompt_with_document_text(question, document_context)
            
            # Debug: Show what's being sent to Gemini
            if document_context:
                logger.info(f"Sending document context to Gemini: {len(prompt)} chars")
                print(f"🔍 DEBUG: Document context available: {document_context.get('document_data', {}).get('filename', 'Unknown')}")
            else:
                logger.info("No document context - general legal question")
                print("🔍 DEBUG: No document context")
            
            # Call Gemini with the complete prompt
            response = self.gemini_model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Simple source tracking
            sources = []
            if document_context:
                doc_data = document_context.get('document_data', {})
                filename = doc_data.get('document_metadata', {}).get('filename', 'Document')
                sources = [{'reference': f"Document: {filename}", 'type': 'document'}]
            
            return {
                'answer': response_text,
                'sources': sources,
                'confidence': 0.8,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return {
                'answer': self._get_fallback_response(question, document_context),
                'sources': [],
                'confidence': 0.4,
                'timestamp': datetime.now().isoformat()
            }
    
    def _build_prompt_with_document_text(self, question: str, document_context: Optional[Dict] = None) -> str:
        """CRITICAL FIX: Build prompt with ACTUAL document text included"""
        
        # Start with system instruction
        prompt = """You are a specialized AI legal assistant for Indian law. You provide clear, accurate, and helpful legal information.

IMPORTANT: You are analyzing a specific legal document. DO NOT ask the user to provide document text - it is already provided below.

"""
        
        # ADD ACTUAL DOCUMENT TEXT if available
        if document_context and document_context.get('document_data'):
            doc_data = document_context['document_data']
            document_text = doc_data.get('document_text', '')
            
            if document_text and document_text.strip():
                # Include the ACTUAL document text in the prompt
                filename = doc_data.get('document_metadata', {}).get('filename', 'Legal Document')
                doc_type = doc_data.get('document_type', 'Legal Document')
                
                prompt += f"""LEGAL DOCUMENT ANALYSIS:
Document Name: {filename}
Document Type: {doc_type}

FULL DOCUMENT TEXT:
{document_text}

INSTRUCTIONS:
- You are analyzing the above document text
- Answer questions based on this specific document content
- Reference specific sections, clauses, or terms from the document
- Provide insights based on Indian legal framework
- DO NOT ask for document text - it is provided above

"""
            else:
                prompt += "Note: Document uploaded but text extraction may have failed.\n\n"
        
        # Add conversation context if any
        if len(self.conversation_history) > 0:
            prompt += "PREVIOUS CONVERSATION:\n"
            for item in self.conversation_history[-2:]:  # Last 2 interactions
                if 'user' in item:
                    prompt += f"User: {item['user']}\n"
                elif 'ai' in item:
                    prompt += f"AI: {item['ai'][:100]}...\n"
            prompt += "\n"
        
        # Add the current question
        prompt += f"USER QUESTION: {question}\n\n"
        
        # Final instruction
        if document_context and document_context.get('document_data', {}).get('document_text'):
            prompt += "Please provide a detailed response based on the document text provided above. Reference specific parts of the document in your answer."
        else:
            prompt += "Please provide a helpful response about Indian law. If this relates to a document, please let the user know that document text is needed for specific analysis."
        
        return prompt
    
    def _get_fallback_response(self, question: str, document_context: Optional[Dict] = None) -> str:
        """Enhanced fallback responses"""
        
        question_lower = question.lower()
        
        # If we have document context, mention it
        if document_context:
            doc_data = document_context.get('document_data', {})
            filename = doc_data.get('document_metadata', {}).get('filename', 'your document')
            
            return f"""I'm having technical difficulties processing your question about {filename}.

However, I can provide general guidance:

If you're asking about **legal risks**, consider:
- Review all contractual obligations and liabilities
- Check compliance with applicable Indian laws
- Identify potential dispute resolution issues

If you're asking about **financial terms**:
- Review payment schedules and amounts
- Check penalty clauses and interest rates
- Verify currency and calculation methods

If you're asking about **dates and deadlines**:
- Note all performance deadlines
- Check notice periods and termination dates
- Review renewal and expiration terms

For specific analysis of your document, please try asking again or consult with a legal professional."""
        
        # General legal responses
        if any(word in question_lower for word in ['contract', 'agreement']):
            return """A contract under Indian law (Indian Contract Act, 1872) requires:

1. **Offer and Acceptance**: Clear proposal and acceptance
2. **Consideration**: Something of value exchanged
3. **Legal Capacity**: Parties must be legally competent
4. **Free Consent**: Agreement without coercion, fraud, or mistake
5. **Lawful Object**: Purpose must be legal

Key sections to understand:
- Section 10: Essential elements of valid contract
- Section 2(a): Definition of offer
- Section 2(b): Definition of acceptance

For specific contract questions, professional legal advice is recommended."""
        
        else:
            return f"""I understand you're asking: "{question}"

I'm here to help with legal questions about Indian law. For specific document analysis, please ensure:

1. The document was uploaded successfully
2. Text was extracted properly
3. Try rephrasing your question more specifically

For important legal matters, consulting with a qualified legal professional is always recommended."""
    
    def clear_conversation(self):
        """Clear conversation history"""
        with self._lock:
            self.conversation_history = []
            self.current_document_context = None
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get simple conversation statistics"""
        return {
            'total_interactions': len(self.conversation_history),
            'session_duration': 'Active session',
            'last_interaction': datetime.now().isoformat()
        }
    
    def set_document_context(self, document_context: Dict[str, Any]):
        """Set current document context"""
        with self._lock:
            self.current_document_context = document_context

# Test function with document context
def test_with_document_context():
    """Test with actual document context"""
    import os
    
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ GEMINI_API_KEY not found")
        return False
    
    try:
        print("🧪 Testing Legal RAG with Document Context...")
        rag = LegalConversationalRAG(api_key)
        
        # Test with mock document context
        mock_document_context = {
            'document_data': {
                'document_text': '''EMPLOYMENT AGREEMENT

This Employment Agreement ("Agreement") is entered into on January 1, 2024, between ABC Corporation ("Company"), a company incorporated under the laws of India, and John Doe ("Employee").

1. POSITION AND DUTIES
The Employee shall serve as Senior Software Engineer and shall perform duties including software development, code review, and project management.

2. COMPENSATION
The Employee shall receive a monthly salary of Rs. 1,50,000 (One Lakh Fifty Thousand Rupees) payable on the last working day of each month.

3. TERM
This Agreement shall commence on January 1, 2024, and shall continue for a period of 2 years unless terminated earlier.

4. CONFIDENTIALITY
The Employee agrees to maintain confidentiality of all proprietary information of the Company.

5. TERMINATION
Either party may terminate this Agreement with 30 days written notice.''',
                'document_metadata': {'filename': 'employment_agreement.pdf'},
                'document_type': 'Employment Agreement'
            }
        }
        
        # Test question about the document
        response = rag.process_legal_conversation(
            "What is the monthly salary mentioned in this agreement?", 
            mock_document_context
        )
        
        if isinstance(response, dict) and response.get('answer'):
            answer = response['answer']
            print("✅ Document context test passed")
            print(f"📄 Response: {answer[:200]}...")
            
            # Check if the response includes the salary amount
            if '1,50,000' in answer or '150000' in answer or 'one lakh fifty thousand' in answer.lower():
                print("✅ Document text was properly processed!")
                return True
            else:
                print("❌ Document text may not have been processed correctly")
                print(f"Full response: {answer}")
                return False
        else:
            print("❌ Test failed - invalid response")
            return False
            
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

if __name__ == "__main__":
    success = test_with_document_context()
    print("✅ Context passing works!" if success else "❌ Context passing needs fix")
