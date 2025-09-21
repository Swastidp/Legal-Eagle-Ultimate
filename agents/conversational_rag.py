import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import streamlit as st
import google.generativeai as genai
import threading

logger = logging.getLogger(__name__)

class LegalConversationalRAG:
    """FIXED: Legal Conversational RAG with PROPER document context handling"""
    
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
        FIXED: Process conversation with PROPER document text structure matching app.py
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
                    'ai': response.get('response', response.get('answer', 'No response')),
                    'timestamp': datetime.now().isoformat()
                })
                
                return response
                
        except Exception as e:
            logger.error(f"Chat processing failed: {e}")
            return {
                'response': f"I apologize, but I encountered an error: {str(e)}. Please try rephrasing your question.",
                'sources': [],
                'confidence_score': 0.3,
                'timestamp': datetime.now().isoformat()
            }

    def _generate_response_with_document_context(self, question: str, document_context: Optional[Dict] = None) -> Dict[str, Any]:
        """FIXED: Generate response with ACTUAL document text structure from app.py"""
        try:
            # Build prompt with ACTUAL document text structure
            prompt = self._build_prompt_with_document_text(question, document_context)
            
            # Debug: Show what's being sent to Gemini
            if document_context and document_context.get('document_text'):
                logger.info(f"Sending document context to Gemini: {len(prompt)} chars")
                logger.info(f"Document text length: {len(document_context['document_text'])} chars")
            else:
                logger.info("No document context - general legal question")
            
            # Call Gemini with the complete prompt
            response = self.gemini_model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Simple source tracking
            sources = []
            if document_context and document_context.get('document_text'):
                sources = [{'reference': 'Uploaded Document', 'type': 'document'}]
            
            return {
                'response': response_text,
                'sources': sources,
                'confidence_score': 0.8,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return {
                'response': self._get_fallback_response(question, document_context),
                'sources': [],
                'confidence_score': 0.4,
                'timestamp': datetime.now().isoformat()
            }

    def _build_prompt_with_document_text(self, question: str, document_context: Optional[Dict] = None) -> str:
        """CRITICAL FIX: Build prompt with ACTUAL document text structure from app.py"""
        
        # Start with system instruction
        prompt = """You are a specialized AI legal assistant for Indian law. You provide clear, accurate, and helpful legal information.

IMPORTANT: If a document is provided, analyze it thoroughly and answer questions based on its content.
"""
        
        # ADD ACTUAL DOCUMENT TEXT if available (matching app.py structure)
        if document_context and document_context.get('document_text'):
            document_text = document_context['document_text']
            
            if document_text and document_text.strip():
                # Include the ACTUAL document text in the prompt
                prompt += f"""LEGAL DOCUMENT ANALYSIS:

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
            for item in self.conversation_history[-4:]:  # Last 4 interactions
                if 'user' in item:
                    prompt += f"User: {item['user']}\n"
                elif 'ai' in item:
                    prompt += f"AI: {item['ai'][:100]}...\n"
            prompt += "\n"
        
        # Add the current question
        prompt += f"USER QUESTION: {question}\n\n"
        
        # Final instruction
        if document_context and document_context.get('document_text'):
            prompt += "Please provide a detailed response based on the document text provided above. Reference specific parts of the document in your answer."
        else:
            prompt += """Please provide a helpful response about Indian law. Some topics I can help with:

1. **Contract Law** (Indian Contract Act 1872)
2. **Corporate Law** (Companies Act 2013)
3. **Employment Law** (Industrial Relations Code 2020)
4. **Property Law** (Transfer of Property Act 1882)
5. **Cyber Law** (Information Technology Act 2000)
6. **Consumer Protection** (Consumer Protection Act 2019)
7. **Constitutional Law** and fundamental rights

Feel free to ask specific questions about any of these areas of Indian law."""
        
        return prompt

    def _get_fallback_response(self, question: str, document_context: Optional[Dict] = None) -> str:
        """Enhanced fallback responses for better user experience"""
        question_lower = question.lower()
        
        # If we have document context, mention it
        if document_context and document_context.get('document_text'):
            return f"""I'm having technical difficulties processing your question about the document.

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

Please try rephrasing your question more specifically, or consult with a legal professional for detailed analysis."""
        
        # General legal responses based on keywords
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

For specific contract questions, please upload the contract document for detailed analysis."""
            
        elif any(word in question_lower for word in ['company', 'corporate', 'director']):
            return """Under the Companies Act 2013:

**Key Provisions:**
- Section 149: Board composition requirements
- Section 166: Directors' duties and responsibilities
- Section 177: Audit committee requirements
- Section 188: Related party transactions

**Director Duties:**
- Act in good faith
- Exercise due care and diligence
- Avoid conflicts of interest
- Comply with company's memorandum and articles

For specific corporate legal advice, please consult with a qualified corporate lawyer."""
            
        elif any(word in question_lower for word in ['employment', 'worker', 'salary', 'job']):
            return """Employment law in India is governed by:

**Industrial Relations Code 2020:**
- Section 25: Industrial dispute resolution
- Section 13: Works committees
- Section 77: Strikes and lockouts

**Key Employment Rights:**
- Minimum wages as per state laws
- Equal pay for equal work
- Safe working conditions
- Social security benefits

**Termination:**
- Notice periods as per contract or law
- Severance pay requirements
- Due process requirements

For specific employment disputes, please consult with a labor law specialist."""
        
        else:
            return f"""I understand you're asking: "{question}"

I'm here to help with legal questions about Indian law. I can provide guidance on:

**Major Indian Legal Areas:**
- Contract Law (Indian Contract Act 1872)
- Corporate Law (Companies Act 2013)
- Employment Law (Industrial Relations Code 2020)
- Property Law (Transfer of Property Act 1882)
- Cyber Law (Information Technology Act 2000)
- Consumer Protection (Consumer Protection Act 2019)

For document-specific analysis:
1. Upload your document using the Document Analysis feature
2. Then ask questions about it here

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

# Test function with the correct document context structure
def test_with_app_context():
    """Test with the actual document context structure from app.py"""
    import os
    
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ GEMINI_API_KEY not found")
        return False
    
    try:
        print("🧪 Testing Legal RAG with App Context Structure...")
        rag = LegalConversationalRAG(api_key)
        
        # Test with mock document context matching app.py structure
        mock_document_context = {
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
                        'analysis_available': True
        }
        
        # Test question about the document
        response = rag.process_legal_conversation(
            "What is the monthly salary mentioned in this agreement?",
            mock_document_context
        )
        
        if isinstance(response, dict) and response.get('response'):
            answer = response['response']
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
            print(f"Response: {response}")
            return False
            
    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = test_with_app_context()
    print("✅ Fixed RAG works!" if success else "❌ RAG needs more fixes")
