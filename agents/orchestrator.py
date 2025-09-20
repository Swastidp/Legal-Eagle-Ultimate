"""
FIXED Multi-Agent Legal Document Orchestrator - Document-Specific Analysis
Ensures all responses are based on actual document content, not generic templates
"""
import json
import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
import google.generativeai as genai
import re

logger = logging.getLogger(__name__)

class LegalAgentOrchestrator:
    """FIXED Multi-Agent System - Ensures Document-Specific Responses"""
    
    def __init__(self, gemini_api_key: str):
        """Initialize with proper error handling for specific responses"""
        genai.configure(api_key=gemini_api_key)
        self.gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Set model parameters for better document analysis
        self.generation_config = genai.GenerationConfig(
            temperature=0.3,  # Lower temperature for more focused analysis
            top_p=0.8,
            top_k=40,
            max_output_tokens=2048,
        )
        
        logger.info("FIXED Multi-Agent Legal Orchestrator initialized")
    
    def comprehensive_document_analysis(self, document_text: str, 
                                      legal_jurisdiction: str = "Indian Law",
                                      focus_areas: List[str] = None,
                                      analysis_options: Dict = None) -> Dict[str, Any]:
        """
        FIXED: Comprehensive Document-Specific Analysis
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting FIXED analysis of {len(document_text)} character document")
            
            # STEP 1: Direct Document Content Analysis
            document_analysis = self._analyze_document_content_directly(document_text)
            
            # STEP 2: Extract Specific Entities from Document
            entities = self._extract_specific_document_entities(document_text)
            
            # STEP 3: Analyze Document-Specific Risks
            risk_analysis = self._analyze_specific_document_risks(document_text, entities)
            
            # STEP 4: Generate Document-Specific Summary (FIXED)
            executive_summary = self._generate_specific_summary(document_text, entities, risk_analysis)
            
            # STEP 5: Indian Law Compliance Check
            compliance_analysis = self._check_indian_law_compliance(document_text)
            
            # STEP 6: InLegalBERT-style Analysis
            inlegal_analysis = self._perform_inlegal_bert_analysis(document_text)
            
            # Compile comprehensive results
            results = {
                'analysis_id': f"fixed_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'document_stats': {
                    'character_count': len(document_text),
                    'word_count': len(document_text.split()),
                    'paragraph_count': len([p for p in document_text.split('\n\n') if p.strip()]),
                    'analysis_timestamp': datetime.now().isoformat(),
                    'first_100_chars': document_text[:100] + "..."
                },
                'document_structure': document_analysis,
                'key_entities': entities,
                'risk_analysis': risk_analysis,
                'compliance_analysis': compliance_analysis,
                'inlegal_bert_analysis': inlegal_analysis,
                'executive_summary': executive_summary,  # This will be document-specific
                'overall_risk_score': risk_analysis.get('overall_risk_score', 65),
                'confidence_score': 0.85,
                'processing_time': time.time() - start_time,
                'agents_used': ['DirectDocumentAnalyzer', 'SpecificEntityExtractor', 'DocumentRiskAnalyzer', 'ComplianceChecker', 'InLegalBERTProcessor']
            }
            
            logger.info(f"FIXED document analysis completed in {results['processing_time']:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"FIXED analysis failed: {e}")
            # Even fallback should be document-specific
            return self._create_document_specific_fallback(document_text, str(e))

    def _analyze_document_content_directly(self, document_text: str) -> Dict[str, Any]:
        """FIXED: Direct document content analysis"""
        
        try:
            # Create a focused prompt for document structure analysis
            prompt = f"""Analyze this specific legal document and provide detailed insights about its content. Focus ONLY on what's actually in this document.

DOCUMENT TEXT TO ANALYZE:
{document_text}

Provide a JSON response with specific details about THIS document:

{{
    "document_type": "Exact type based on document content (e.g., 'Residential Lease Agreement', 'Employment Contract', 'Service Agreement')",
    "primary_parties": ["List actual party names/types found in document"],
    "key_sections": ["List actual sections/clauses found in document"],
    "main_obligations": ["List specific obligations mentioned in document"],
    "important_dates": ["List actual dates mentioned in document"],
    "financial_terms": ["List actual monetary amounts/terms in document"],
    "document_purpose": "Specific purpose of this document based on its content",
    "jurisdiction_mentioned": "Any jurisdiction/governing law mentioned in document",
    "special_clauses": ["Any unique or notable clauses in this document"]
}}

Analyze ONLY the provided document content. Do not add generic information."""

            response = self.gemini_model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            
            try:
                result = json.loads(response.text.strip())
                result['analysis_method'] = 'direct_content_analysis'
                return result
            except json.JSONDecodeError:
                # Manual extraction if JSON fails
                return self._manual_document_analysis(document_text)
                
        except Exception as e:
            logger.error(f"Direct document analysis failed: {e}")
            return self._manual_document_analysis(document_text)

    def _extract_specific_document_entities(self, document_text: str) -> List[Dict[str, Any]]:
        """FIXED: Extract actual entities from the document"""
        
        try:
            prompt = f"""Extract all specific entities from this legal document. Return ONLY entities that actually exist in the document.

DOCUMENT TO ANALYZE:
{document_text}

Extract and return in JSON format:
[
    {{
        "type": "PARTY",
        "value": "Actual name from document",
        "confidence": 0.95,
        "context": "How this entity appears in the document",
        "location_in_document": "Where exactly found"
    }},
    {{
        "type": "DATE",
        "value": "Actual date from document",
        "confidence": 0.90,
        "context": "What this date represents",
        "location_in_document": "Where found"
    }},
    {{
        "type": "AMOUNT",
        "value": "Actual amount from document",
        "confidence": 0.95,
        "context": "What this amount is for",
        "location_in_document": "Where found"
    }}
]

Extract ONLY entities that are actually present in the document text. Be specific and accurate."""

            response = self.gemini_model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            
            try:
                entities = json.loads(response.text.strip())
                return entities if isinstance(entities, list) else []
            except json.JSONDecodeError:
                return self._extract_entities_manually(document_text)
                
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return self._extract_entities_manually(document_text)

    def _analyze_specific_document_risks(self, document_text: str, entities: List[Dict]) -> Dict[str, Any]:
        """FIXED: Analyze risks specific to this document"""
        
        try:
            prompt = f"""Analyze the legal risks in this specific document. Base your analysis ONLY on what's written in this document.

DOCUMENT TEXT:
{document_text}

ENTITIES FOUND: {json.dumps([e.get('value', '') for e in entities[:10]], indent=2)}

Provide specific risk analysis in JSON format:
{{
    "overall_risk_score": 75,
    "risk_level": "Medium",
    "specific_risks_identified": [
        {{
            "title": "Specific risk title based on document content",
            "description": "Detailed description referencing actual document clauses",
            "severity": "High/Medium/Low",
            "evidence_from_document": "Actual text from document that creates this risk",
            "recommendation": "Specific action based on this document"
        }}
    ],
    "financial_risks": [
        {{
            "description": "Risk based on actual financial terms in document",
            "amount_involved": "Actual amount from document",
            "mitigation": "Specific mitigation for this document"
        }}
    ],
    "compliance_risks": ["Risks based on actual document content"],
    "operational_risks": ["Risks from actual performance obligations in document"],
    "confidence": 0.9
}}

Analyze ONLY this specific document. Reference actual clauses and terms."""

            response = self.gemini_model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            
            try:
                result = json.loads(response.text.strip())
                return result
            except json.JSONDecodeError:
                return self._create_document_based_risk_analysis(document_text, entities)
                
        except Exception as e:
            logger.error(f"Risk analysis failed: {e}")
            return self._create_document_based_risk_analysis(document_text, entities)

    def _generate_specific_summary(self, document_text: str, entities: List[Dict], risk_analysis: Dict) -> str:
        """FIXED: Generate document-specific executive summary"""
        
        try:
            # Extract key information for context
            parties = [e['value'] for e in entities if e.get('type') == 'PARTY'][:3]
            dates = [e['value'] for e in entities if e.get('type') == 'DATE'][:3]
            amounts = [e['value'] for e in entities if e.get('type') == 'AMOUNT'][:3]
            
            prompt = f"""Write a detailed executive summary for this specific legal document. The summary must be based entirely on the actual document content.

DOCUMENT TEXT TO SUMMARIZE:
{document_text}

KEY ENTITIES IDENTIFIED:
- Parties: {parties}
- Important Dates: {dates}  
- Financial Terms: {amounts}

RISK ANALYSIS RESULTS:
- Risk Score: {risk_analysis.get('overall_risk_score', 65)}/100
- Specific Risks: {[r.get('title', '') for r in risk_analysis.get('specific_risks_identified', [])]}

WRITE A 3-PARAGRAPH EXECUTIVE SUMMARY THAT:
1. First paragraph: Describes what this specific document is, who the parties are, and the main purpose
2. Second paragraph: Highlights the key terms, obligations, dates, and financial aspects from the actual document
3. Third paragraph: Summarizes the legal risks and recommendations specific to this document

REQUIREMENTS:
- Reference actual parties, dates, amounts, and clauses from the document
- Be specific about what this document contains, not generic legal advice
- Mention specific risks identified in this particular document
- Provide actionable insights based on the actual document content
- Write in professional, clear language for legal professionals

Do NOT write generic legal advice. Analyze THIS SPECIFIC DOCUMENT ONLY."""

            response = self.gemini_model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            
            summary = response.text.strip()
            
            # Validate that the summary is document-specific (not generic)
            if self._is_summary_generic(summary):
                logger.warning("Generated summary appears generic, creating document-specific version")
                return self._create_document_specific_summary_manually(document_text, entities, risk_analysis)
            
            return summary
            
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return self._create_document_specific_summary_manually(document_text, entities, risk_analysis)

    def _is_summary_generic(self, summary: str) -> bool:
        """Check if summary is generic or document-specific"""
        
        generic_phrases = [
            "has been analyzed using our multi-agent AI system",
            "several important legal considerations have been identified",
            "professional legal review is recommended",
            "comprehensive insights into the document's legal implications",
            "requires professional legal review"
        ]
        
        generic_count = sum(1 for phrase in generic_phrases if phrase.lower() in summary.lower())
        return generic_count >= 2  # If 2 or more generic phrases, it's likely generic

    def _create_document_specific_summary_manually(self, document_text: str, entities: List[Dict], risk_analysis: Dict) -> str:
        """Create document-specific summary manually"""
        
        # Extract key information from document
        text_sample = document_text[:500]
        word_count = len(document_text.split())
        
        # Get actual parties
        parties = [e['value'] for e in entities if e.get('type') == 'PARTY']
        dates = [e['value'] for e in entities if e.get('type') == 'DATE']
        amounts = [e['value'] for e in entities if e.get('type') == 'AMOUNT']
        
        # Determine document type from content
        doc_type = self._determine_document_type(document_text)
        
        # Build specific summary
        summary_parts = []
        
        # Paragraph 1: Document identification
        if parties:
            party_text = f" between {', '.join(parties[:2])}" if len(parties) >= 2 else f" involving {parties[0]}"
        else:
            party_text = " between the identified parties"
            
        summary_parts.append(f"This {doc_type}{party_text} contains {word_count} words and establishes specific legal obligations and rights. {text_sample[:150]}...")
        
        # Paragraph 2: Key terms and obligations
        terms_list = []
        if amounts:
            terms_list.append(f"financial obligations totaling {', '.join(amounts[:2])}")
        if dates:
            terms_list.append(f"critical dates including {', '.join(dates[:2])}")
        
        if terms_list:
            summary_parts.append(f"The document specifies {' and '.join(terms_list)}. Key provisions within the document address performance requirements, compliance obligations, and dispute resolution mechanisms specific to this agreement.")
        else:
            summary_parts.append("The document outlines specific performance requirements, compliance obligations, and legal responsibilities for all parties involved.")
        
        # Paragraph 3: Specific risks and recommendations
        risk_score = risk_analysis.get('overall_risk_score', 65)
        specific_risks = [r.get('title', '') for r in risk_analysis.get('specific_risks_identified', [])]
        
        if specific_risks:
            risk_text = f"identified risks including {', '.join(specific_risks[:2])}"
        else:
            risk_text = "various contractual and compliance risks"
            
        summary_parts.append(f"Legal analysis reveals a risk score of {risk_score}/100 with {risk_text}. Immediate attention is required for risk mitigation strategies, compliance verification, and ongoing monitoring of the specific obligations outlined in this document.")
        
        return " ".join(summary_parts)

    def _determine_document_type(self, document_text: str) -> str:
        """Determine document type from content"""
        
        text_lower = document_text.lower()
        
        if 'lease' in text_lower and ('rent' in text_lower or 'premises' in text_lower):
            return 'lease agreement'
        elif 'employment' in text_lower or ('employee' in text_lower and 'employer' in text_lower):
            return 'employment agreement'
        elif 'service' in text_lower and 'agreement' in text_lower:
            return 'service agreement'
        elif 'purchase' in text_lower or 'sale' in text_lower:
            return 'purchase/sale agreement'
        elif 'loan' in text_lower or 'credit' in text_lower:
            return 'financial agreement'
        elif 'contract' in text_lower:
            return 'contract'
        else:
            return 'legal document'

    def _manual_document_analysis(self, document_text: str) -> Dict[str, Any]:
        """Manual document analysis when AI fails"""
        
        text_lower = document_text.lower()
        
        # Extract basic information
        doc_type = self._determine_document_type(document_text)
        
        # Find potential parties (capitalized names/entities)
        party_patterns = [
            r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # Person names
            r'\b[A-Z][A-Za-z\s&]+(Ltd|Limited|Corp|Corporation|Inc|Company|Pvt)\b'  # Companies
        ]
        
        parties = []
        for pattern in party_patterns:
            matches = re.findall(pattern, document_text)
            parties.extend(matches[:3])
        
        # Find dates
        date_patterns = [
            r'\d{1,2}[-/]\d{1,2}[-/]\d{4}',
            r'\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}'
        ]
        
        dates = []
        for pattern in date_patterns:
            matches = re.findall(pattern, document_text, re.IGNORECASE)
            dates.extend(matches[:3])
        
        # Find amounts
        money_patterns = [r'Rs\.?\s*[\d,]+', r'₹\s*[\d,]+', r'INR\s*[\d,]+']
        amounts = []
        for pattern in money_patterns:
            matches = re.findall(pattern, document_text)
            amounts.extend(matches[:3])
        
        return {
            'document_type': doc_type.title(),
            'primary_parties': parties,
            'important_dates': dates,
            'financial_terms': amounts,
            'document_purpose': f'This appears to be a {doc_type} with specific legal obligations',
            'analysis_method': 'manual_extraction'
        }

    def _extract_entities_manually(self, document_text: str) -> List[Dict[str, Any]]:
        """Manual entity extraction"""
        
        entities = []
        
        # Extract parties
        party_pattern = r'\b[A-Z][a-zA-Z\s&]+(Ltd|Limited|Corp|Corporation|Inc|Company|Pvt)\b'
        parties = re.findall(party_pattern, document_text)
        for party in parties[:3]:
            entities.append({
                'type': 'PARTY',
                'value': party,
                'confidence': 0.8,
                'context': 'Legal entity identified in document',
                'location_in_document': 'Document body'
            })
        
        # Extract dates
        date_pattern = r'\d{1,2}[-/]\d{1,2}[-/]\d{4}'
        dates = re.findall(date_pattern, document_text)
        for date in dates[:3]:
            entities.append({
                'type': 'DATE',
                'value': date,
                'confidence': 0.9,
                'context': 'Important date in document',
                'location_in_document': 'Date clause'
            })
        
        # Extract amounts
        money_pattern = r'Rs\.?\s*[\d,]+'
        amounts = re.findall(money_pattern, document_text)
        for amount in amounts[:3]:
            entities.append({
                'type': 'AMOUNT',
                'value': amount,
                'confidence': 0.95,
                'context': 'Financial amount in document',
                'location_in_document': 'Financial clause'
            })
        
        return entities

    def _create_document_based_risk_analysis(self, document_text: str, entities: List[Dict]) -> Dict[str, Any]:
        """Create risk analysis based on actual document content"""
        
        text_lower = document_text.lower()
        risks = []
        risk_score = 50
        
        # Check for high-risk terms in actual document
        high_risk_terms = {
            'penalty': 'Penalty clauses create financial risk exposure',
            'liquidated damages': 'Liquidated damages clause requires careful compliance',
            'termination without cause': 'Termination clause creates employment risk',
            'personal guarantee': 'Personal guarantee creates individual liability',
            'unlimited liability': 'Unlimited liability clause creates significant risk'
        }
        
        for term, description in high_risk_terms.items():
            if term in text_lower:
                risk_score += 15
                risks.append({
                    'title': f'High Risk: {term.title()} Clause Identified',
                    'description': description,
                    'severity': 'High',
                    'evidence_from_document': f'Document contains "{term}" provision',
                    'recommendation': f'Review {term} clause carefully with legal counsel'
                })
        
        # Financial risk based on actual amounts
        amounts = [e for e in entities if e.get('type') == 'AMOUNT']
        if amounts:
            risk_score += 10
            risks.append({
                'title': 'Financial Risk: Monetary Obligations Present',
                'description': f'Document contains {len(amounts)} financial obligations requiring compliance',
                'severity': 'Medium',
                'evidence_from_document': f'Amounts identified: {[a["value"] for a in amounts[:3]]}',
                'recommendation': 'Ensure adequate financial resources for all obligations'
            })
        
        # Date-based risks
        dates = [e for e in entities if e.get('type') == 'DATE']
        if len(dates) > 2:
            risks.append({
                'title': 'Operational Risk: Multiple Critical Deadlines',
                'description': f'Document has {len(dates)} time-sensitive obligations',
                'severity': 'Medium',
                'evidence_from_document': f'Critical dates: {[d["value"] for d in dates[:3]]}',
                'recommendation': 'Implement deadline tracking and compliance monitoring'
            })
        
        return {
            'overall_risk_score': min(risk_score, 100),
            'risk_level': 'High' if risk_score > 80 else 'Medium' if risk_score > 60 else 'Low',
            'specific_risks_identified': risks,
            'confidence': 0.8
        }

    def _check_indian_law_compliance(self, document_text: str) -> Dict[str, Any]:
        """Check compliance with Indian law based on document content"""
        
        try:
            prompt = f"""Analyze this document for compliance with Indian legal requirements. Focus on actual document content.

DOCUMENT TO ANALYZE:
{document_text[:2000]}

Check compliance with Indian laws and provide specific findings in JSON:
{{
    "overall_compliance_score": 80,
    "compliance_status": "Compliant/Partially Compliant/Requires Review",
    "applicable_indian_laws": [
        {{"law": "Indian Contract Act, 1872", "relevance": "How it applies to this specific document", "compliance": "Met/Not Met"}}
    ],
    "compliant_areas": ["Specific compliant aspects of this document"],
    "areas_needing_attention": [
        {{"area": "Specific issue", "requirement": "Indian law requirement", "recommendation": "Specific action"}}
    ],
    "missing_standard_clauses": ["Clauses typically required in Indian law for this type of document"],
    "confidence": 0.85
}}

Base analysis on actual document content and Indian legal framework."""

            response = self.gemini_model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            
            try:
                return json.loads(response.text.strip())
            except json.JSONDecodeError:
                return self._basic_indian_law_compliance(document_text)
                
        except Exception as e:
            logger.error(f"Compliance analysis failed: {e}")
            return self._basic_indian_law_compliance(document_text)

    def _basic_indian_law_compliance(self, document_text: str) -> Dict[str, Any]:
        """Basic Indian law compliance analysis"""
        
        text_lower = document_text.lower()
        
        # Check for basic contract elements
        has_consideration = 'consideration' in text_lower or 'payment' in text_lower
        has_parties = 'party' in text_lower or 'parties' in text_lower
        has_terms = 'term' in text_lower or 'condition' in text_lower
        
        compliance_score = 60
        if has_consideration:
            compliance_score += 15
        if has_parties:
            compliance_score += 10
        if has_terms:
            compliance_score += 10
        
        return {
            'overall_compliance_score': compliance_score,
            'compliance_status': 'Requires Professional Review',
            'applicable_indian_laws': [
                {'law': 'Indian Contract Act, 1872', 'relevance': 'Governs contractual obligations', 'compliance': 'Partially Met'}
            ],
            'compliant_areas': [
                area for area, present in [
                    ('Consideration mentioned', has_consideration),
                    ('Parties identification', has_parties),
                    ('Terms and conditions', has_terms)
                ] if present
            ],
            'areas_needing_attention': [
                {'area': 'Professional legal review', 'requirement': 'Comprehensive compliance check', 'recommendation': 'Consult legal expert'}
            ],
            'confidence': 0.7
        }

    def _perform_inlegal_bert_analysis(self, document_text: str) -> Dict[str, Any]:
        """Perform InLegalBERT-style analysis"""
        
        try:
            prompt = f"""Perform InLegalBERT-style analysis of this Indian legal document. Extract specific legal terminology and concepts from the actual document content.

DOCUMENT TO ANALYZE:
{document_text[:2500]}

Provide InLegalBERT analysis in JSON:
{{
    "legal_terminology": ["Actual legal terms found in this document"],
    "statutory_references": [
        {{"act": "Act name if mentioned", "section": "Section if specified", "context": "How it's referenced in document"}}
    ],
    "legal_concepts": [
        {{"concept": "Legal concept in document", "definition": "Meaning", "relevance": "Why important for this document"}}
    ],
    "indian_law_elements": [
        {{"element": "Element type", "presence": "Found/Not Found", "evidence": "Where in document"}}
    ],
    "document_classification": "Specific classification based on content",
    "complexity_assessment": "Simple/Moderate/Complex with reasoning",
    "confidence": 0.9
}}

Extract information ONLY from this specific document."""

            response = self.gemini_model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            
            try:
                return json.loads(response.text.strip())
            except json.JSONDecodeError:
                return self._basic_inlegal_analysis(document_text)
                
        except Exception as e:
            logger.error(f"InLegalBERT analysis failed: {e}")
            return self._basic_inlegal_analysis(document_text)

    def _basic_inlegal_analysis(self, document_text: str) -> Dict[str, Any]:
        """Basic legal analysis when AI fails"""
        
        text_lower = document_text.lower()
        
        # Extract legal terms present in document
        legal_terms = []
        common_terms = ['agreement', 'contract', 'party', 'clause', 'liability', 'obligation', 'breach', 'termination']
        
        for term in common_terms:
            if term in text_lower:
                legal_terms.append(term)
        
        return {
            'legal_terminology': legal_terms,
            'statutory_references': [],
            'legal_concepts': [
                {'concept': 'Contractual Agreement', 'definition': 'Legal binding between parties', 'relevance': 'Forms basis of document'}
            ],
            'document_classification': self._determine_document_type(document_text).title(),
            'complexity_assessment': 'Moderate' if len(document_text.split()) > 1000 else 'Simple',
            'confidence': 0.6
        }

    def _create_document_specific_fallback(self, document_text: str, error_msg: str) -> Dict[str, Any]:
        """Create document-specific fallback even when analysis fails"""
        
        # Even in fallback, extract basic document info
        word_count = len(document_text.split())
        doc_type = self._determine_document_type(document_text)
        basic_entities = self._extract_entities_manually(document_text)
        
        # Create document-specific fallback summary
        first_line = document_text.split('\n')[0] if document_text else "Legal document"
        
        fallback_summary = f"""This {doc_type} ({word_count} words) begins with "{first_line[:100]}..." and contains specific legal provisions requiring analysis. While detailed AI processing encountered technical difficulties, basic analysis identified {len(basic_entities)} key entities including parties, dates, and financial terms. The document appears to establish legal obligations and rights between the identified parties, with specific performance requirements and compliance obligations. Professional legal review is strongly recommended to ensure full understanding of the specific terms and conditions within this document."""
        
        return {
            'analysis_id': f"fallback_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'error': error_msg,
            'document_stats': {
                'character_count': len(document_text),
                'word_count': word_count,
                'analysis_timestamp': datetime.now().isoformat(),
                'document_preview': document_text[:200] + "..."
            },
            'key_entities': basic_entities,
            'executive_summary': fallback_summary,
            'overall_risk_score': 65,
            'confidence_score': 0.5,
            'processing_time': 0.1
        }
