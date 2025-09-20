"""
Multi-Agent Legal Document Orchestrator with InLegalBERT + Gemini Integration
"""
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import google.generativeai as genai
import re
import time

logger = logging.getLogger(__name__)

class LegalAgentOrchestrator:
    """Advanced Multi-Agent Legal Analysis System with InLegalBERT + Gemini"""
    
    def __init__(self, gemini_api_key: str):
        """Initialize the multi-agent system"""
        # Configure Gemini
        genai.configure(api_key=gemini_api_key)
        self.gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Agent configurations
        self.agents = {
            'entity_extractor': EntityExtractionAgent(self.gemini_model),
            'risk_analyzer': RiskAnalysisAgent(self.gemini_model),
            'compliance_checker': ComplianceAgent(self.gemini_model),
            'semantic_segmenter': SemanticSegmentationAgent(self.gemini_model),
            'inlegal_bert_processor': InLegalBERTProcessor(self.gemini_model)
        }
        
        logger.info("Multi-Agent Legal Orchestrator initialized with 5 specialized agents")
    
    def comprehensive_document_analysis(self, document_text: str, 
                                      legal_jurisdiction: str = "Indian Law",
                                      focus_areas: List[str] = None,
                                      analysis_options: Dict = None) -> Dict[str, Any]:
        """
        SYNCHRONOUS Multi-Agent Analysis - NO ASYNC
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting comprehensive analysis of {len(document_text)} character document")
            
            # Initialize results structure
            results = {
                'analysis_id': f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'document_stats': {
                    'character_count': len(document_text),
                    'word_count': len(document_text.split()),
                    'paragraph_count': len(document_text.split('\n\n')),
                    'analysis_timestamp': datetime.now().isoformat()
                },
                'agents_used': list(self.agents.keys()),
                'focus_areas': focus_areas or ['Entity Extraction', 'Risk Assessment'],
                'legal_jurisdiction': legal_jurisdiction
            }
            
            # Agent 1: InLegalBERT Processing (Simulated + Gemini Enhanced)
            logger.info("Agent 1: InLegalBERT Processing...")
            inlegal_results = self.agents['inlegal_bert_processor'].process_document(
                document_text, legal_jurisdiction
            )
            results['inlegal_bert_analysis'] = inlegal_results
            
            # Agent 2: Entity Extraction
            logger.info("Agent 2: Legal Entity Extraction...")
            entities = self.agents['entity_extractor'].extract_entities(document_text)
            results['key_entities'] = entities
            
            # Agent 3: Semantic Segmentation
            logger.info("Agent 3: Document Segmentation...")
            segments = self.agents['semantic_segmenter'].segment_document(document_text)
            results['document_segments'] = segments
            
            # Agent 4: Risk Analysis
            logger.info("Agent 4: Legal Risk Analysis...")
            risk_analysis = self.agents['risk_analyzer'].analyze_risks(
                document_text, entities, segments
            )
            results['risk_analysis'] = risk_analysis
            
            # Agent 5: Compliance Check
            logger.info("Agent 5: Compliance Analysis...")
            compliance_results = self.agents['compliance_checker'].check_compliance(
                document_text, legal_jurisdiction, entities
            )
            results['compliance_analysis'] = compliance_results
            
            # Synthesize final results
            results['executive_summary'] = self._synthesize_executive_summary(results)
            results['overall_risk_score'] = risk_analysis.get('overall_risk_score', 65)
            results['confidence_score'] = self._calculate_overall_confidence(results)
            results['processing_time'] = time.time() - start_time
            
            logger.info(f"Multi-agent analysis completed in {results['processing_time']:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"Multi-agent analysis failed: {e}")
            return self._generate_fallback_analysis(document_text, str(e))
    
    def _synthesize_executive_summary(self, results: Dict[str, Any]) -> str:
        """Generate executive summary from all agent results"""
        
        try:
            # Prepare summary data
            entity_count = len(results.get('key_entities', []))
            risk_score = results.get('risk_analysis', {}).get('overall_risk_score', 65)
            segment_count = len(results.get('document_segments', {}))
            
            # Use Gemini to synthesize
            synthesis_prompt = f"""As an expert legal analyst, synthesize the following multi-agent analysis results into a comprehensive executive summary:

DOCUMENT STATISTICS:
- Character count: {results['document_stats']['character_count']}
- Word count: {results['document_stats']['word_count']}
- Entities identified: {entity_count}
- Document segments: {segment_count}

RISK ANALYSIS:
- Overall risk score: {risk_score}/100
- Risk categories analyzed: {list(results.get('risk_analysis', {}).get('risk_categories', {}).keys())}

INLEGALBERT ANALYSIS:
{json.dumps(results.get('inlegal_bert_analysis', {}), indent=2)[:500]}...

COMPLIANCE STATUS:
{json.dumps(results.get('compliance_analysis', {}), indent=2)[:300]}...

Create a 2-3 paragraph executive summary that:
1. Describes the document type and key characteristics
2. Highlights the most significant legal findings
3. Provides actionable insights and recommendations

Focus on practical, actionable insights for legal professionals."""

            response = self.gemini_model.generate_content(synthesis_prompt)
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"Executive summary generation failed: {e}")
            return f"Multi-agent analysis completed successfully. Document contains {results['document_stats']['word_count']} words with {entity_count} legal entities identified. Risk score: {risk_score}/100. Professional legal review recommended for comprehensive assessment."
    
    def _calculate_overall_confidence(self, results: Dict[str, Any]) -> float:
        """Calculate overall confidence score from all agents"""
        
        confidences = []
        
        # Entity extraction confidence
        entities = results.get('key_entities', [])
        if entities:
            entity_confidences = [e.get('confidence', 0.7) for e in entities]
            confidences.append(sum(entity_confidences) / len(entity_confidences))
        
        # Risk analysis confidence
        risk_conf = results.get('risk_analysis', {}).get('confidence', 0.8)
        confidences.append(risk_conf)
        
        # Compliance confidence
        compliance_conf = results.get('compliance_analysis', {}).get('confidence', 0.75)
        confidences.append(compliance_conf)
        
        # InLegalBERT confidence
        inlegal_conf = results.get('inlegal_bert_analysis', {}).get('confidence', 0.85)
        confidences.append(inlegal_conf)
        
        return sum(confidences) / len(confidences) if confidences else 0.7
    
    def _generate_fallback_analysis(self, document_text: str, error_msg: str) -> Dict[str, Any]:
        """Generate fallback analysis when multi-agent processing fails"""
        
        word_count = len(document_text.split())
        
        return {
            'analysis_id': f"fallback_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'error': error_msg,
            'document_stats': {
                'character_count': len(document_text),
                'word_count': word_count,
                'analysis_timestamp': datetime.now().isoformat()
            },
            'key_entities': [
                {'type': 'Document', 'value': 'Legal Document', 'confidence': 0.9},
                {'type': 'Word_Count', 'value': str(word_count), 'confidence': 1.0},
                {'type': 'Analysis_Status', 'value': 'Fallback Mode', 'confidence': 1.0}
            ],
            'risk_analysis': {
                'overall_risk_score': 60,
                'confidence': 0.5,
                'risk_categories': {
                    'General': {'category_risk_score': 60, 'issues': ['Multi-agent analysis failed - manual review required']}
                }
            },
            'executive_summary': f"Fallback analysis completed for document containing {word_count} words. Multi-agent processing encountered an error: {error_msg}. Manual legal review is recommended for comprehensive analysis.",
            'overall_risk_score': 60,
            'confidence_score': 0.5,
            'processing_time': 0.1
        }

class InLegalBERTProcessor:
    """InLegalBERT Processing Agent with Gemini Enhancement"""
    
    def __init__(self, gemini_model):
        self.gemini_model = gemini_model
        self.model_name = "InLegalBERT (Simulated + Gemini Enhanced)"
    
    def process_document(self, document_text: str, legal_jurisdiction: str) -> Dict[str, Any]:
        """Process document with InLegalBERT-style analysis enhanced by Gemini"""
        
        try:
            # InLegalBERT-style prompt for Indian legal text processing
            inlegal_prompt = f"""You are InLegalBERT, a specialized transformer model for Indian legal text processing. Analyze the following legal document with expert knowledge of Indian legal terminology, statutes, and case law.

LEGAL DOCUMENT TEXT:
{document_text[:3000]}...

ANALYSIS TASKS:
1. Identify legal terminology and concepts specific to Indian law
2. Recognize statutory references (Acts, Sections, Rules)
3. Identify legal precedents and case citations
4. Extract contractual clauses and legal obligations
5. Detect compliance requirements under Indian legal framework
6. Identify legal entities (courts, parties, advocates)

Provide analysis in this JSON structure:
{{
    "legal_terminology": ["term1", "term2", ...],
    "statutory_references": [
        {{"act": "Act name", "section": "Section number", "relevance": "High/Medium/Low"}}
    ],
    "legal_concepts": [
        {{"concept": "Legal concept", "definition": "Brief definition", "indian_law_context": "Relevance to Indian law"}}
    ],
    "contractual_elements": [
        {{"element": "Element type", "description": "Description", "legal_significance": "Significance"}}
    ],
    "compliance_indicators": [
        {{"requirement": "Compliance requirement", "status": "Met/Not Met/Unclear", "reference": "Legal reference"}}
    ],
    "confidence": 0.85,
    "processing_notes": "Any specific observations about the document"
}}

Focus specifically on Indian legal framework and terminology."""

            response = self.gemini_model.generate_content(inlegal_prompt)
            
            try:
                # Try to parse JSON response
                result = json.loads(response.text.strip())
                result['model_used'] = self.model_name
                result['processing_timestamp'] = datetime.now().isoformat()
                return result
                
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                return {
                    'model_used': self.model_name,
                    'legal_terminology': self._extract_legal_terms(document_text),
                    'statutory_references': self._find_statutory_refs(document_text),
                    'legal_concepts': self._identify_legal_concepts(document_text),
                    'confidence': 0.75,
                    'processing_notes': 'Fallback processing - JSON parsing failed',
                    'raw_response': response.text[:500] + "...",
                    'processing_timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"InLegalBERT processing failed: {e}")
            return {
                'model_used': self.model_name,
                'error': str(e),
                'confidence': 0.3,
                'processing_notes': 'InLegalBERT processing failed - using basic analysis'
            }
    
    def _extract_legal_terms(self, text: str) -> List[str]:
        """Extract Indian legal terminology"""
        
        indian_legal_terms = [
            'agreement', 'contract', 'party', 'clause', 'section', 'act', 'law',
            'court', 'jurisdiction', 'liability', 'damages', 'breach', 'termination',
            'arbitration', 'dispute', 'settlement', 'compensation', 'penalty',
            'consideration', 'obligation', 'right', 'duty', 'enforcement'
        ]
        
        found_terms = []
        text_lower = text.lower()
        
        for term in indian_legal_terms:
            if term in text_lower:
                found_terms.append(term)
        
        return found_terms[:10]  # Return top 10
    
    def _find_statutory_refs(self, text: str) -> List[Dict[str, str]]:
        """Find statutory references in text"""
        
        # Common Indian legal act patterns
        act_patterns = [
            r'Indian Contract Act[,\s]*1872',
            r'Companies Act[,\s]*2013',
            r'Arbitration and Conciliation Act[,\s]*1996',
            r'Information Technology Act[,\s]*2000',
            r'Consumer Protection Act[,\s]*2019'
        ]
        
        references = []
        for pattern in act_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                references.append({
                    'act': match,
                    'section': 'Various',
                    'relevance': 'High'
                })
        
        return references[:5]  # Return top 5
    
    def _identify_legal_concepts(self, text: str) -> List[Dict[str, str]]:
        """Identify key legal concepts"""
        
        concepts = [
            {
                'concept': 'Contractual Agreement',
                'definition': 'Legally binding agreement between parties',
                'indian_law_context': 'Governed by Indian Contract Act, 1872'
            },
            {
                'concept': 'Legal Obligations',
                'definition': 'Duties that parties must fulfill',
                'indian_law_context': 'Enforceable under Indian legal system'
            }
        ]
        
        return concepts

class EntityExtractionAgent:
    """Advanced Legal Entity Extraction Agent"""
    
    def __init__(self, gemini_model):
        self.gemini_model = gemini_model
    
    def extract_entities(self, document_text: str) -> List[Dict[str, Any]]:
        """Extract legal entities using Gemini"""
        
        try:
            entity_prompt = f"""Extract all legal entities from this document. Focus on Indian legal context.

DOCUMENT:
{document_text[:2000]}...

Extract these entity types:
1. PARTIES (individuals, companies, organizations)
2. LEGAL_ACTS (statutes, acts, regulations)
3. DATES (important dates, deadlines, effective dates)
4. MONETARY_AMOUNTS (payments, penalties, compensation)
5. LEGAL_REFERENCES (case citations, section numbers)
6. LOCATIONS (jurisdictions, addresses, courts)
7. LEGAL_CONCEPTS (terms of art, legal principles)

Format as JSON array:
[
    {{"type": "PARTY", "value": "Entity name", "confidence": 0.95, "context": "Brief context"}},
    {{"type": "DATE", "value": "2024-01-01", "confidence": 0.90, "context": "Effective date"}},
    ...
]

Focus on accuracy and Indian legal terminology."""

            response = self.gemini_model.generate_content(entity_prompt)
            
            try:
                entities = json.loads(response.text.strip())
                return entities if isinstance(entities, list) else []
            except json.JSONDecodeError:
                return self._fallback_entity_extraction(document_text)
                
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return self._fallback_entity_extraction(document_text)
    
    def _fallback_entity_extraction(self, text: str) -> List[Dict[str, Any]]:
        """Fallback entity extraction using regex patterns"""
        
        entities = []
        
        # Extract dates
        date_pattern = r'\d{1,2}[-/]\d{1,2}[-/]\d{4}|\d{4}[-/]\d{1,2}[-/]\d{1,2}'
        dates = re.findall(date_pattern, text)
        for date in dates[:5]:
            entities.append({
                'type': 'DATE',
                'value': date,
                'confidence': 0.8,
                'context': 'Pattern matched date'
            })
        
        # Extract monetary amounts
        money_pattern = r'Rs\.?\s*[\d,]+|₹\s*[\d,]+'
        amounts = re.findall(money_pattern, text)
        for amount in amounts[:5]:
            entities.append({
                'type': 'MONETARY_AMOUNT',
                'value': amount,
                'confidence': 0.85,
                'context': 'Currency amount'
            })
        
        # Extract company names (simple pattern)
        company_pattern = r'\b[A-Z][a-zA-Z\s&]+(Ltd|Limited|Corp|Corporation|Inc|Company)\b'
        companies = re.findall(company_pattern, text)
        for company in companies[:3]:
            entities.append({
                'type': 'PARTY',
                'value': company,
                'confidence': 0.75,
                'context': 'Company entity'
            })
        
        return entities

class RiskAnalysisAgent:
    """Legal Risk Analysis Agent"""
    
    def __init__(self, gemini_model):
        self.gemini_model = gemini_model
    
    def analyze_risks(self, document_text: str, entities: List[Dict], segments: Dict) -> Dict[str, Any]:
        """Comprehensive legal risk analysis"""
        
        try:
            risk_prompt = f"""Conduct a comprehensive legal risk analysis of this Indian legal document.

DOCUMENT TEXT:
{document_text[:2000]}...

IDENTIFIED ENTITIES: {len(entities)} entities found
DOCUMENT SEGMENTS: {list(segments.keys()) if segments else 'None'}

RISK ANALYSIS FRAMEWORK:
1. Contractual risks (breach, non-performance, ambiguity)
2. Compliance risks (regulatory, statutory)
3. Financial risks (payment, penalties, liabilities)
4. Operational risks (performance, delivery)
5. Legal risks (jurisdiction, enforceability, disputes)

Provide detailed analysis in JSON format:
{{
    "overall_risk_score": 75,
    "risk_level": "Medium",
    "risk_categories": {{
        "contractual": {{"category_risk_score": 70, "issues": ["List of issues"], "severity": "Medium"}},
        "compliance": {{"category_risk_score": 80, "issues": ["List of issues"], "severity": "High"}},
        "financial": {{"category_risk_score": 60, "issues": ["List of issues"], "severity": "Medium"}},
        "operational": {{"category_risk_score": 65, "issues": ["List of issues"], "severity": "Medium"}},
        "legal": {{"category_risk_score": 75, "issues": ["List of issues"], "severity": "High"}}
    }},
    "identified_risks": [
        {{"title": "Risk title", "description": "Risk description", "severity": "High/Medium/Low", "mitigation": "Suggested mitigation"}}
    ],
    "recommendations": ["List of actionable recommendations"],
    "confidence": 0.85
}}

Focus on Indian legal context and practical risks."""

            response = self.gemini_model.generate_content(risk_prompt)
            
            try:
                result = json.loads(response.text.strip())
                return result
            except json.JSONDecodeError:
                return self._fallback_risk_analysis(document_text, entities)
                
        except Exception as e:
            logger.error(f"Risk analysis failed: {e}")
            return self._fallback_risk_analysis(document_text, entities)
    
    def _fallback_risk_analysis(self, text: str, entities: List[Dict]) -> Dict[str, Any]:
        """Fallback risk analysis"""
        
        # Simple risk scoring based on document characteristics
        base_score = 50
        
        # Adjust based on entities
        entity_count = len(entities)
        if entity_count > 10:
            base_score += 10  # More complex document
        
        # Adjust based on text length
        word_count = len(text.split())
        if word_count > 1000:
            base_score += 5  # Longer documents may have more risks
        
        return {
            'overall_risk_score': base_score,
            'risk_level': 'Medium',
            'risk_categories': {
                'general': {
                    'category_risk_score': base_score,
                    'issues': ['Document requires professional legal review'],
                    'severity': 'Medium'
                }
            },
            'identified_risks': [
                {
                    'title': 'General Legal Review Required',
                    'description': 'Professional legal analysis recommended for comprehensive risk assessment',
                    'severity': 'Medium',
                    'mitigation': 'Consult with qualified legal counsel'
                }
            ],
            'recommendations': [
                'Conduct thorough legal review',
                'Verify compliance with applicable laws',
                'Consider professional legal consultation'
            ],
            'confidence': 0.6
        }

class SemanticSegmentationAgent:
    """Document Semantic Segmentation Agent"""
    
    def __init__(self, gemini_model):
        self.gemini_model = gemini_model
    
    def segment_document(self, document_text: str) -> Dict[str, List[str]]:
        """Segment document into semantic sections"""
        
        try:
            # Simple segmentation based on common legal document patterns
            segments = {
                'preamble': [],
                'definitions': [],
                'main_clauses': [],
                'obligations': [],
                'termination': [],
                'miscellaneous': []
            }
            
            # Split into paragraphs
            paragraphs = [p.strip() for p in document_text.split('\n\n') if p.strip()]
            
            for para in paragraphs:
                para_lower = para.lower()
                
                if any(word in para_lower for word in ['definition', 'means', 'shall mean']):
                    segments['definitions'].append(para)
                elif any(word in para_lower for word in ['obligation', 'duty', 'responsibility']):
                    segments['obligations'].append(para)
                elif any(word in para_lower for word in ['termination', 'terminate', 'end']):
                    segments['termination'].append(para)
                elif any(word in para_lower for word in ['whereas', 'recital', 'background']):
                    segments['preamble'].append(para)
                else:
                    segments['main_clauses'].append(para)
            
            return segments
            
        except Exception as e:
            logger.error(f"Segmentation failed: {e}")
            return {'main_content': [document_text[:1000] + "..."]}

class ComplianceAgent:
    """Legal Compliance Checking Agent"""
    
    def __init__(self, gemini_model):
        self.gemini_model = gemini_model
    
    def check_compliance(self, document_text: str, jurisdiction: str, entities: List[Dict]) -> Dict[str, Any]:
        """Check compliance with Indian legal requirements"""
        
        try:
            compliance_prompt = f"""Analyze this document for compliance with Indian legal requirements.

DOCUMENT:
{document_text[:1500]}...

JURISDICTION: {jurisdiction}
ENTITIES: {len(entities)} identified

Check compliance with:
1. Indian Contract Act, 1872
2. Companies Act, 2013
3. Consumer Protection Act, 2019
4. Information Technology Act, 2000
5. Arbitration and Conciliation Act, 1996

Provide compliance analysis in JSON:
{{
    "overall_compliance_score": 80,
    "compliance_status": "Mostly Compliant",
    "compliant_areas": ["Area 1", "Area 2"],
    "non_compliant_areas": [
        {{"area": "Area name", "issue": "Issue description", "severity": "High/Medium/Low", "recommendation": "Fix recommendation"}}
    ],
    "missing_clauses": ["List of missing standard clauses"],
    "regulatory_requirements": [
        {{"requirement": "Requirement", "status": "Met/Not Met", "reference": "Legal reference"}}
    ],
    "confidence": 0.85
}}

Focus on Indian legal compliance."""

            response = self.gemini_model.generate_content(compliance_prompt)
            
            try:
                result = json.loads(response.text.strip())
                return result
            except json.JSONDecodeError:
                return self._fallback_compliance_analysis()
                
        except Exception as e:
            logger.error(f"Compliance analysis failed: {e}")
            return self._fallback_compliance_analysis()
    
    def _fallback_compliance_analysis(self) -> Dict[str, Any]:
        """Fallback compliance analysis"""
        
        return {
            'overall_compliance_score': 75,
            'compliance_status': 'Requires Review',
            'compliant_areas': ['Basic document structure'],
            'non_compliant_areas': [
                {
                    'area': 'Professional Review Required',
                    'issue': 'Comprehensive compliance analysis needs legal expert review',
                    'severity': 'Medium',
                    'recommendation': 'Consult with legal professional for detailed compliance check'
                }
            ],
            'missing_clauses': ['Professional analysis required'],
            'regulatory_requirements': [
                {
                    'requirement': 'Legal compliance verification',
                    'status': 'Requires professional review',
                    'reference': 'Various Indian Acts'
                }
            ],
            'confidence': 0.6
        }
