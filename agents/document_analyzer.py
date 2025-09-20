import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np
import re

class DocumentAnalysisAgent:
    """Specialized agent for document structure analysis using InLegalBERT"""
    
    def __init__(self, inlegal_bert_wrapper):
        self.inlegal_bert = inlegal_bert_wrapper
        self.analysis_confidence_threshold = 0.6
        
        # Indian legal document patterns
        self.indian_document_patterns = {
            'contract_indicators': [
                'agreement', 'contract', 'memorandum of understanding', 'mou',
                'terms and conditions', 'service agreement', 'license agreement'
            ],
            'corporate_indicators': [
                'board resolution', 'shareholder agreement', 'articles of association',
                'memorandum of association', 'annual report', 'companies act'
            ],
            'employment_indicators': [
                'employment agreement', 'appointment letter', 'service contract',
                'non-disclosure agreement', 'non-compete', 'salary', 'designation'
            ],
            'judgment_indicators': [
                'judgment', 'order', 'writ petition', 'civil appeal', 'criminal appeal',
                'high court', 'supreme court', 'district court', 'sessions court'
            ],
            'regulatory_indicators': [
                'notification', 'circular', 'guidelines', 'regulations',
                'sebi', 'rbi', 'mca', 'compliance', 'statutory'
            ]
        }
        
        # Indian legal entities patterns
        self.indian_entity_patterns = {
            'court_references': r'(Supreme Court|High Court|District Court|Sessions Court|Magistrate Court)',
            'legal_citations': r'(AIR\s+\d+|SCC\s+\d+|\d+\s+SCC\s+\d+)',
            'act_references': r'(Companies Act,?\s*\d+|Contract Act,?\s*\d+|Evidence Act,?\s*\d+)',
            'section_references': r'Section\s+\d+[A-Z]?',
            'indian_currency': r'Rs\.?\s*[\d,]+(?:\.\d{2})?',
            'corporate_entities': r'[A-Z][a-zA-Z\s]+(?:Limited|Ltd\.?|Private Limited|Pvt\.?\s*Ltd\.?)',
            'dates': r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}'
        }
    
    async def analyze_document_structure(self, document_text: str, jurisdiction: str) -> Dict[str, Any]:
        """Comprehensive document structure analysis"""
        
        analysis_result = {
            'analysis_timestamp': datetime.now().isoformat(),
            'jurisdiction': jurisdiction,
            'document_length': len(document_text),
            'document_type': 'Unknown',
            'legal_domain': 'general',
            'confidence': 0.0,
            'key_entities': [],
            'segments': {},
            'metadata': {}
        }
        
        try:
            # Phase 1: Document Classification
            classification_result = await self._classify_document_type(document_text)
            analysis_result.update(classification_result)
            
            # Phase 2: Semantic Segmentation using InLegalBERT
            if self.inlegal_bert and self.inlegal_bert.initialized:
                segmentation_result = self.inlegal_bert.semantic_segmentation(document_text)
                analysis_result['segments'] = segmentation_result['segments']
                analysis_result['segmentation_confidence'] = segmentation_result['segmentation_confidence']
            else:
                analysis_result['segments'] = self._fallback_segmentation(document_text)
                analysis_result['segmentation_confidence'] = 0.5
            
            # Phase 3: Legal Entity Extraction
            entities = self._extract_indian_legal_entities(document_text)
            analysis_result['key_entities'] = entities
            
            # Phase 4: Statute Identification using InLegalBERT
            if self.inlegal_bert and self.inlegal_bert.initialized:
                relevant_statutes = self.inlegal_bert.identify_statutes(document_text)
                analysis_result['relevant_statutes'] = relevant_statutes
            else:
                analysis_result['relevant_statutes'] = self._fallback_statute_identification(document_text)
            
            # Phase 5: Document Quality Assessment
            quality_metrics = self._assess_document_quality(document_text, analysis_result)
            analysis_result['quality_metrics'] = quality_metrics
            
            # Phase 6: Generate Document Fingerprint
            document_fingerprint = self._generate_document_fingerprint(analysis_result)
            analysis_result['document_fingerprint'] = document_fingerprint
            
            # Update overall confidence
            analysis_result['confidence'] = self._calculate_analysis_confidence(analysis_result)
            
        except Exception as e:
            analysis_result['error'] = str(e)
            analysis_result['analysis_status'] = 'failed'
        
        return analysis_result
    
    async def _classify_document_type(self, document_text: str) -> Dict[str, Any]:
        """Classify document type based on content patterns"""
        
        text_lower = document_text.lower()
        classification_scores = {}
        
        # Score each document type
        for doc_type, indicators in self.indian_document_patterns.items():
            score = sum(1 for indicator in indicators if indicator in text_lower)
            if score > 0:
                classification_scores[doc_type] = score / len(indicators)
        
        if classification_scores:
            # Get best match
            best_type = max(classification_scores, key=classification_scores.get)
            confidence = classification_scores[best_type]
            
            # Map to human-readable types
            type_mapping = {
                'contract_indicators': ('Contract/Agreement', 'contract'),
                'corporate_indicators': ('Corporate Document', 'corporate'),
                'employment_indicators': ('Employment Document', 'employment'),
                'judgment_indicators': ('Court Judgment/Order', 'legal_judgment'),
                'regulatory_indicators': ('Regulatory Document', 'regulatory')
            }
            
            document_type, legal_domain = type_mapping.get(best_type, ('Legal Document', 'general'))
        else:
            document_type, legal_domain = 'Legal Document', 'general'
            confidence = 0.5
        
        return {
            'document_type': document_type,
            'legal_domain': legal_domain,
            'classification_confidence': confidence,
            'classification_scores': classification_scores
        }
    
    def _extract_indian_legal_entities(self, document_text: str) -> List[Dict[str, Any]]:
        """Extract Indian legal entities using pattern matching"""
        
        entities = []
        
        for entity_type, pattern in self.indian_entity_patterns.items():
            matches = re.findall(pattern, document_text, re.IGNORECASE)
            
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match[0] else match[1]
                
                entity = {
                    'type': entity_type,
                    'value': match.strip(),
                    'confidence': 0.8 if entity_type in ['court_references', 'act_references'] else 0.7
                }
                
                # Avoid duplicates
                if entity not in entities:
                    entities.append(entity)
        
        # Sort by confidence and limit results
        entities.sort(key=lambda x: x['confidence'], reverse=True)
        return entities[:15]  # Top 15 entities
    
    def _fallback_segmentation(self, document_text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Fallback segmentation when InLegalBERT is not available"""
        
        paragraphs = document_text.split('\n\n')
        segments = {'GENERAL': []}
        
        for i, paragraph in enumerate(paragraphs[:10]):  # Limit to 10 paragraphs
            if len(paragraph.strip()) > 20:  # Skip very short paragraphs
                segments['GENERAL'].append({
                    'text': paragraph.strip(),
                    'position': i,
                    'confidence': 0.5
                })
        
        return segments
    
    def _fallback_statute_identification(self, document_text: str) -> List[Dict[str, Any]]:
        """Fallback statute identification using pattern matching"""
        
        text_lower = document_text.lower()
        statutes = []
        
        # Common Indian acts
        indian_acts = [
            ('Companies Act 2013', ['company', 'companies act', 'director', 'shareholder']),
            ('Contract Act 1872', ['contract', 'agreement', 'breach', 'consideration']),
            ('IT Act 2000', ['computer', 'electronic', 'data', 'cyber', 'digital']),
            ('Evidence Act 1872', ['evidence', 'proof', 'witness', 'testimony']),
            ('Civil Procedure Code 1908', ['civil', 'procedure', 'suit', 'decree'])
        ]
        
        for act_name, keywords in indian_acts:
            relevance_score = sum(1 for keyword in keywords if keyword in text_lower) / len(keywords)
            
            if relevance_score > 0.2:  # Minimum threshold
                statutes.append({
                    'statute': act_name,
                    'relevance_score': relevance_score,
                    'keywords_matched': [kw for kw in keywords if kw in text_lower]
                })
        
        return sorted(statutes, key=lambda x: x['relevance_score'], reverse=True)
    
    def _assess_document_quality(self, document_text: str, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Assess document quality and completeness"""
        
        quality_metrics = {
            'readability_score': 0.0,
            'completeness_score': 0.0,
            'legal_terminology_density': 0.0,
            'structure_quality': 0.0,
            'entity_richness': 0.0
        }
        
        try:
            # Readability (simplified)
            sentences = len(re.findall(r'[.!?]+', document_text))
            words = len(document_text.split())
            avg_sentence_length = words / max(sentences, 1)
            quality_metrics['readability_score'] = min(1.0, max(0.0, 1.0 - (avg_sentence_length - 15) / 30))
            
            # Completeness (based on identified segments)
            segments_count = len(analysis_result.get('segments', {}))
            quality_metrics['completeness_score'] = min(1.0, segments_count / 5.0)  # Expect ~5 segments
            
            # Legal terminology density
            legal_terms = [
                'pursuant', 'whereas', 'therefore', 'hereby', 'agreement',
                'contract', 'party', 'clause', 'section', 'provision'
            ]
            legal_term_count = sum(1 for term in legal_terms if term.lower() in document_text.lower())
            quality_metrics['legal_terminology_density'] = min(1.0, legal_term_count / 10.0)
            
            # Structure quality (based on segmentation confidence)
            quality_metrics['structure_quality'] = analysis_result.get('segmentation_confidence', 0.5)
            
            # Entity richness
            entities_count = len(analysis_result.get('key_entities', []))
            quality_metrics['entity_richness'] = min(1.0, entities_count / 10.0)
            
            # Overall quality score
            quality_metrics['overall_score'] = np.mean(list(quality_metrics.values()))
            
        except Exception as e:
            quality_metrics['assessment_error'] = str(e)
        
        return quality_metrics
    
    def _generate_document_fingerprint(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate unique document fingerprint for similarity detection"""
        
        fingerprint = {
            'document_type_hash': hash(analysis_result.get('document_type', '')),
            'legal_domain_hash': hash(analysis_result.get('legal_domain', '')),
            'entity_signature': [],
            'segment_signature': [],
            'statute_signature': []
        }
        
        try:
            # Entity signature
            entities = analysis_result.get('key_entities', [])
            entity_types = [entity['type'] for entity in entities]
            fingerprint['entity_signature'] = list(set(entity_types))  # Unique entity types
            
            # Segment signature
            segments = analysis_result.get('segments', {})
            fingerprint['segment_signature'] = list(segments.keys())
            
            # Statute signature
            statutes = analysis_result.get('relevant_statutes', [])
            statute_names = [statute.get('statute', '') for statute in statutes]
            fingerprint['statute_signature'] = list(set(statute_names))
            
            # Generate composite hash
            composite_data = (
                str(fingerprint['document_type_hash']) +
                str(fingerprint['legal_domain_hash']) +
                ''.join(sorted(fingerprint['entity_signature'])) +
                ''.join(sorted(fingerprint['segment_signature']))
            )
            fingerprint['composite_hash'] = hash(composite_data)
            
        except Exception as e:
            fingerprint['fingerprint_error'] = str(e)
        
        return fingerprint
    
    def _calculate_analysis_confidence(self, analysis_result: Dict[str, Any]) -> float:
        """Calculate overall analysis confidence score"""
        
        confidence_factors = []
        
        # Classification confidence
        classification_conf = analysis_result.get('classification_confidence', 0.5)
        confidence_factors.append(classification_conf)
        
        # Segmentation confidence
        segmentation_conf = analysis_result.get('segmentation_confidence', 0.5)
        confidence_factors.append(segmentation_conf)
        
        # Entity extraction confidence
        entities = analysis_result.get('key_entities', [])
        if entities:
            entity_confidences = [entity.get('confidence', 0.5) for entity in entities]
            avg_entity_conf = np.mean(entity_confidences)
            confidence_factors.append(avg_entity_conf)
        else:
            confidence_factors.append(0.3)  # Low confidence if no entities found
        
        # Statute identification confidence
        statutes = analysis_result.get('relevant_statutes', [])
        if statutes:
            statute_confidences = [statute.get('relevance_score', 0.5) for statute in statutes]
            avg_statute_conf = np.mean(statute_confidences)
            confidence_factors.append(avg_statute_conf)
        else:
            confidence_factors.append(0.4)
        
        # Quality metrics confidence
        quality_metrics = analysis_result.get('quality_metrics', {})
        overall_quality = quality_metrics.get('overall_score', 0.5)
        confidence_factors.append(overall_quality)
        
        # Calculate weighted average
        weights = [0.25, 0.25, 0.2, 0.15, 0.15]  # Weights for different factors
        weighted_confidence = sum(w * f for w, f in zip(weights, confidence_factors))
        
        return min(max(weighted_confidence, 0.0), 1.0)  # Clamp to [0, 1]
