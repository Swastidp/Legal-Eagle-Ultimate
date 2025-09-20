import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import google.generativeai as genai

class SummaryGeneratorAgent:
    """Specialized agent for generating intelligent legal document summaries"""
    
    def __init__(self, gemini_model):
        self.gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Summary templates for different document types
        self.summary_templates = {
            'contract': {
                'key_sections': ['Parties', 'Terms', 'Financial', 'Obligations', 'Termination', 'Dispute Resolution'],
                'focus_areas': ['payment terms', 'liability', 'termination clauses', 'governing law']
            },
            'corporate': {
                'key_sections': ['Corporate Structure', 'Governance', 'Compliance', 'Financial', 'Regulatory'],
                'focus_areas': ['board composition', 'regulatory compliance', 'shareholder rights', 'disclosure requirements']
            },
            'employment': {
                'key_sections': ['Employment Terms', 'Compensation', 'Responsibilities', 'Benefits', 'Termination'],
                'focus_areas': ['salary structure', 'notice period', 'confidentiality', 'non-compete']
            },
            'legal_judgment': {
                'key_sections': ['Facts', 'Issues', 'Arguments', 'Reasoning', 'Holdings', 'Orders'],
                'focus_areas': ['legal precedents', 'ratio decidendi', 'court orders', 'legal principles']
            }
        }
    
    async def generate_comprehensive_summary(self, document_text: str, 
                                           structure_analysis: Dict[str, Any],
                                           jurisdiction: str = "Indian Law") -> Dict[str, Any]:
        """Generate comprehensive intelligent summary"""
        
        document_type = structure_analysis.get('document_type', 'Legal Document').lower()
        legal_domain = structure_analysis.get('legal_domain', 'general')
        
        # Select appropriate template
        template_key = self._select_template(document_type, legal_domain)
        template = self.summary_templates.get(template_key, self.summary_templates['contract'])
        
        try:
            # Generate executive summary
            executive_summary = await self._generate_executive_summary(
                document_text, document_type, jurisdiction, template
            )
            
            # Extract key information
            key_information = await self._extract_key_information(
                document_text, template, structure_analysis
            )
            
            # Generate action-oriented insights
            actionable_insights = await self._generate_actionable_insights(
                document_text, key_information, jurisdiction
            )
            
            # Create summary dashboard data
            summary_dashboard = self._create_summary_dashboard(
                key_information, actionable_insights, structure_analysis
            )
            
            return {
                'executive_summary': executive_summary,
                'key_information': key_information,
                'actionable_insights': actionable_insights,
                'summary_dashboard': summary_dashboard,
                'document_type': document_type,
                'template_used': template_key,
                'confidence': 0.85,
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            return self._generate_fallback_summary(document_text, str(e))
    
    async def _generate_executive_summary(self, document_text: str, document_type: str,
                                        jurisdiction: str, template: Dict[str, Any]) -> str:
        """Generate executive summary using Gemini"""
        
        summary_prompt = f"""
        Create a professional executive summary for this {document_type} under {jurisdiction}:
        
        Focus Areas: {', '.join(template['focus_areas'])}
        
        DOCUMENT CONTENT:
        {document_text[:2000]}
        
        Generate a concise executive summary (3-4 sentences) that:
        1. Identifies the document type and primary purpose
        2. Highlights the most critical legal and business implications
        3. Notes any significant risks or opportunities
        4. Provides context for decision-making
        
        Write in clear, business-friendly language suitable for executives who may not have legal background.
        """
        
        try:
            response = await self.gemini_model.generate_content_async(summary_prompt)
            return response.text.strip()
            
        except Exception as e:
            return f"This {document_type} requires legal review. Key provisions and obligations should be carefully examined, with particular attention to risks and compliance requirements under {jurisdiction}."
    
    async def _extract_key_information(self, document_text: str, template: Dict[str, Any],
                                     structure_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key information based on document template"""
        
        extraction_prompt = f"""
        Extract key information from this legal document:
        
        Target Sections: {', '.join(template['key_sections'])}
        Focus Areas: {', '.join(template['focus_areas'])}
        Document Domain: {structure_analysis.get('legal_domain', 'general')}
        
        DOCUMENT:
        {document_text[:3000]}
        
        Extract and structure the following information in JSON format:
        {{
            "parties": ["party1", "party2"],
            "key_dates": [
                {{"date": "YYYY-MM-DD", "event": "description"}}
            ],
            "financial_terms": [
                {{"amount": "value", "description": "purpose", "currency": "INR"}}
            ],
            "obligations": [
                {{"party": "party_name", "obligation": "description", "deadline": "timeline"}}
            ],
            "risks_identified": ["risk1", "risk2"],
            "governing_provisions": ["law/section1", "law/section2"],
            "critical_clauses": [
                {{"clause_type": "type", "content": "summary", "importance": "high/medium/low"}}
            ]
        }}
        """
        
        try:
            response = await self.gemini_model.generate_content_async(extraction_prompt)
            
            # Parse JSON response
            extracted_data = json.loads(response.text.strip('``````'))
            
            # Validate and enhance extracted data
            return self._validate_extracted_information(extracted_data)
            
        except Exception as e:
            return self._generate_fallback_key_information(document_text)
    
    async def _generate_actionable_insights(self, document_text: str, key_information: Dict[str, Any],
                                          jurisdiction: str) -> Dict[str, Any]:
        """Generate actionable insights and recommendations"""
        
        insights_prompt = f"""
        Based on this legal document analysis, provide actionable insights for {jurisdiction}:
        
        Key Information Extracted:
        - Parties: {key_information.get('parties', [])}
        - Financial Terms: {len(key_information.get('financial_terms', []))} items
        - Obligations: {len(key_information.get('obligations', []))} items
        - Risks: {key_information.get('risks_identified', [])}
        
        Generate actionable insights in JSON format:
        {{
            "immediate_actions": [
                {{"action": "specific_action", "deadline": "timeline", "responsible": "role"}}
            ],
            "compliance_requirements": [
                {{"requirement": "description", "deadline": "date", "penalty": "consequence"}}
            ],
            "opportunities": ["opportunity1", "opportunity2"],
            "recommendations": [
                {{"category": "risk/compliance/business", "recommendation": "specific_advice", "priority": "high/medium/low"}}
            ],
            "monitoring_points": ["point1", "point2"],
            "escalation_triggers": ["trigger1", "trigger2"]
        }}
        """
        
        try:
            response = await self.gemini_model.generate_content_async(insights_prompt)
            insights_data = json.loads(response.text.strip('``````'))
            
            return insights_data
            
        except Exception as e:
            return {
                'immediate_actions': [
                    {'action': 'Review document with legal counsel', 'deadline': '1 week', 'responsible': 'Legal Team'}
                ],
                'compliance_requirements': [],
                'opportunities': ['Document review completed'],
                'recommendations': [
                    {'category': 'general', 'recommendation': 'Conduct thorough legal review', 'priority': 'medium'}
                ],
                'monitoring_points': ['Legal compliance status'],
                'escalation_triggers': ['Regulatory changes']
            }
    
    def _create_summary_dashboard(self, key_information: Dict[str, Any],
                                actionable_insights: Dict[str, Any],
                                structure_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create dashboard-style summary data"""
        
        return {
            'document_metrics': {
                'parties_count': len(key_information.get('parties', [])),
                'financial_terms_count': len(key_information.get('financial_terms', [])),
                'obligations_count': len(key_information.get('obligations', [])),
                'critical_clauses_count': len(key_information.get('critical_clauses', [])),
                'risks_count': len(key_information.get('risks_identified', []))
            },
            'priority_indicators': {
                'high_priority_actions': len([a for a in actionable_insights.get('immediate_actions', []) 
                                            if 'deadline' in a and '1 week' in a['deadline']]),
                'compliance_deadlines': len(actionable_insights.get('compliance_requirements', [])),
                'escalation_triggers': len(actionable_insights.get('escalation_triggers', []))
            },
            'document_health_score': self._calculate_document_health_score(key_information, actionable_insights),
            'complexity_level': self._assess_document_complexity(structure_analysis),
            'review_recommendations': self._generate_review_schedule(actionable_insights)
        }
    
    def _select_template(self, document_type: str, legal_domain: str) -> str:
        """Select appropriate summary template"""
        
        if 'contract' in document_type or legal_domain == 'contract':
            return 'contract'
        elif 'corporate' in document_type or legal_domain == 'corporate':
            return 'corporate'
        elif 'employment' in document_type or legal_domain == 'employment':
            return 'employment'
        elif 'judgment' in document_type or 'court' in document_type:
            return 'legal_judgment'
        else:
            return 'contract'  # Default template
    
    def _validate_extracted_information(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and ensure consistency in extracted information"""
        
        # Ensure all required fields exist
        required_fields = ['parties', 'key_dates', 'financial_terms', 'obligations', 
                          'risks_identified', 'governing_provisions', 'critical_clauses']
        
        for field in required_fields:
            if field not in extracted_data:
                extracted_data[field] = []
        
        # Validate parties
        if not isinstance(extracted_data['parties'], list):
            extracted_data['parties'] = []
        
        # Validate financial terms
        for term in extracted_data.get('financial_terms', []):
            if not isinstance(term, dict):
                continue
            if 'currency' not in term:
                term['currency'] = 'INR'  # Default to Indian Rupees
        
        # Add extraction metadata
        extracted_data['extraction_metadata'] = {
            'extracted_at': datetime.now().isoformat(),
            'validation_passed': True,
            'total_fields_extracted': sum(len(v) if isinstance(v, list) else 1 for v in extracted_data.values())
        }
        
        return extracted_data
    
    def _generate_fallback_key_information(self, document_text: str) -> Dict[str, Any]:
        """Generate fallback key information when extraction fails"""
        
        import re
        
        # Simple extraction using regex patterns
        parties = re.findall(r'[A-Z][a-zA-Z\s]+(?:Ltd|Limited|Inc|Corp|Company)', document_text)
        dates = re.findall(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', document_text)
        amounts = re.findall(r'Rs\.?\s*[\d,]+', document_text)
        
        return {
            'parties': list(set(parties[:5])),  # Unique parties, max 5
            'key_dates': [{'date': date, 'event': 'Extracted date'} for date in dates[:3]],
            'financial_terms': [{'amount': amount, 'description': 'Extracted amount', 'currency': 'INR'} 
                              for amount in amounts[:3]],
            'obligations': [],
            'risks_identified': ['Document analysis incomplete - manual review required'],
            'governing_provisions': [],
            'critical_clauses': [],
            'extraction_method': 'fallback_regex'
        }
    
    def _generate_fallback_summary(self, document_text: str, error: str) -> Dict[str, Any]:
        """Generate fallback summary when main process fails"""
        
        # Basic summary using first few sentences
        sentences = document_text.split('.')[:3]
        basic_summary = '. '.join(sentences) + '.'
        
        return {
            'executive_summary': basic_summary,
            'key_information': self._generate_fallback_key_information(document_text),
            'actionable_insights': {
                'immediate_actions': [
                    {'action': 'Complete manual document review', 'deadline': '1 week', 'responsible': 'Legal Team'}
                ],
                'recommendations': [
                    {'category': 'technical', 'recommendation': 'Retry analysis with different approach', 'priority': 'medium'}
                ]
            },
            'summary_dashboard': {
                'document_metrics': {'analysis_status': 'incomplete'},
                'document_health_score': 0.5,
                'complexity_level': 'unknown'
            },
            'error': error,
            'confidence': 0.3,
            'generated_at': datetime.now().isoformat()
        }
    
    def _calculate_document_health_score(self, key_info: Dict[str, Any], insights: Dict[str, Any]) -> float:
        """Calculate overall document health score (0-1)"""
        
        score_factors = []
        
        # Factor 1: Completeness
        required_elements = ['parties', 'obligations', 'governing_provisions']
        completeness = sum(1 for elem in required_elements if key_info.get(elem)) / len(required_elements)
        score_factors.append(completeness)
        
        # Factor 2: Risk level (inverse)
        risk_count = len(key_info.get('risks_identified', []))
        risk_factor = max(0, 1 - (risk_count / 10))  # Assume 10+ risks = 0 score
        score_factors.append(risk_factor)
        
        # Factor 3: Action requirements (inverse)
        immediate_actions = len(insights.get('immediate_actions', []))
        action_factor = max(0, 1 - (immediate_actions / 5))  # Assume 5+ actions = 0 score
        score_factors.append(action_factor)
        
        return sum(score_factors) / len(score_factors)
    
    def _assess_document_complexity(self, structure_analysis: Dict[str, Any]) -> str:
        """Assess document complexity level"""
        
        segments_count = len(structure_analysis.get('segments', {}))
        entities_count = len(structure_analysis.get('key_entities', []))
        
        total_complexity_score = segments_count + entities_count
        
        if total_complexity_score > 15:
            return 'high'
        elif total_complexity_score > 8:
            return 'medium'
        else:
            return 'low'
    
    def _generate_review_schedule(self, insights: Dict[str, Any]) -> Dict[str, str]:
        """Generate recommended review schedule"""
        
        immediate_actions = len(insights.get('immediate_actions', []))
        compliance_reqs = len(insights.get('compliance_requirements', []))
        
        if immediate_actions > 2 or compliance_reqs > 1:
            return {
                'next_review': '1 week',
                'regular_review_frequency': 'Monthly',
                'compliance_review': 'Quarterly'
            }
        else:
            return {
                'next_review': '1 month',
                'regular_review_frequency': 'Quarterly',
                'compliance_review': 'Semi-annually'
            }
