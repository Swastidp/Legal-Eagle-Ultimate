import asyncio
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import google.generativeai as genai

class LegalResearchAgent:
    """Specialized agent for conducting legal research using Gemini Pro"""
    
    def __init__(self, gemini_pro_model):
        self.gemini_pro = gemini_pro_model
        
        # Indian legal research databases and sources
        self.indian_legal_sources = {
            'supreme_court_cases': 'Supreme Court of India judgments and orders',
            'high_court_cases': 'High Court judgments across all Indian states',
            'acts_and_statutes': 'Central and State Acts, Rules, and Regulations',
            'regulatory_notifications': 'SEBI, RBI, MCA, and other regulatory body notifications',
            'legal_precedents': 'Binding and persuasive precedents in Indian law',
            'constitutional_provisions': 'Constitutional law and fundamental rights jurisprudence'
        }
        
        # Research methodologies
        self.research_approaches = {
            'precedent_analysis': 'Analysis of case law and judicial precedents',
            'statutory_interpretation': 'Analysis of statutes, rules, and regulations',
            'comparative_analysis': 'Comparison with similar cases or legal issues',
            'constitutional_review': 'Constitutional law and fundamental rights analysis',
            'regulatory_compliance': 'Regulatory requirements and compliance analysis',
            'recent_developments': 'Latest legal developments and amendments'
        }
    
    async def conduct_legal_research(self, document_text: str, jurisdiction: str,
                                   key_entities: List[str] = None,
                                   research_scope: str = "comprehensive") -> Dict[str, Any]:
        """Conduct comprehensive legal research"""
        
        research_result = {
            'research_id': f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'timestamp': datetime.now().isoformat(),
            'jurisdiction': jurisdiction,
            'research_scope': research_scope,
            'document_summary': document_text[:500] + "..." if len(document_text) > 500 else document_text,
            'key_entities': key_entities or [],
            'research_findings': {},
            'legal_precedents': [],
            'statutory_analysis': {},
            'recent_developments': [],
            'research_confidence': 0.0
        }
        
        try:
            # Phase 1: Identify research areas
            research_areas = await self._identify_research_areas(document_text, key_entities)
            research_result['research_areas'] = research_areas
            
            # Phase 2: Conduct parallel research tasks
            research_tasks = []
            
            if 'case_law' in research_areas:
                research_tasks.append(
                    ('case_law_research', self._research_case_law(document_text, jurisdiction))
                )
            
            if 'statutory_analysis' in research_areas:
                research_tasks.append(
                    ('statutory_research', self._research_statutes_and_acts(document_text, jurisdiction))
                )
            
            if 'regulatory_compliance' in research_areas:
                research_tasks.append(
                    ('regulatory_research', self._research_regulatory_requirements(document_text, jurisdiction))
                )
            
            if 'recent_developments' in research_areas:
                research_tasks.append(
                    ('developments_research', self._research_recent_developments(document_text, jurisdiction))
                )
            
            # Execute research tasks in parallel
            if research_tasks:
                task_results = await asyncio.gather(
                    *[task[1] for task in research_tasks],
                    return_exceptions=True
                )
                
                # Process research results
                for i, (task_name, result) in enumerate(zip([task[0] for task in research_tasks], task_results)):
                    if not isinstance(result, Exception):
                        research_result['research_findings'][task_name] = result
                    else:
                        research_result['research_findings'][task_name] = {
                            'error': str(result),
                            'status': 'failed'
                        }
            
            # Phase 3: Generate comprehensive analysis
            comprehensive_analysis = await self._generate_comprehensive_analysis(research_result)
            research_result.update(comprehensive_analysis)
            
            # Phase 4: Calculate research confidence
            research_result['research_confidence'] = self._calculate_research_confidence(research_result)
            
        except Exception as e:
            research_result['error'] = str(e)
            research_result['status'] = 'failed'
        
        return research_result
    
    async def _identify_research_areas(self, document_text: str, key_entities: List[str]) -> List[str]:
        """Identify key areas that require legal research"""
        
        identification_prompt = f"""
        Analyze this legal document and identify key areas that require legal research:
        
        Document Content: {document_text[:1500]}
        Key Entities: {', '.join(key_entities) if key_entities else 'None identified'}
        
        Identify which of these research areas are relevant:
        1. case_law - Need to research judicial precedents and court decisions
        2. statutory_analysis - Need to analyze relevant statutes, acts, and legislation
        3. regulatory_compliance - Need to research regulatory requirements and compliance
        4. constitutional_law - Need to analyze constitutional provisions and rights
        5. recent_developments - Need to research recent legal developments and amendments
        6. comparative_analysis - Need to compare with similar legal situations
        
        Return only a JSON array of relevant research areas: ["area1", "area2", ...]
        """
        
        try:
            response = await self.gemini_pro.generate_content_async(identification_prompt)
            research_areas = json.loads(response.text.strip())
            
            if isinstance(research_areas, list):
                return research_areas
            else:
                return ['case_law', 'statutory_analysis']  # Default areas
                
        except Exception as e:
            return ['case_law', 'statutory_analysis', 'regulatory_compliance']  # Fallback
    
    async def _research_case_law(self, document_text: str, jurisdiction: str) -> Dict[str, Any]:
        """Research relevant case law and judicial precedents"""
        
        case_law_prompt = f"""
        Research relevant case law and judicial precedents for this legal matter in {jurisdiction}:
        
        Document Context: {document_text[:2000]}
        
        Focus on:
        1. Supreme Court of India landmark judgments
        2. Relevant High Court decisions
        3. Recent judicial trends and interpretations
        4. Binding precedents that apply to this matter
        5. Persuasive precedents from other jurisdictions
        
        Provide detailed analysis in JSON format:
        {{
            "landmark_cases": [
                {{
                    "case_name": "Case citation",
                    "year": "YYYY",
                    "court": "Court name",
                    "legal_principle": "Key legal principle established",
                    "relevance": "How it applies to current matter",
                    "ratio_decidendi": "Core legal reasoning"
                }}
            ],
            "recent_decisions": [
                {{
                    "case_name": "Recent case",
                    "year": "YYYY",
                    "court": "Court name",
                    "impact": "Impact on current legal landscape"
                }}
            ],
            "judicial_trends": ["trend1", "trend2"],
            "applicable_principles": ["principle1", "principle2"]
        }}
        """
        
        try:
            response = await self.gemini_pro.generate_content_async(case_law_prompt)
            case_law_data = json.loads(response.text.strip('``````'))
            
            case_law_data['research_timestamp'] = datetime.now().isoformat()
            case_law_data['jurisdiction'] = jurisdiction
            
            return case_law_data
            
        except Exception as e:
            return {
                'error': str(e),
                'landmark_cases': [],
                'recent_decisions': [],
                'judicial_trends': ['Research unavailable'],
                'applicable_principles': ['Manual legal research recommended']
            }
    
    async def _research_statutes_and_acts(self, document_text: str, jurisdiction: str) -> Dict[str, Any]:
        """Research relevant statutes, acts, and legislation"""
        
        statutory_prompt = f"""
        Research relevant Indian statutes, acts, and legislation for this legal matter:
        
        Document Context: {document_text[:2000]}
        Jurisdiction: {jurisdiction}
        
        Focus on:
        1. Central Acts and their relevant sections
        2. State-specific legislation (if applicable)
        3. Rules and regulations under the Acts
        4. Recent amendments and notifications
        5. Statutory interpretation by courts
        
        Provide analysis in JSON format:
        {{
            "primary_acts": [
                {{
                    "act_name": "Full name of the Act",
                    "year": "Year of enactment",
                    "relevant_sections": ["Section X", "Section Y"],
                    "key_provisions": "Summary of key applicable provisions",
                    "recent_amendments": "Recent changes if any"
                }}
            ],
            "rules_and_regulations": [
                {{
                    "rule_name": "Name of rules/regulations",
                    "under_act": "Parent Act",
                    "relevant_provisions": "Applicable provisions"
                }}
            ],
            "statutory_compliance": [
                {{
                    "requirement": "Compliance requirement",
                    "statutory_basis": "Legal basis",
                    "penalty": "Consequences of non-compliance"
                }}
            ],
            "interpretation_guidelines": ["guideline1", "guideline2"]
        }}
        """
        
        try:
            response = await self.gemini_pro.generate_content_async(statutory_prompt)
            statutory_data = json.loads(response.text.strip('``````'))
            
            statutory_data['research_timestamp'] = datetime.now().isoformat()
            statutory_data['jurisdiction'] = jurisdiction
            
            return statutory_data
            
        except Exception as e:
            return {
                'error': str(e),
                'primary_acts': [],
                'rules_and_regulations': [],
                'statutory_compliance': [],
                'interpretation_guidelines': ['Manual statutory research recommended']
            }
    
    async def _research_regulatory_requirements(self, document_text: str, jurisdiction: str) -> Dict[str, Any]:
        """Research regulatory requirements and compliance obligations"""
        
        regulatory_prompt = f"""
        Research regulatory requirements and compliance obligations for this matter in {jurisdiction}:
        
        Document Context: {document_text[:2000]}
        
        Focus on relevant regulatory bodies:
        1. SEBI (Securities and Exchange Board of India)
        2. RBI (Reserve Bank of India)
        3. MCA (Ministry of Corporate Affairs)
        4. TRAI (Telecom Regulatory Authority)
        5. IRDAI (Insurance Regulatory Authority)
        6. State regulatory authorities
        
        Provide analysis in JSON format:
        {{
            "regulatory_bodies": [
                {{
                    "authority": "Regulatory body name",
                    "jurisdiction": "Area of regulation",
                    "applicable_regulations": ["regulation1", "regulation2"],
                    "compliance_requirements": "Key compliance obligations",
                    "reporting_requirements": "Reporting and disclosure needs"
                }}
            ],
            "compliance_timeline": [
                {{
                    "requirement": "Specific requirement",
                    "deadline": "Compliance deadline",
                    "consequence": "Penalty for non-compliance"
                }}
            ],
            "regulatory_trends": ["trend1", "trend2"],
            "upcoming_changes": ["change1", "change2"]
        }}
        """
        
        try:
            response = await self.gemini_pro.generate_content_async(regulatory_prompt)
            regulatory_data = json.loads(response.text.strip('``````'))
            
            regulatory_data['research_timestamp'] = datetime.now().isoformat()
            regulatory_data['jurisdiction'] = jurisdiction
            
            return regulatory_data
            
        except Exception as e:
            return {
                'error': str(e),
                'regulatory_bodies': [],
                'compliance_timeline': [],
                'regulatory_trends': ['Regulatory research unavailable'],
                'upcoming_changes': ['Manual regulatory research recommended']
            }
    
    async def _research_recent_developments(self, document_text: str, jurisdiction: str) -> Dict[str, Any]:
        """Research recent legal developments and amendments"""
        
        developments_prompt = f"""
        Research recent legal developments, amendments, and emerging trends relevant to this matter in {jurisdiction}:
        
        Document Context: {document_text[:1500]}
        
        Focus on developments in the last 2 years:
        1. Recent legislative amendments
        2. New judicial interpretations
        3. Regulatory policy changes
        4. Emerging legal trends
        5. Technology and law intersection
        
        Provide analysis in JSON format:
        {{
            "recent_amendments": [
                {{
                    "legislation": "Act/Rule name",
                    "amendment_date": "Date",
                    "key_changes": "Summary of changes",
                    "impact": "Impact on current matter"
                }}
            ],
            "judicial_developments": [
                {{
                    "development": "New judicial trend/interpretation",
                    "significance": "Why it's important",
                    "implications": "Practical implications"
                }}
            ],
            "regulatory_changes": [
                {{
                    "change": "Policy/regulation change",
                    "effective_date": "When it takes effect",
                    "relevance": "How it affects current matter"
                }}
            ],
            "emerging_trends": ["trend1", "trend2"],
            "future_outlook": "Expected future developments"
        }}
        """
        
        try:
            response = await self.gemini_pro.generate_content_async(developments_prompt)
            developments_data = json.loads(response.text.strip('``````'))
            
            developments_data['research_timestamp'] = datetime.now().isoformat()
            developments_data['research_period'] = 'Last 24 months'
            
            return developments_data
            
        except Exception as e:
            return {
                'error': str(e),
                'recent_amendments': [],
                'judicial_developments': [],
                'regulatory_changes': [],
                'emerging_trends': ['Developments research unavailable'],
                'future_outlook': 'Manual trend analysis recommended'
            }
    
    async def _generate_comprehensive_analysis(self, research_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive analysis of all research findings"""
        
        research_findings = research_result.get('research_findings', {})
        
        analysis_prompt = f"""
        Based on comprehensive legal research, provide executive analysis:
        
        Research Areas Covered: {', '.join(research_result.get('research_areas', []))}
        Key Entities: {', '.join(research_result.get('key_entities', []))}
        Jurisdiction: {research_result.get('jurisdiction')}
        
        Research Findings Summary:
        {json.dumps(research_findings, indent=2)[:3000]}
        
        Provide executive analysis addressing:
        1. Key legal implications and risks
        2. Strategic recommendations based on research
        3. Priority actions required
        4. Areas requiring further investigation
        5. Overall legal position assessment
        
        Format as professional legal research summary suitable for decision-makers.
        """
        
        try:
            response = await self.gemini_pro.generate_content_async(analysis_prompt)
            
            return {
                'comprehensive_analysis': response.text.strip(),
                'research_summary': self._create_research_summary(research_findings),
                'strategic_insights': self._extract_strategic_insights(research_findings),
                'action_recommendations': self._generate_action_recommendations(research_findings)
            }
            
        except Exception as e:
            return {
                'comprehensive_analysis': f'Comprehensive analysis generation failed: {str(e)}',
                'research_summary': 'Research summary unavailable',
                'strategic_insights': ['Manual analysis recommended'],
                'action_recommendations': ['Consult legal counsel for detailed guidance']
            }
    
    def _create_research_summary(self, research_findings: Dict[str, Any]) -> Dict[str, Any]:
        """Create structured summary of research findings"""
        
        summary = {
            'total_research_areas': len(research_findings),
            'successful_research': sum(1 for finding in research_findings.values() 
                                     if 'error' not in finding),
            'failed_research': sum(1 for finding in research_findings.values() 
                                 if 'error' in finding),
            'key_findings': []
        }
        
        # Extract key findings from each research area
        for area, findings in research_findings.items():
            if 'error' not in findings:
                if area == 'case_law_research':
                    landmark_count = len(findings.get('landmark_cases', []))
                    recent_count = len(findings.get('recent_decisions', []))
                    summary['key_findings'].append(
                        f"Case Law: {landmark_count} landmark cases, {recent_count} recent decisions"
                    )
                elif area == 'statutory_research':
                    acts_count = len(findings.get('primary_acts', []))
                    summary['key_findings'].append(f"Statutory: {acts_count} relevant acts identified")
                elif area == 'regulatory_research':
                    bodies_count = len(findings.get('regulatory_bodies', []))
                    summary['key_findings'].append(f"Regulatory: {bodies_count} regulatory bodies involved")
        
        return summary
    
    def _extract_strategic_insights(self, research_findings: Dict[str, Any]) -> List[str]:
        """Extract strategic insights from research findings"""
        
        insights = []
        
        # Case law insights
        case_law_data = research_findings.get('case_law_research', {})
        if case_law_data and 'error' not in case_law_data:
            landmark_cases = case_law_data.get('landmark_cases', [])
            if landmark_cases:
                insights.append(f"Strong precedent support available from {len(landmark_cases)} landmark cases")
            
            judicial_trends = case_law_data.get('judicial_trends', [])
            if judicial_trends:
                insights.append(f"Current judicial trends: {', '.join(judicial_trends[:2])}")
        
        # Regulatory insights
        regulatory_data = research_findings.get('regulatory_research', {})
        if regulatory_data and 'error' not in regulatory_data:
            compliance_timeline = regulatory_data.get('compliance_timeline', [])
            if compliance_timeline:
                insights.append("Multiple compliance deadlines require immediate attention")
        
        # Recent developments insights
        developments_data = research_findings.get('developments_research', {})
        if developments_data and 'error' not in developments_data:
            recent_amendments = developments_data.get('recent_amendments', [])
            if recent_amendments:
                insights.append("Recent legislative changes may impact current legal position")
        
        return insights or ["Comprehensive legal research completed - review detailed findings"]
    
    def _generate_action_recommendations(self, research_findings: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actionable recommendations based on research"""
        
        recommendations = []
        
        # High-priority recommendations based on research findings
        for area, findings in research_findings.items():
            if 'error' not in findings:
                if area == 'regulatory_research':
                    compliance_timeline = findings.get('compliance_timeline', [])
                    for compliance in compliance_timeline[:2]:  # Top 2 compliance items
                        recommendations.append({
                            'priority': 'High',
                            'category': 'Regulatory Compliance',
                            'action': f"Address {compliance.get('requirement', 'compliance requirement')}",
                            'deadline': compliance.get('deadline', 'TBD'),
                            'basis': 'Regulatory research findings'
                        })
                
                elif area == 'case_law_research':
                    landmark_cases = findings.get('landmark_cases', [])
                    if landmark_cases:
                        recommendations.append({
                            'priority': 'Medium',
                            'category': 'Legal Strategy',
                            'action': 'Analyze precedent applicability and develop case strategy',
                            'deadline': '2 weeks',
                            'basis': 'Case law research findings'
                        })
        
        # Default recommendations if no specific findings
        if not recommendations:
            recommendations = [
                {
                    'priority': 'Medium',
                    'category': 'Legal Review',
                    'action': 'Conduct detailed legal review based on research findings',
                    'deadline': '1-2 weeks',
                    'basis': 'General research completion'
                }
            ]
        
        return recommendations
    
    def _calculate_research_confidence(self, research_result: Dict[str, Any]) -> float:
        """Calculate confidence score for research results"""
        
        confidence_factors = []
        
        # Research completion rate
        research_findings = research_result.get('research_findings', {})
        if research_findings:
            successful_research = sum(1 for finding in research_findings.values() 
                                    if 'error' not in finding)
            completion_rate = successful_research / len(research_findings)
            confidence_factors.append(completion_rate)
        
        # Research area coverage
        research_areas = research_result.get('research_areas', [])
        coverage_score = min(len(research_areas) / 4.0, 1.0)  # Expect up to 4 research areas
        confidence_factors.append(coverage_score)
        
        # Entity-based research depth
        key_entities = research_result.get('key_entities', [])
        entity_score = min(len(key_entities) / 5.0, 1.0) if key_entities else 0.3
        confidence_factors.append(entity_score)
        
        return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5
