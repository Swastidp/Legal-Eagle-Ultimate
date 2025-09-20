import asyncio
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import google.generativeai as genai
import numpy as np

class RiskAssessmentAgent:
    """Specialized agent for comprehensive legal risk assessment"""
    
    def __init__(self, gemini_model):
        self.gemini_model = gemini_model
        
        # Indian legal risk categories
        self.indian_risk_categories = {
            'contractual_risks': {
                'weight': 0.25,
                'subcategories': [
                    'breach_of_contract', 'termination_risks', 'penalty_clauses',
                    'indemnity_exposure', 'limitation_of_liability', 'force_majeure'
                ]
            },
            'regulatory_compliance_risks': {
                'weight': 0.30,
                'subcategories': [
                    'companies_act_compliance', 'sebi_regulations', 'rbi_guidelines',
                    'tax_compliance', 'labor_law_compliance', 'environmental_compliance'
                ]
            },
            'litigation_risks': {
                'weight': 0.20,
                'subcategories': [
                    'dispute_likelihood', 'court_jurisdiction_issues', 'enforcement_challenges',
                    'precedent_unfavorability', 'limitation_period_issues'
                ]
            },
            'financial_risks': {
                'weight': 0.15,
                'subcategories': [
                    'payment_default', 'currency_fluctuation', 'interest_rate_changes',
                    'credit_risk', 'liquidity_risk', 'operational_cost_escalation'
                ]
            },
            'operational_risks': {
                'weight': 0.10,
                'subcategories': [
                    'business_disruption', 'key_personnel_dependency', 'technology_risks',
                    'supply_chain_risks', 'reputation_risks', 'data_security_risks'
                ]
            }
        }
        
        # Risk severity levels
        self.risk_severity_levels = {
            'critical': {'score': 90, 'color': 'red', 'action_timeline': 'immediate'},
            'high': {'score': 75, 'color': 'orange', 'action_timeline': '1-2 weeks'},
            'medium': {'score': 50, 'color': 'yellow', 'action_timeline': '1-3 months'},
            'low': {'score': 25, 'color': 'green', 'action_timeline': '3-6 months'}
        }
    
    async def comprehensive_risk_assessment(self, document_text: str, jurisdiction: str,
                                          legal_domain: str = "general",
                                          context_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Perform comprehensive legal risk assessment"""
        
        risk_assessment = {
            'assessment_id': f"risk_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'timestamp': datetime.now().isoformat(),
            'jurisdiction': jurisdiction,
            'legal_domain': legal_domain,
            'document_context': context_data or {},
            'overall_risk_score': 0,
            'overall_risk_level': 'medium',
            'risk_categories': {},
            'identified_risks': [],
            'critical_risks': [],
            'risk_mitigation_strategies': [],
            'confidence_score': 0.0
        }
        
        try:
            # Phase 1: Category-wise Risk Analysis
            category_analyses = await self._analyze_risk_categories(
                document_text, jurisdiction, legal_domain
            )
            risk_assessment['risk_categories'] = category_analyses
            
            # Phase 2: Specific Risk Identification
            identified_risks = await self._identify_specific_risks(
                document_text, jurisdiction, legal_domain, category_analyses
            )
            risk_assessment['identified_risks'] = identified_risks
            
            # Phase 3: Critical Risk Evaluation
            critical_risks = self._evaluate_critical_risks(identified_risks)
            risk_assessment['critical_risks'] = critical_risks
            
            # Phase 4: Risk Quantification
            overall_score, overall_level = self._calculate_overall_risk_score(
                category_analyses, identified_risks
            )
            risk_assessment['overall_risk_score'] = overall_score
            risk_assessment['overall_risk_level'] = overall_level
            
            # Phase 5: Mitigation Strategy Generation
            mitigation_strategies = await self._generate_mitigation_strategies(
                critical_risks, identified_risks, jurisdiction
            )
            risk_assessment['risk_mitigation_strategies'] = mitigation_strategies
            
            # Phase 6: Risk Monitoring Recommendations
            monitoring_plan = self._create_risk_monitoring_plan(identified_risks)
            risk_assessment['monitoring_plan'] = monitoring_plan
            
            # Phase 7: Confidence Assessment
            risk_assessment['confidence_score'] = self._calculate_confidence_score(risk_assessment)
            
        except Exception as e:
            risk_assessment['error'] = str(e)
            risk_assessment['status'] = 'assessment_failed'
        
        return risk_assessment
    
    async def _analyze_risk_categories(self, document_text: str, jurisdiction: str,
                                     legal_domain: str) -> Dict[str, Any]:
        """Analyze risks across different categories"""
        
        category_results = {}
        
        for category, config in self.indian_risk_categories.items():
            category_prompt = f"""
            Analyze {category.replace('_', ' ')} in this legal document for {jurisdiction}:
            
            Document Type: {legal_domain}
            Document Content: {document_text[:2500]}
            
            Focus on these specific risk areas:
            {', '.join(config['subcategories'])}
            
            Provide detailed analysis in JSON format:
            {{
                "category_risk_score": 0-100,
                "identified_subcategory_risks": [
                    {{
                        "subcategory": "risk_subcategory",
                        "risk_level": "critical/high/medium/low",
                        "description": "Detailed risk description",
                        "likelihood": "high/medium/low",
                        "impact": "severe/moderate/minor",
                        "specific_provisions": "Relevant document sections"
                    }}
                ],
                "category_summary": "Overall category risk assessment",
                "immediate_concerns": ["concern1", "concern2"],
                "long_term_risks": ["risk1", "risk2"]
            }}
            """
            
            try:
                response = await self.gemini_model.generate_content_async(category_prompt)
                category_data = json.loads(response.text.strip('``````'))
                
                category_data['category_weight'] = config['weight']
                category_data['analysis_timestamp'] = datetime.now().isoformat()
                
                category_results[category] = category_data
                
            except Exception as e:
                category_results[category] = {
                    'error': str(e),
                    'category_risk_score': 50,  # Default moderate risk
                    'category_weight': config['weight'],
                    'identified_subcategory_risks': [],
                    'category_summary': f'Analysis failed for {category}: {str(e)}'
                }
        
        return category_results
    
    async def _identify_specific_risks(self, document_text: str, jurisdiction: str,
                                     legal_domain: str, category_analyses: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify specific risks with detailed analysis"""
        
        # Compile risks from category analysis
        all_risks = []
        
        for category, analysis in category_analyses.items():
            if 'identified_subcategory_risks' in analysis:
                for risk in analysis['identified_subcategory_risks']:
                    detailed_risk = {
                        'risk_id': f"{category}_{len(all_risks) + 1}",
                        'category': category,
                        'subcategory': risk.get('subcategory', 'general'),
                        'title': self._generate_risk_title(risk),
                        'description': risk.get('description', 'Risk identified'),
                        'severity': risk.get('risk_level', 'medium'),
                        'likelihood': risk.get('likelihood', 'medium'),
                        'impact': risk.get('impact', 'moderate'),
                        'specific_provisions': risk.get('specific_provisions', 'General document provisions'),
                        'quantified_score': self._quantify_risk_score(
                            risk.get('risk_level', 'medium'),
                            risk.get('likelihood', 'medium'),
                            risk.get('impact', 'moderate')
                        )
                    }
                    all_risks.append(detailed_risk)
        
        # Conduct additional deep-dive risk analysis
        additional_risks = await self._deep_dive_risk_analysis(
            document_text, jurisdiction, legal_domain
        )
        all_risks.extend(additional_risks)
        
        # Sort by quantified score
        all_risks.sort(key=lambda x: x['quantified_score'], reverse=True)
        
        return all_risks
    
    async def _deep_dive_risk_analysis(self, document_text: str, jurisdiction: str,
                                     legal_domain: str) -> List[Dict[str, Any]]:
        """Conduct deep-dive analysis for subtle or complex risks"""
        
        deep_dive_prompt = f"""
        Conduct deep-dive risk analysis for subtle and complex legal risks in this {legal_domain} document under {jurisdiction}:
        
        Document Content: {document_text[:3000]}
        
        Identify risks that might be overlooked in standard analysis:
        1. Hidden contractual obligations
        2. Ambiguous language creating interpretation disputes
        3. Regulatory compliance gaps not immediately obvious
        4. Cross-border jurisdiction issues
        5. Technology and digital law implications
        6. Future legal changes impact
        7. Third-party dependency risks
        8. Intellectual property vulnerabilities
        
        Return JSON array of complex risks:
        [
            {{
                "risk_type": "hidden_obligation/ambiguous_language/regulatory_gap/etc",
                "title": "Clear risk title",
                "description": "Detailed description of the subtle risk",
                "severity": "critical/high/medium/low",
                "likelihood": "high/medium/low",
                "impact": "severe/moderate/minor",
                "complexity_factor": "Why this risk is easily overlooked",
                "detection_difficulty": "high/medium/low"
            }}
        ]
        """
        
        try:
            response = await self.gemini_model.generate_content_async(deep_dive_prompt)
            additional_risks_data = json.loads(response.text.strip('``````'))
            
            processed_risks = []
            for i, risk in enumerate(additional_risks_data):
                processed_risk = {
                    'risk_id': f"deep_dive_{i + 1}",
                    'category': 'complex_analysis',
                    'subcategory': risk.get('risk_type', 'complex'),
                    'title': risk.get('title', 'Complex Risk Identified'),
                    'description': risk.get('description', 'Complex risk requiring attention'),
                    'severity': risk.get('severity', 'medium'),
                    'likelihood': risk.get('likelihood', 'medium'),
                    'impact': risk.get('impact', 'moderate'),
                    'complexity_factor': risk.get('complexity_factor', 'High complexity risk'),
                    'detection_difficulty': risk.get('detection_difficulty', 'medium'),
                    'quantified_score': self._quantify_risk_score(
                        risk.get('severity', 'medium'),
                        risk.get('likelihood', 'medium'),
                        risk.get('impact', 'moderate')
                    )
                }
                processed_risks.append(processed_risk)
            
            return processed_risks
            
        except Exception as e:
            return [{
                'risk_id': 'deep_dive_error',
                'category': 'analysis_limitation',
                'title': 'Deep Analysis Unavailable',
                'description': f'Complex risk analysis failed: {str(e)}. Manual expert review recommended.',
                'severity': 'medium',
                'quantified_score': 50
            }]
    
    def _evaluate_critical_risks(self, identified_risks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify and evaluate critical risks requiring immediate attention"""
        
        critical_risks = []
        
        for risk in identified_risks:
            # Critical risk criteria
            is_critical = (
                risk.get('severity') == 'critical' or
                (risk.get('severity') == 'high' and risk.get('likelihood') == 'high') or
                risk.get('quantified_score', 0) >= 85
            )
            
            if is_critical:
                critical_risk = risk.copy()
                critical_risk.update({
                    'criticality_reason': self._determine_criticality_reason(risk),
                    'immediate_actions_required': self._generate_immediate_actions(risk),
                    'escalation_required': True,
                    'business_impact': self._assess_business_impact(risk)
                })
                critical_risks.append(critical_risk)
        
        return critical_risks
    
    async def _generate_mitigation_strategies(self, critical_risks: List[Dict[str, Any]],
                                           all_risks: List[Dict[str, Any]],
                                           jurisdiction: str) -> List[Dict[str, Any]]:
        """Generate comprehensive risk mitigation strategies"""
        
        mitigation_prompt = f"""
        Generate comprehensive risk mitigation strategies for these legal risks under {jurisdiction}:
        
        Critical Risks ({len(critical_risks)}):
        {json.dumps([r['title'] + ': ' + r['description'][:100] for r in critical_risks], indent=2)}
        
        High-Priority Risks:
        {json.dumps([r['title'] + ': ' + r['description'][:100] for r in all_risks[:5]], indent=2)}
        
        Provide mitigation strategies in JSON format:
        [
            {{
                "risk_addressed": "Risk title/ID",
                "mitigation_strategy": "Specific mitigation approach",
                "implementation_steps": ["step1", "step2", "step3"],
                "timeline": "Implementation timeline",
                "cost_estimate": "rough cost estimate",
                "effectiveness": "high/medium/low",
                "legal_basis": "Legal foundation for mitigation",
                "responsible_party": "Who should implement",
                "monitoring_approach": "How to monitor effectiveness"
            }}
        ]
        """
        
        try:
            response = await self.gemini_model.generate_content_async(mitigation_prompt)
            mitigation_strategies = json.loads(response.text.strip('``````'))
            
            # Add metadata to each strategy
            for strategy in mitigation_strategies:
                strategy.update({
                    'strategy_id': f"mitigation_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(mitigation_strategies)}",
                    'jurisdiction': jurisdiction,
                    'created_timestamp': datetime.now().isoformat()
                })
            
            return mitigation_strategies
            
        except Exception as e:
            return [{
                'risk_addressed': 'All identified risks',
                'mitigation_strategy': f'Mitigation strategy generation failed: {str(e)}. Recommend immediate consultation with qualified legal counsel.',
                'implementation_steps': [
                    'Consult with legal experts',
                    'Conduct detailed risk assessment',
                    'Develop customized mitigation plan'
                ],
                'timeline': 'Immediate',
                'effectiveness': 'high',
                'responsible_party': 'Legal Team'
            }]
    
    def _create_risk_monitoring_plan(self, identified_risks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create ongoing risk monitoring plan"""
        
        monitoring_plan = {
            'monitoring_frequency': {},
            'key_indicators': [],
            'review_triggers': [],
            'reporting_schedule': {},
            'escalation_matrix': {}
        }
        
        # Monitoring frequency based on risk severity
        severity_counts = {}
        for risk in identified_risks:
            severity = risk.get('severity', 'medium')
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        if severity_counts.get('critical', 0) > 0:
            monitoring_plan['monitoring_frequency']['critical_risks'] = 'Weekly'
        if severity_counts.get('high', 0) > 0:
            monitoring_plan['monitoring_frequency']['high_risks'] = 'Bi-weekly'
        if severity_counts.get('medium', 0) > 0:
            monitoring_plan['monitoring_frequency']['medium_risks'] = 'Monthly'
        
        # Key indicators
        monitoring_plan['key_indicators'] = [
            'Regulatory compliance status changes',
            'Market conditions affecting contractual obligations',
            'Legal precedent developments',
            'Counterparty financial stability',
            'Technology and operational changes'
        ]
        
        # Review triggers
        monitoring_plan['review_triggers'] = [
            'Material change in business operations',
            'New regulatory announcements',
            'Significant legal precedent changes',
            'Counterparty disputes or issues',
            'External risk factor changes'
        ]
        
        return monitoring_plan
    
    def _calculate_overall_risk_score(self, category_analyses: Dict[str, Any],
                                    identified_risks: List[Dict[str, Any]]) -> tuple:
        """Calculate weighted overall risk score and level"""
        
        # Method 1: Category-weighted approach
        category_weighted_score = 0
        total_weight = 0
        
        for category, analysis in category_analyses.items():
            if 'category_risk_score' in analysis and 'category_weight' in analysis:
                category_weighted_score += analysis['category_risk_score'] * analysis['category_weight']
                total_weight += analysis['category_weight']
        
        if total_weight > 0:
            category_score = category_weighted_score / total_weight
        else:
            category_score = 50  # Default moderate risk
        
        # Method 2: Individual risk aggregation
        if identified_risks:
            risk_scores = [risk.get('quantified_score', 50) for risk in identified_risks]
            individual_risk_score = np.mean(risk_scores)
            
            # Weight towards higher risks
            high_risk_count = sum(1 for score in risk_scores if score >= 75)
            if high_risk_count > 0:
                high_risk_penalty = min(high_risk_count * 5, 20)  # Up to 20 points penalty
                individual_risk_score += high_risk_penalty
        else:
            individual_risk_score = 50
        
        # Combine both methods
        overall_score = (category_score * 0.6 + individual_risk_score * 0.4)
        overall_score = min(max(overall_score, 0), 100)  # Clamp to 0-100
        
        # Determine risk level
        if overall_score >= 85:
            risk_level = 'critical'
        elif overall_score >= 70:
            risk_level = 'high'
        elif overall_score >= 40:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        return round(overall_score, 1), risk_level
    
    def _quantify_risk_score(self, severity: str, likelihood: str, impact: str) -> int:
        """Quantify individual risk score based on severity, likelihood, and impact"""
        
        severity_scores = {'critical': 100, 'high': 75, 'medium': 50, 'low': 25}
        likelihood_multipliers = {'high': 1.0, 'medium': 0.7, 'low': 0.4}
        impact_multipliers = {'severe': 1.0, 'moderate': 0.8, 'minor': 0.6}
        
        base_score = severity_scores.get(severity, 50)
        likelihood_mult = likelihood_multipliers.get(likelihood, 0.7)
        impact_mult = impact_multipliers.get(impact, 0.8)
        
        quantified_score = base_score * likelihood_mult * impact_mult
        return min(max(int(quantified_score), 0), 100)
    
    def _generate_risk_title(self, risk_data: Dict[str, Any]) -> str:
        """Generate clear, actionable risk title"""
        
        subcategory = risk_data.get('subcategory', 'legal_risk')
        severity = risk_data.get('risk_level', 'medium')
        
        # Create human-readable titles
        title_templates = {
            'breach_of_contract': 'Contract Breach Risk',
            'regulatory_compliance': 'Regulatory Compliance Gap',
            'litigation_risk': 'Potential Litigation Exposure',
            'financial_exposure': 'Financial Liability Risk',
            'operational_disruption': 'Business Operations Risk'
        }
        
        base_title = title_templates.get(subcategory, subcategory.replace('_', ' ').title())
        
        if severity == 'critical':
            return f"CRITICAL: {base_title}"
        elif severity == 'high':
            return f"HIGH RISK: {base_title}"
        else:
            return base_title
    
    def _determine_criticality_reason(self, risk: Dict[str, Any]) -> str:
        """Determine why a risk is considered critical"""
        
        severity = risk.get('severity')
        likelihood = risk.get('likelihood')
        impact = risk.get('impact')
        score = risk.get('quantified_score', 0)
        
        reasons = []
        
        if severity == 'critical':
            reasons.append("classified as critical severity")
        if likelihood == 'high' and severity == 'high':
            reasons.append("high likelihood of high-severity risk")
        if score >= 85:
            reasons.append(f"quantified risk score of {score}/100")
        if 'regulatory' in risk.get('category', ''):
            reasons.append("regulatory compliance implications")
        
        return "; ".join(reasons) if reasons else "exceeds critical risk thresholds"
    
    def _generate_immediate_actions(self, risk: Dict[str, Any]) -> List[str]:
        """Generate immediate actions for critical risks"""
        
        actions = []
        category = risk.get('category', '')
        
        if 'regulatory' in category:
            actions.extend([
                "Review current compliance status immediately",
                "Consult with regulatory compliance experts",
                "Prepare corrective action plan"
            ])
        elif 'contractual' in category:
            actions.extend([
                "Review contract terms with legal counsel",
                "Assess breach probability and impact",
                "Prepare risk mitigation measures"
            ])
        elif 'litigation' in category:
            actions.extend([
                "Engage litigation counsel for assessment",
                "Prepare documentary evidence",
                "Consider settlement alternatives"
            ])
        else:
            actions.extend([
                "Conduct immediate detailed risk assessment",
                "Engage appropriate legal expertise",
                "Implement temporary risk controls"
            ])
        
        return actions
    
    def _assess_business_impact(self, risk: Dict[str, Any]) -> Dict[str, str]:
        """Assess business impact of critical risk"""
        
        impact = risk.get('impact', 'moderate')
        category = risk.get('category', '')
        
        business_impact = {
            'financial': 'moderate',
            'operational': 'moderate',
            'reputational': 'low',
            'strategic': 'low'
        }
        
        if impact == 'severe':
            business_impact.update({
                'financial': 'high',
                'operational': 'high',
                'reputational': 'moderate'
            })
        elif impact == 'moderate':
            business_impact.update({
                'financial': 'moderate',
                'operational': 'moderate'
            })
        
        # Category-specific adjustments
        if 'regulatory' in category:
            business_impact['reputational'] = 'high'
        elif 'financial' in category:
            business_impact['financial'] = 'high'
        
        return business_impact
    
    def _calculate_confidence_score(self, risk_assessment: Dict[str, Any]) -> float:
        """Calculate confidence score for risk assessment"""
        
        confidence_factors = []
        
        # Category analysis completeness
        risk_categories = risk_assessment.get('risk_categories', {})
        successful_categories = sum(1 for cat in risk_categories.values() if 'error' not in cat)
        total_categories = len(self.indian_risk_categories)
        category_confidence = successful_categories / total_categories if total_categories > 0 else 0
        confidence_factors.append(category_confidence)
        
        # Risk identification depth
        identified_risks = risk_assessment.get('identified_risks', [])
        risk_depth_score = min(len(identified_risks) / 10.0, 1.0)  # Expect ~10 risks for good analysis
        confidence_factors.append(risk_depth_score)
        
        # Mitigation strategy availability
        mitigation_strategies = risk_assessment.get('risk_mitigation_strategies', [])
        mitigation_confidence = 1.0 if mitigation_strategies else 0.3
        confidence_factors.append(mitigation_confidence)
        
        # Overall analysis success
        analysis_success = 1.0 if 'error' not in risk_assessment else 0.2
        confidence_factors.append(analysis_success)
        
        return sum(confidence_factors) / len(confidence_factors)
