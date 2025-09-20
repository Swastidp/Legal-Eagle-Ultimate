import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import google.generativeai as genai
import numpy as np

class OutcomePredictorAgent:
    """Specialized agent for predicting legal outcomes using Indian legal precedents"""
    
    def __init__(self, gemini_pro_model):
        self.gemini_pro = gemini_pro_model
        
        # Indian legal system characteristics
        self.indian_court_system = {
            'supreme_court': {
                'jurisdiction': 'Constitutional and appeal matters',
                'precedent_weight': 1.0,
                'case_complexity': 'high',
                'average_timeline': '2-5 years'
            },
            'high_courts': {
                'jurisdiction': 'State-level appeals and constitutional matters',
                'precedent_weight': 0.8,
                'case_complexity': 'medium-high',
                'average_timeline': '1-3 years'
            },
            'district_courts': {
                'jurisdiction': 'Original civil and criminal matters',
                'precedent_weight': 0.6,
                'case_complexity': 'medium',
                'average_timeline': '6 months - 2 years'
            },
            'specialized_tribunals': {
                'jurisdiction': 'Domain-specific matters (NCLT, NCLAT, etc.)',
                'precedent_weight': 0.7,
                'case_complexity': 'medium',
                'average_timeline': '6 months - 1 year'
            }
        }
        
        # Outcome prediction factors for Indian legal system
        self.prediction_factors = {
            'legal_merit': {
                'weight': 0.30,
                'components': ['statutory_basis', 'precedent_support', 'legal_principles']
            },
            'factual_strength': {
                'weight': 0.25,
                'components': ['evidence_quality', 'witness_credibility', 'documentation']
            },
            'procedural_compliance': {
                'weight': 0.15,
                'components': ['jurisdiction', 'limitation_period', 'proper_notice']
            },
            'judicial_trends': {
                'weight': 0.15,
                'components': ['recent_judgments', 'court_inclination', 'policy_considerations']
            },
            'strategic_factors': {
                'weight': 0.15,
                'components': ['timing', 'forum_selection', 'alternative_remedies']
            }
        }
        
        # Indian legal domains and their success patterns
        self.domain_success_patterns = {
            'contract_disputes': {'base_success_rate': 0.65, 'factors': ['clear_terms', 'performance_evidence']},
            'corporate_law': {'base_success_rate': 0.70, 'factors': ['compliance_history', 'documentation']},
            'employment_law': {'base_success_rate': 0.60, 'factors': ['procedural_compliance', 'just_cause']},
            'property_disputes': {'base_success_rate': 0.55, 'factors': ['title_clarity', 'possession_evidence']},
            'regulatory_compliance': {'base_success_rate': 0.75, 'factors': ['good_faith', 'corrective_action']},
            'intellectual_property': {'base_success_rate': 0.68, 'factors': ['registration_status', 'prior_use']}
        }
    
    async def predict_legal_outcomes(self, document_text: str, jurisdiction: str,
                                   legal_domain: str = "general",
                                   case_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Comprehensive legal outcome prediction"""
        
        prediction_result = {
            'prediction_id': f"outcome_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'timestamp': datetime.now().isoformat(),
            'jurisdiction': jurisdiction,
            'legal_domain': legal_domain,
            'case_context': case_context or {},
            'success_probability': 0.5,
            'confidence_level': 'medium',
            'prediction_factors': {},
            'scenario_analysis': {},
            'strategic_recommendations': [],
            'timeline_predictions': {},
            'cost_estimates': {},
            'alternative_outcomes': []
        }
        
        try:
            # Phase 1: Legal Merit Analysis
            legal_merit = await self._analyze_legal_merit(document_text, jurisdiction, legal_domain)
            prediction_result['prediction_factors']['legal_merit'] = legal_merit
            
            # Phase 2: Precedent Analysis
            precedent_analysis = await self._analyze_precedents(document_text, jurisdiction, legal_domain)
            prediction_result['prediction_factors']['precedent_analysis'] = precedent_analysis
            
            # Phase 3: Factual Strength Assessment
            factual_strength = await self._assess_factual_strength(document_text, legal_domain)
            prediction_result['prediction_factors']['factual_strength'] = factual_strength
            
            # Phase 4: Procedural Analysis
            procedural_analysis = self._analyze_procedural_factors(document_text, jurisdiction)
            prediction_result['prediction_factors']['procedural_analysis'] = procedural_analysis
            
            # Phase 5: Success Probability Calculation
            success_prob, confidence = self._calculate_success_probability(
                prediction_result['prediction_factors'], legal_domain
            )
            prediction_result['success_probability'] = success_prob
            prediction_result['confidence_level'] = confidence
            
            # Phase 6: Scenario Analysis
            scenario_analysis = await self._conduct_scenario_analysis(
                document_text, success_prob, legal_domain, jurisdiction
            )
            prediction_result['scenario_analysis'] = scenario_analysis
            
            # Phase 7: Strategic Recommendations
            strategic_recs = await self._generate_strategic_recommendations(
                prediction_result, document_text, jurisdiction
            )
            prediction_result['strategic_recommendations'] = strategic_recs
            
            # Phase 8: Timeline and Cost Predictions
            timeline_pred = self._predict_timeline(legal_domain, jurisdiction, success_prob)
            prediction_result['timeline_predictions'] = timeline_pred
            
            cost_estimates = self._estimate_costs(legal_domain, timeline_pred, success_prob)
            prediction_result['cost_estimates'] = cost_estimates
            
            # Phase 9: Alternative Outcomes
            alternatives = await self._analyze_alternative_outcomes(
                document_text, prediction_result, jurisdiction
            )
            prediction_result['alternative_outcomes'] = alternatives
            
        except Exception as e:
            prediction_result['error'] = str(e)
            prediction_result['status'] = 'prediction_failed'
        
        return prediction_result
    
    async def _analyze_legal_merit(self, document_text: str, jurisdiction: str, legal_domain: str) -> Dict[str, Any]:
        """Analyze legal merit and strength of case"""
        
        legal_merit_prompt = f"""
        Analyze the legal merit and strength of this case under {jurisdiction} law in {legal_domain}:
        
        Document/Case Details: {document_text[:3000]}
        
        Evaluate the following aspects:
        1. Statutory basis and legal foundation
        2. Applicable legal principles and doctrines
        3. Strength of legal arguments
        4. Potential legal defenses or challenges
        5. Clarity of legal rights and obligations
        6. Compliance with procedural requirements
        
        Provide analysis in JSON format:
        {{
            "statutory_basis_score": 0-100,
            "legal_principle_strength": 0-100,
            "argument_quality": 0-100,
            "overall_merit_score": 0-100,
            "strong_points": ["strength1", "strength2"],
            "weak_points": ["weakness1", "weakness2"],
            "legal_challenges": ["challenge1", "challenge2"],
            "applicable_statutes": ["statute1", "statute2"],
            "legal_basis_assessment": "detailed assessment"
        }}
        """
        
        try:
            response = await self.gemini_pro.generate_content_async(legal_merit_prompt)
            merit_data = json.loads(response.text.strip('``````'))
            
            merit_data['analysis_timestamp'] = datetime.now().isoformat()
            merit_data['jurisdiction'] = jurisdiction
            
            return merit_data
            
        except Exception as e:
            return {
                'error': str(e),
                'overall_merit_score': 50,
                'statutory_basis_score': 50,
                'legal_principle_strength': 50,
                'argument_quality': 50,
                'strong_points': ['Legal analysis unavailable'],
                'weak_points': ['Detailed assessment required'],
                'legal_basis_assessment': f'Merit analysis failed: {str(e)}'
            }
    
    async def _analyze_precedents(self, document_text: str, jurisdiction: str, legal_domain: str) -> Dict[str, Any]:
        """Analyze relevant precedents and their impact on outcome"""
        
        precedent_prompt = f"""
        Analyze relevant legal precedents for this {legal_domain} matter under {jurisdiction}:
        
        Case Context: {document_text[:2500]}
        
        Focus on:
        1. Supreme Court of India binding precedents
        2. Relevant High Court decisions
        3. Recent judicial trends and interpretations
        4. Conflicting precedents and their resolution
        5. Evolution of legal principles in this area
        6. Precedent applicability to current facts
        
        Provide precedent analysis in JSON format:
        {{
            "binding_precedents": [
                {{
                    "case_name": "Citation",
                    "year": "YYYY",
                    "court": "Court name",
                    "legal_principle": "Principle established",
                    "factual_similarity": 0-100,
                    "precedent_strength": "strong/moderate/weak",
                    "favorable_outcome": true/false
                }}
            ],
            "precedent_support_score": 0-100,
            "conflicting_precedents": "Analysis of conflicts",
            "recent_trends": ["trend1", "trend2"],
            "precedent_summary": "Overall precedent position"
        }}
        """
        
        try:
            response = await self.gemini_pro.generate_content_async(precedent_prompt)
            precedent_data = json.loads(response.text.strip('``````'))
            
            precedent_data['analysis_timestamp'] = datetime.now().isoformat()
            precedent_data['jurisdiction'] = jurisdiction
            
            return precedent_data
            
        except Exception as e:
            return {
                'error': str(e),
                'precedent_support_score': 50,
                'binding_precedents': [],
                'conflicting_precedents': f'Precedent analysis failed: {str(e)}',
                'recent_trends': ['Analysis unavailable'],
                'precedent_summary': 'Manual precedent research recommended'
            }
    
    async def _assess_factual_strength(self, document_text: str, legal_domain: str) -> Dict[str, Any]:
        """Assess strength of factual case and evidence"""
        
        factual_prompt = f"""
        Assess the factual strength and evidence quality for this {legal_domain} matter:
        
        Case Facts and Evidence: {document_text[:2500]}
        
        Evaluate:
        1. Quality and reliability of evidence
        2. Completeness of factual record
        3. Witness credibility and availability
        4. Documentary evidence strength
        5. Potential factual disputes
        6. Evidence gaps and their impact
        
        Provide factual assessment in JSON format:
        {{
            "evidence_quality_score": 0-100,
            "factual_completeness": 0-100,
            "documentation_strength": 0-100,
            "overall_factual_score": 0-100,
            "strong_evidence": ["evidence1", "evidence2"],
            "evidence_gaps": ["gap1", "gap2"],
            "factual_disputes": ["dispute1", "dispute2"],
            "evidence_recommendations": ["rec1", "rec2"]
        }}
        """
        
        try:
            response = await self.gemini_pro.generate_content_async(factual_prompt)
            factual_data = json.loads(response.text.strip('``````'))
            
            factual_data['assessment_timestamp'] = datetime.now().isoformat()
            
            return factual_data
            
        except Exception as e:
            return {
                'error': str(e),
                'overall_factual_score': 50,
                'evidence_quality_score': 50,
                'factual_completeness': 50,
                'documentation_strength': 50,
                'strong_evidence': ['Assessment unavailable'],
                'evidence_gaps': ['Detailed review required'],
                'evidence_recommendations': ['Consult with legal counsel for evidence evaluation']
            }
    
    def _analyze_procedural_factors(self, document_text: str, jurisdiction: str) -> Dict[str, Any]:
        """Analyze procedural compliance and factors"""
        
        # Simplified procedural analysis (could be enhanced with Gemini)
        procedural_factors = {
            'jurisdiction_compliance': 75,  # Default moderate compliance
            'limitation_period_status': 'within_limits',
            'procedural_compliance_score': 70,
            'forum_appropriateness': 80,
            'notice_requirements': 'satisfied',
            'procedural_risks': ['Standard procedural considerations apply'],
            'procedural_recommendations': [
                'Ensure all procedural requirements are met',
                'Verify jurisdiction and venue appropriateness',
                'Confirm limitation periods compliance'
            ]
        }
        
        # Basic analysis of document for procedural indicators
        doc_lower = document_text.lower()
        
        # Check for limitation period concerns
        if any(term in doc_lower for term in ['old', 'years ago', 'long time', 'delay']):
            procedural_factors['limitation_period_status'] = 'requires_review'
            procedural_factors['procedural_compliance_score'] -= 10
        
        # Check for jurisdiction indicators
        if any(court in doc_lower for court in ['supreme court', 'high court', 'district court']):
            procedural_factors['forum_appropriateness'] = 90
        
        procedural_factors['analysis_timestamp'] = datetime.now().isoformat()
        
        return procedural_factors
    
    def _calculate_success_probability(self, prediction_factors: Dict[str, Any], legal_domain: str) -> Tuple[float, str]:
        """Calculate overall success probability"""
        
        # Base success rate for domain
        domain_pattern = self.domain_success_patterns.get(legal_domain, {'base_success_rate': 0.6})
        base_rate = domain_pattern['base_success_rate']
        
        # Extract scores from prediction factors
        scores = []
        
        legal_merit = prediction_factors.get('legal_merit', {})
        if 'overall_merit_score' in legal_merit:
            scores.append(legal_merit['overall_merit_score'] / 100.0)
        
        precedent_analysis = prediction_factors.get('precedent_analysis', {})
        if 'precedent_support_score' in precedent_analysis:
            scores.append(precedent_analysis['precedent_support_score'] / 100.0)
        
        factual_strength = prediction_factors.get('factual_strength', {})
        if 'overall_factual_score' in factual_strength:
            scores.append(factual_strength['overall_factual_score'] / 100.0)
        
        procedural_analysis = prediction_factors.get('procedural_analysis', {})
        if 'procedural_compliance_score' in procedural_analysis:
            scores.append(procedural_analysis['procedural_compliance_score'] / 100.0)
        
        # Calculate weighted average
        if scores:
            factor_weights = [0.35, 0.30, 0.25, 0.10]  # Weights for merit, precedent, factual, procedural
            weighted_score = sum(s * w for s, w in zip(scores[:4], factor_weights))
            
            # Combine with base rate
            success_probability = (base_rate * 0.3) + (weighted_score * 0.7)
        else:
            success_probability = base_rate
        
        # Clamp to reasonable range
        success_probability = max(0.05, min(0.95, success_probability))
        
        # Determine confidence level
        if len(scores) >= 3 and all(s > 0.3 for s in scores):
            confidence = 'high'
        elif len(scores) >= 2:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        return round(success_probability, 3), confidence
    
    async def _conduct_scenario_analysis(self, document_text: str, success_prob: float,
                                       legal_domain: str, jurisdiction: str) -> Dict[str, Any]:
        """Conduct scenario analysis for different outcomes"""
        
        scenario_prompt = f"""
        Conduct scenario analysis for this {legal_domain} case in {jurisdiction} with {success_prob:.1%} success probability:
        
        Case Overview: {document_text[:2000]}
        
        Analyze these scenarios:
        1. Best case scenario - everything goes favorably
        2. Most likely scenario - based on success probability
        3. Worst case scenario - adverse outcome
        4. Settlement scenario - negotiated resolution
        
        Provide analysis in JSON format:
        {{
            "best_case": {{
                "outcome": "description",
                "probability": 0.0-1.0,
                "benefits": ["benefit1", "benefit2"],
                "timeline": "estimated time",
                "requirements": ["requirement1", "requirement2"]
            }},
            "most_likely": {{
                "outcome": "description",
                "probability": 0.0-1.0,
                "mixed_results": ["result1", "result2"],
                "timeline": "estimated time"
            }},
            "worst_case": {{
                "outcome": "description",
                "probability": 0.0-1.0,
                "consequences": ["consequence1", "consequence2"],
                "mitigation": ["mitigation1", "mitigation2"]
            }},
            "settlement": {{
                "likelihood": 0.0-1.0,
                "optimal_timing": "when to settle",
                "expected_terms": ["term1", "term2"],
                "advantages": ["advantage1", "advantage2"]
            }}
        }}
        """
        
        try:
            response = await self.gemini_pro.generate_content_async(scenario_prompt)
            scenario_data = json.loads(response.text.strip('``````'))
            
            scenario_data['analysis_timestamp'] = datetime.now().isoformat()
            
            return scenario_data
            
        except Exception as e:
            return {
                'error': str(e),
                'best_case': {'outcome': 'Scenario analysis unavailable', 'probability': success_prob * 1.2},
                'most_likely': {'outcome': 'Mixed results expected', 'probability': success_prob},
                'worst_case': {'outcome': 'Adverse outcome possible', 'probability': 1 - success_prob},
                'settlement': {'likelihood': 0.6, 'optimal_timing': 'Early stages preferred'}
            }
    
    async def _generate_strategic_recommendations(self, prediction_result: Dict[str, Any],
                                               document_text: str, jurisdiction: str) -> List[Dict[str, Any]]:
        """Generate strategic recommendations based on prediction"""
        
        success_prob = prediction_result.get('success_probability', 0.5)
        legal_domain = prediction_result.get('legal_domain', 'general')
        
        recommendations = []
        
        # High success probability strategies
        if success_prob > 0.7:
            recommendations.append({
                'category': 'Aggressive Strategy',
                'priority': 'High',
                'recommendation': 'Pursue case aggressively given strong success probability',
                'rationale': f'With {success_prob:.1%} success probability, strong position for favorable outcome',
                'timeline': 'Immediate',
                'expected_benefit': 'Maximize favorable outcome potential'
            })
        
        # Medium success probability strategies
        elif success_prob > 0.4:
            recommendations.append({
                'category': 'Balanced Strategy',
                'priority': 'Medium',
                'recommendation': 'Pursue case while preparing settlement alternatives',
                'rationale': f'Moderate {success_prob:.1%} success probability suggests balanced approach',
                'timeline': '1-2 weeks',
                'expected_benefit': 'Optimize outcome while managing risk'
            })
        
        # Low success probability strategies
        else:
            recommendations.append({
                'category': 'Risk Management',
                'priority': 'High',
                'recommendation': 'Focus on settlement negotiations and risk mitigation',
                'rationale': f'Lower {success_prob:.1%} success probability suggests settlement focus',
                'timeline': 'Immediate',
                'expected_benefit': 'Minimize losses and avoid adverse outcome'
            })
        
        # Evidence strengthening recommendations
        factual_strength = prediction_result.get('prediction_factors', {}).get('factual_strength', {})
        if factual_strength.get('overall_factual_score', 50) < 70:
            recommendations.append({
                'category': 'Evidence Enhancement',
                'priority': 'High',
                'recommendation': 'Strengthen evidence base before proceeding',
                'rationale': 'Factual case needs strengthening for better outcome probability',
                'timeline': '2-4 weeks',
                'expected_benefit': 'Improved success probability through stronger evidence'
            })
        
        # Procedural recommendations
        procedural_factors = prediction_result.get('prediction_factors', {}).get('procedural_analysis', {})
        if procedural_factors.get('procedural_compliance_score', 70) < 80:
            recommendations.append({
                'category': 'Procedural Compliance',
                'priority': 'Medium',
                'recommendation': 'Address procedural compliance issues',
                'rationale': 'Procedural weaknesses could undermine case',
                'timeline': '1 week',
                'expected_benefit': 'Avoid procedural dismissal or complications'
            })
        
        return recommendations
    
    def _predict_timeline(self, legal_domain: str, jurisdiction: str, success_prob: float) -> Dict[str, Any]:
        """Predict case timeline based on domain and jurisdiction"""
        
        # Base timeline estimates for Indian courts
        base_timelines = {
            'contract_disputes': {'min': 6, 'max': 18, 'average': 12},
            'corporate_law': {'min': 8, 'max': 24, 'average': 16},
            'employment_law': {'min': 4, 'max': 12, 'average': 8},
            'property_disputes': {'min': 12, 'max': 36, 'average': 24},
            'regulatory_compliance': {'min': 3, 'max': 12, 'average': 6}
        }
        
        timeline_data = base_timelines.get(legal_domain, {'min': 6, 'max': 18, 'average': 12})
        
        # Adjust based on success probability (weaker cases may take longer)
        if success_prob < 0.3:
            timeline_data = {k: int(v * 1.3) for k, v in timeline_data.items()}
        elif success_prob > 0.8:
            timeline_data = {k: int(v * 0.8) for k, v in timeline_data.items()}
        
        return {
            'estimated_duration_months': timeline_data,
            'key_milestones': [
                {'phase': 'Case Filing', 'timeline': '1-2 weeks'},
                {'phase': 'Initial Hearings', 'timeline': '1-3 months'},
                {'phase': 'Evidence Phase', 'timeline': f"{timeline_data['average']//2} months"},
                {'phase': 'Final Arguments', 'timeline': f"{timeline_data['average']-2} months"},
                {'phase': 'Judgment', 'timeline': f"{timeline_data['average']} months"}
            ],
            'factors_affecting_timeline': [
                'Court backlog and scheduling',
                'Complexity of evidence presentation',
                'Number of witnesses and experts',
                'Settlement negotiations timeline',
                'Appeals possibility'
            ]
        }
    
    def _estimate_costs(self, legal_domain: str, timeline_pred: Dict[str, Any], success_prob: float) -> Dict[str, Any]:
        """Estimate legal costs based on case characteristics"""
        
        avg_duration = timeline_pred.get('estimated_duration_months', {}).get('average', 12)
        
        # Base cost estimates for Indian legal system
        base_costs = {
            'lawyer_fees': {
                'junior_counsel': 50000 * avg_duration,  # Rs. 50k per month
                'senior_counsel': 150000 * avg_duration,  # Rs. 1.5L per month
                'specialized_counsel': 200000 * avg_duration  # Rs. 2L per month
            },
            'court_fees': max(5000, avg_duration * 2000),  # Minimum Rs. 5k
            'documentation': 25000,  # Rs. 25k for documentation
            'expert_witnesses': 50000,  # Rs. 50k for experts
            'miscellaneous': 30000  # Rs. 30k miscellaneous
        }
        
        # Calculate total ranges
        min_cost = (base_costs['lawyer_fees']['junior_counsel'] + 
                   base_costs['court_fees'] + 
                   base_costs['documentation'])
        
        max_cost = (base_costs['lawyer_fees']['specialized_counsel'] + 
                   base_costs['court_fees'] + 
                   base_costs['documentation'] + 
                   base_costs['expert_witnesses'] + 
                   base_costs['miscellaneous'])
        
        avg_cost = (min_cost + max_cost) // 2
        
        # Adjust based on success probability (higher investment in stronger cases)
        if success_prob > 0.7:
            recommended_investment = max_cost * 0.8  # Invest more in strong cases
        elif success_prob < 0.3:
            recommended_investment = min_cost * 1.2  # Conservative investment in weak cases
        else:
            recommended_investment = avg_cost
        
        return {
            'cost_estimates_inr': {
                'minimum': int(min_cost),
                'maximum': int(max_cost),
                'average': int(avg_cost),
                'recommended': int(recommended_investment)
            },
            'cost_breakdown': base_costs,
            'cost_factors': [
                'Case complexity and duration',
                'Choice of legal counsel seniority',
                'Number of court hearings required',
                'Expert witness requirements',
                'Documentation and evidence costs'
            ],
            'cost_optimization_tips': [
                'Consider settlement to reduce costs',
                'Use junior counsel for routine matters',
                'Prepare comprehensive documentation upfront',
                'Explore alternative dispute resolution'
            ]
        }
    
    async def _analyze_alternative_outcomes(self, document_text: str, prediction_result: Dict[str, Any],
                                          jurisdiction: str) -> List[Dict[str, Any]]:
        """Analyze alternative dispute resolution and outcome options"""
        
        alternatives_prompt = f"""
        Analyze alternative dispute resolution options for this case in {jurisdiction}:
        
        Case Overview: {document_text[:2000]}
        Success Probability: {prediction_result.get('success_probability', 0.5):.1%}
        
        Evaluate these alternatives:
        1. Mediation
        2. Arbitration
        3. Conciliation
        4. Lok Adalat (if applicable)
        5. Settlement negotiations
        6. Regulatory resolution (if applicable)
        
        Provide analysis in JSON format:
        [
            {{
                "alternative": "option name",
                "suitability": 0-100,
                "advantages": ["advantage1", "advantage2"],
                "disadvantages": ["disadvantage1", "disadvantage2"],
                "timeline": "estimated timeline",
                "cost_comparison": "vs litigation costs",
                "success_likelihood": 0.0-1.0,
                "requirements": ["requirement1", "requirement2"],
                "recommendation": "strong/moderate/weak recommendation"
            }}
        ]
        """
        
        try:
            response = await self.gemini_pro.generate_content_async(alternatives_prompt)
            alternatives_data = json.loads(response.text.strip('``````'))
            
            # Add metadata to each alternative
            for alt in alternatives_data:
                alt['analysis_timestamp'] = datetime.now().isoformat()
                alt['jurisdiction'] = jurisdiction
            
            return alternatives_data
            
        except Exception as e:
            return [
                {
                    'alternative': 'Settlement Negotiation',
                    'suitability': 70,
                    'advantages': ['Cost-effective', 'Faster resolution', 'Confidential'],
                    'disadvantages': ['May not get full compensation', 'Depends on other party cooperation'],
                    'timeline': '1-3 months',
                    'cost_comparison': '70-80% less than litigation',
                    'success_likelihood': 0.6,
                    'requirements': ['Willing parties', 'Good faith negotiation'],
                    'recommendation': 'moderate',
                    'error': f'Alternative analysis failed: {str(e)}'
                },
                {
                    'alternative': 'Mediation',
                    'suitability': 60,
                    'advantages': ['Neutral mediator', 'Preserves relationships', 'Flexible solutions'],
                    'disadvantages': ['Non-binding', 'Requires cooperation'],
                    'timeline': '2-4 months',
                    'cost_comparison': '60-70% less than litigation',
                    'success_likelihood': 0.5,
                    'requirements': ['Neutral mediator', 'Willingness to compromise'],
                    'recommendation': 'moderate',
                    'error': f'Alternative analysis failed: {str(e)}'
                }
            ]
    
    def generate_outcome_summary(self, prediction_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary of outcome prediction"""
        
        success_prob = prediction_result.get('success_probability', 0.5)
        confidence_level = prediction_result.get('confidence_level', 'medium')
        legal_domain = prediction_result.get('legal_domain', 'general')
        
        # Determine overall recommendation
        if success_prob > 0.8 and confidence_level == 'high':
            recommendation = 'Strongly Recommend Proceeding'
            risk_level = 'Low Risk'
        elif success_prob > 0.6:
            recommendation = 'Recommend Proceeding with Caution'
            risk_level = 'Medium Risk'
        elif success_prob > 0.4:
            recommendation = 'Consider Settlement Options'
            risk_level = 'Medium-High Risk'
        else:
            recommendation = 'Recommend Settlement or Alternative Resolution'
            risk_level = 'High Risk'
        
        # Extract key factors
        prediction_factors = prediction_result.get('prediction_factors', {})
        
        strongest_factors = []
        weakest_factors = []
        
        for factor_name, factor_data in prediction_factors.items():
            if isinstance(factor_data, dict):
                # Look for score fields
                score_fields = [k for k in factor_data.keys() if 'score' in k.lower()]
                if score_fields:
                    avg_score = np.mean([factor_data[field] for field in score_fields if isinstance(factor_data[field], (int, float))])
                    if avg_score > 70:
                        strongest_factors.append(factor_name.replace('_', ' ').title())
                    elif avg_score < 50:
                        weakest_factors.append(factor_name.replace('_', ' ').title())
        
        # Timeline and cost summary
        timeline_pred = prediction_result.get('timeline_predictions', {})
        cost_estimates = prediction_result.get('cost_estimates', {})
        
        avg_duration = timeline_pred.get('estimated_duration_months', {}).get('average', 12)
        avg_cost = cost_estimates.get('cost_estimates_inr', {}).get('average', 500000)
        
        return {
            'executive_summary': {
                'success_probability': f"{success_prob:.1%}",
                'confidence_level': confidence_level,
                'overall_recommendation': recommendation,
                'risk_assessment': risk_level,
                'expected_timeline': f"{avg_duration} months",
                'estimated_cost': f"₹{avg_cost:,}"
            },
            'key_strengths': strongest_factors[:3],
            'key_weaknesses': weakest_factors[:3],
            'critical_success_factors': [
                'Strong legal merit and statutory basis',
                'Favorable precedent support',
                'Quality evidence and documentation',
                'Procedural compliance and timing'
            ],
            'immediate_priorities': self._generate_immediate_priorities(prediction_result),
            'decision_framework': {
                'proceed_indicators': [
                    f"Success probability > 60% (Current: {success_prob:.1%})",
                    f"High confidence in analysis (Current: {confidence_level})",
                    "Strong legal merit and precedent support",
                    "Adequate evidence and documentation"
                ],
                'settlement_indicators': [
                    "Success probability < 40%",
                    "Weak factual case or evidence gaps",
                    "Adverse precedents or legal challenges",
                    "High cost-benefit ratio concerns"
                ]
            }
        }
    
    def _generate_immediate_priorities(self, prediction_result: Dict[str, Any]) -> List[str]:
        """Generate immediate priority actions based on prediction"""
        
        priorities = []
        success_prob = prediction_result.get('success_probability', 0.5)
        prediction_factors = prediction_result.get('prediction_factors', {})
        
        # Evidence strengthening priorities
        factual_strength = prediction_factors.get('factual_strength', {})
        if factual_strength.get('overall_factual_score', 50) < 65:
            priorities.append("Strengthen evidence base and documentation")
        
        # Legal merit enhancement
        legal_merit = prediction_factors.get('legal_merit', {})
        if legal_merit.get('overall_merit_score', 50) < 65:
            priorities.append("Conduct detailed legal research and strengthen legal arguments")
        
        # Precedent analysis
        precedent_analysis = prediction_factors.get('precedent_analysis', {})
        if precedent_analysis.get('precedent_support_score', 50) < 60:
            priorities.append("Research additional supporting precedents and case law")
        
        # Procedural compliance
        procedural_analysis = prediction_factors.get('procedural_analysis', {})
        if procedural_analysis.get('procedural_compliance_score', 70) < 75:
            priorities.append("Address procedural compliance issues")
        
        # Settlement consideration
        if success_prob < 0.4:
            priorities.append("Explore settlement negotiations and alternative dispute resolution")
        
        # Strategic planning
        if success_prob > 0.7:
            priorities.append("Develop aggressive litigation strategy to maximize favorable outcome")
        else:
            priorities.append("Develop balanced strategy with settlement backup plan")
        
        return priorities[:5]  # Return top 5 priorities
    
    def compare_scenarios(self, prediction_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare multiple legal scenarios or strategies"""
        
        if len(prediction_results) < 2:
            return {'error': 'Need at least 2 scenarios for comparison'}
        
        comparison = {
            'scenario_count': len(prediction_results),
            'comparison_timestamp': datetime.now().isoformat(),
            'scenario_rankings': [],
            'key_differentiators': [],
            'recommended_scenario': None,
            'comparison_matrix': {}
        }
        
        # Rank scenarios by success probability
        ranked_scenarios = sorted(
            enumerate(prediction_results),
            key=lambda x: x[1].get('success_probability', 0),
            reverse=True
        )
        
        for rank, (original_index, scenario) in enumerate(ranked_scenarios, 1):
            scenario_summary = {
                'rank': rank,
                'scenario_index': original_index,
                'success_probability': scenario.get('success_probability', 0),
                'confidence_level': scenario.get('confidence_level', 'medium'),
                'legal_domain': scenario.get('legal_domain', 'general'),
                'overall_score': self._calculate_scenario_score(scenario)
            }
            comparison['scenario_rankings'].append(scenario_summary)
        
        # Identify best scenario
        comparison['recommended_scenario'] = comparison['scenario_rankings'][0]
        
        # Generate key differentiators
        comparison['key_differentiators'] = self._identify_key_differentiators(prediction_results)
        
        # Create comparison matrix
        comparison['comparison_matrix'] = self._create_comparison_matrix(prediction_results)
        
        return comparison
    
    def _calculate_scenario_score(self, scenario: Dict[str, Any]) -> float:
        """Calculate overall scenario score for comparison"""
        
        success_prob = scenario.get('success_probability', 0.5)
        confidence_mapping = {'high': 1.0, 'medium': 0.8, 'low': 0.6}
        confidence_mult = confidence_mapping.get(scenario.get('confidence_level', 'medium'), 0.8)
        
        # Factor in timeline and cost considerations
        timeline_pred = scenario.get('timeline_predictions', {})
        avg_duration = timeline_pred.get('estimated_duration_months', {}).get('average', 12)
        timeline_factor = max(0.5, 1 - (avg_duration - 6) / 24)  # Shorter is better
        
        cost_estimates = scenario.get('cost_estimates', {})
        avg_cost = cost_estimates.get('cost_estimates_inr', {}).get('average', 500000)
        cost_factor = max(0.5, 1 - (avg_cost - 200000) / 1000000)  # Lower cost is better
        
        # Weighted score calculation
        overall_score = (
            success_prob * 0.5 +
            confidence_mult * 0.2 +
            timeline_factor * 0.15 +
            cost_factor * 0.15
        )
        
        return round(overall_score, 3)
    
    def _identify_key_differentiators(self, scenarios: List[Dict[str, Any]]) -> List[str]:
        """Identify key factors that differentiate scenarios"""
        
        differentiators = []
        
        # Success probability variance
        success_probs = [s.get('success_probability', 0.5) for s in scenarios]
        if max(success_probs) - min(success_probs) > 0.2:
            differentiators.append(f"Success probability varies significantly ({min(success_probs):.1%} to {max(success_probs):.1%})")
        
        # Timeline differences
        timelines = []
        for scenario in scenarios:
            timeline_pred = scenario.get('timeline_predictions', {})
            avg_duration = timeline_pred.get('estimated_duration_months', {}).get('average', 12)
            timelines.append(avg_duration)
        
        if max(timelines) - min(timelines) > 6:
            differentiators.append(f"Timeline varies substantially ({min(timelines)} to {max(timelines)} months)")
        
        # Cost differences
        costs = []
        for scenario in scenarios:
            cost_estimates = scenario.get('cost_estimates', {})
            avg_cost = cost_estimates.get('cost_estimates_inr', {}).get('average', 500000)
            costs.append(avg_cost)
        
        if max(costs) - min(costs) > 200000:
            differentiators.append(f"Cost varies significantly (₹{min(costs):,} to ₹{max(costs):,})")
        
        # Legal domain differences
        domains = list(set(s.get('legal_domain', 'general') for s in scenarios))
        if len(domains) > 1:
            differentiators.append(f"Different legal domains involved: {', '.join(domains)}")
        
        return differentiators
    
    def _create_comparison_matrix(self, scenarios: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
        """Create comparison matrix for scenarios"""
        
        matrix = {
            'success_probability': [],
            'confidence_level': [],
            'timeline_months': [],
            'estimated_cost_inr': [],
            'overall_score': []
        }
        
        for i, scenario in enumerate(scenarios):
            matrix['success_probability'].append(f"{scenario.get('success_probability', 0.5):.1%}")
            matrix['confidence_level'].append(scenario.get('confidence_level', 'medium'))
            
            timeline_pred = scenario.get('timeline_predictions', {})
            avg_duration = timeline_pred.get('estimated_duration_months', {}).get('average', 12)
            matrix['timeline_months'].append(avg_duration)
            
            cost_estimates = scenario.get('cost_estimates', {})
            avg_cost = cost_estimates.get('cost_estimates_inr', {}).get('average', 500000)
            matrix['estimated_cost_inr'].append(f"₹{avg_cost:,}")
            
            matrix['overall_score'].append(self._calculate_scenario_score(scenario))
        
        return matrix
    
    def export_prediction_report(self, prediction_result: Dict[str, Any], format_type: str = 'executive') -> Dict[str, Any]:
        """Export prediction analysis as formatted report"""
        
        report = {
            'report_type': f'{format_type.title()} Outcome Prediction Report',
            'generated_at': datetime.now().isoformat(),
            'prediction_id': prediction_result.get('prediction_id', 'unknown'),
            'jurisdiction': prediction_result.get('jurisdiction', 'Unknown'),
            'legal_domain': prediction_result.get('legal_domain', 'general')
        }
        
        if format_type == 'executive':
            report.update(self.generate_outcome_summary(prediction_result))
        
        elif format_type == 'detailed':
            report.update({
                'full_prediction_analysis': prediction_result,
                'detailed_factors': prediction_result.get('prediction_factors', {}),
                'scenario_analysis': prediction_result.get('scenario_analysis', {}),
                'strategic_recommendations': prediction_result.get('strategic_recommendations', []),
                'alternative_outcomes': prediction_result.get('alternative_outcomes', [])
            })
        
        elif format_type == 'summary':
            report.update({
                'success_probability': f"{prediction_result.get('success_probability', 0.5):.1%}",
                'confidence_level': prediction_result.get('confidence_level', 'medium'),
                'key_recommendations': [rec.get('recommendation', '') for rec in prediction_result.get('strategic_recommendations', [])[:3]],
                'timeline_estimate': prediction_result.get('timeline_predictions', {}).get('estimated_duration_months', {}).get('average', 'Unknown'),
                'cost_estimate': prediction_result.get('cost_estimates', {}).get('cost_estimates_inr', {}).get('average', 'Unknown')
            })
        
        return report
    
    def get_prediction_confidence_factors(self, prediction_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze factors affecting prediction confidence"""
        
        confidence_factors = {
            'overall_confidence': prediction_result.get('confidence_level', 'medium'),
            'confidence_contributors': [],
            'confidence_detractors': [],
            'data_quality_assessment': {},
            'prediction_reliability': 'medium'
        }
        
        prediction_factors = prediction_result.get('prediction_factors', {})
        
        # Analyze each prediction factor
        for factor_name, factor_data in prediction_factors.items():
            if isinstance(factor_data, dict) and 'error' not in factor_data:
                confidence_factors['confidence_contributors'].append(f"{factor_name.replace('_', ' ').title()} analysis completed")
            elif 'error' in factor_data:
                confidence_factors['confidence_detractors'].append(f"{factor_name.replace('_', ' ').title()} analysis failed")
        
        # Data quality assessment
        total_factors = len(prediction_factors)
        successful_factors = len([f for f in prediction_factors.values() if isinstance(f, dict) and 'error' not in f])
        
        confidence_factors['data_quality_assessment'] = {
            'completeness_score': successful_factors / total_factors if total_factors > 0 else 0,
            'successful_analyses': successful_factors,
            'total_analyses_attempted': total_factors,
            'data_gaps': [f for f, data in prediction_factors.items() if 'error' in data]
        }
        
        # Determine overall reliability
        completeness = confidence_factors['data_quality_assessment']['completeness_score']
        if completeness >= 0.8:
            confidence_factors['prediction_reliability'] = 'high'
        elif completeness >= 0.6:
            confidence_factors['prediction_reliability'] = 'medium'
        else:
            confidence_factors['prediction_reliability'] = 'low'
        
        return confidence_factors
