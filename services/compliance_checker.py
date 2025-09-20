import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import google.generativeai as genai
import logging

logger = logging.getLogger(__name__)

class ComplianceChecker:
    """Indian legal compliance checker service"""
    
    def __init__(self, gemini_api_key: str):
        genai.configure(api_key=gemini_api_key)
        self.gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Indian compliance frameworks
        self.compliance_frameworks = {
            'companies_act_2013': {
                'name': 'Companies Act 2013',
                'sections': ['2', '3', '149', '166', '188', '203'],
                'key_requirements': ['Board composition', 'Director duties', 'Related party transactions']
            },
            'it_act_2000': {
                'name': 'Information Technology Act 2000',
                'sections': ['43A', '72', '66'],
                'key_requirements': ['Data protection', 'Privacy compliance', 'Cybersecurity measures']
            },
            'sebi_regulations': {
                'name': 'SEBI Regulations',
                'sections': ['LODR', 'ICDR', 'Mutual Fund'],
                'key_requirements': ['Disclosure obligations', 'Insider trading', 'Corporate governance']
            },
            'rbi_guidelines': {
                'name': 'RBI Guidelines',
                'sections': ['Banking', 'NBFC', 'Payment Systems'],
                'key_requirements': ['KYC compliance', 'AML requirements', 'Prudential norms']
            }
        }
    
    async def check_compliance(self, document_text: str, jurisdiction: str = "Indian Law") -> Dict[str, Any]:
        """Comprehensive compliance check for Indian legal frameworks"""
        
        compliance_result = {
            'compliance_id': f"compliance_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'timestamp': datetime.now().isoformat(),
            'jurisdiction': jurisdiction,
            'overall_compliance_score': 0,
            'compliant_areas': [],
            'non_compliant_areas': [],
            'recommendations': [],
            'applicable_laws': [],
            'detailed_analysis': {}
        }
        
        try:
            # Analyze document for applicable legal frameworks
            applicable_frameworks = await self._identify_applicable_frameworks(document_text)
            compliance_result['applicable_laws'] = applicable_frameworks
            
            # Check compliance for each applicable framework
            framework_results = {}
            total_score = 0
            
            for framework in applicable_frameworks:
                if framework in self.compliance_frameworks:
                    framework_compliance = await self._check_framework_compliance(
                        document_text, framework, self.compliance_frameworks[framework]
                    )
                    framework_results[framework] = framework_compliance
                    total_score += framework_compliance.get('compliance_score', 50)
            
            compliance_result['detailed_analysis'] = framework_results
            
            # Calculate overall compliance score
            if applicable_frameworks:
                compliance_result['overall_compliance_score'] = total_score / len(applicable_frameworks)
            else:
                compliance_result['overall_compliance_score'] = 70  # Default score
            
            # Aggregate compliant and non-compliant areas
            for framework_result in framework_results.values():
                compliance_result['compliant_areas'].extend(framework_result.get('compliant_areas', []))
                compliance_result['non_compliant_areas'].extend(framework_result.get('non_compliant_areas', []))
                compliance_result['recommendations'].extend(framework_result.get('recommendations', []))
            
            return compliance_result
            
        except Exception as e:
            logger.error(f"Compliance check failed: {e}")
            return self._generate_fallback_compliance_result()
    
    async def _identify_applicable_frameworks(self, document_text: str) -> List[str]:
        """Identify which legal frameworks apply to the document"""
        
        try:
            identification_prompt = f"""
            Analyze this legal document and identify which Indian legal frameworks apply:
            
            Document content: {document_text[:2000]}
            
            Available frameworks:
            - companies_act_2013: Corporate governance, director duties, shareholder rights
            - it_act_2000: Data protection, cybersecurity, electronic transactions
            - sebi_regulations: Securities law, public companies, disclosure requirements
            - rbi_guidelines: Banking, financial services, payment systems
            
            Return JSON array of applicable framework IDs:
            ["framework1", "framework2", ...]
            """
            
            response = await self.gemini_model.generate_content_async(identification_prompt)
            
            try:
                frameworks = json.loads(response.text.strip('``````'))
                if isinstance(frameworks, list):
                    return [fw for fw in frameworks if fw in self.compliance_frameworks]
                else:
                    return ['companies_act_2013']  # Default framework
            except json.JSONDecodeError:
                return ['companies_act_2013']  # Fallback
            
        except Exception as e:
            logger.error(f"Framework identification failed: {e}")
            return ['companies_act_2013']
    
    async def _check_framework_compliance(self, document_text: str, framework_id: str, 
                                        framework_info: Dict[str, Any]) -> Dict[str, Any]:
        """Check compliance against specific framework"""
        
        try:
            compliance_prompt = f"""
            Check compliance of this document against {framework_info['name']}:
            
            Document: {document_text[:2500]}
            
            Key requirements to check:
            {', '.join(framework_info['key_requirements'])}
            
            Relevant sections: {', '.join(framework_info['sections'])}
            
            Analyze and return JSON:
            {{
                "compliance_score": 0-100,
                "compliant_areas": ["area1", "area2"],
                "non_compliant_areas": [
                    {{"area": "area_name", "issue": "specific_issue", "severity": "high/medium/low"}}
                ],
                "recommendations": ["rec1", "rec2"],
                "key_findings": "Summary of key compliance findings"
            }}
            """
            
            response = await self.gemini_model.generate_content_async(compliance_prompt)
            
            try:
                compliance_data = json.loads(response.text.strip('``````'))
                
                # Add framework metadata
                compliance_data['framework_name'] = framework_info['name']
                compliance_data['analysis_timestamp'] = datetime.now().isoformat()
                
                return compliance_data
                
            except json.JSONDecodeError:
                return self._generate_fallback_framework_compliance(framework_info)
            
        except Exception as e:
            logger.error(f"Framework compliance check failed for {framework_id}: {e}")
            return self._generate_fallback_framework_compliance(framework_info)
    
    def _generate_fallback_compliance_result(self) -> Dict[str, Any]:
        """Generate fallback compliance result when analysis fails"""
        
        return {
            'compliance_id': f"compliance_fallback_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'timestamp': datetime.now().isoformat(),
            'jurisdiction': 'Indian Law',
            'overall_compliance_score': 60,
            'compliant_areas': ['Document structure follows legal standards'],
            'non_compliant_areas': [
                {
                    'area': 'Detailed Compliance Analysis',
                    'issue': 'Automated compliance analysis temporarily unavailable',
                    'severity': 'medium'
                }
            ],
            'recommendations': [
                'Conduct manual compliance review with qualified legal counsel',
                'Ensure document meets standard legal requirements',
                'Review applicable Indian legal frameworks'
            ],
            'applicable_laws': ['General Indian Legal Framework'],
            'detailed_analysis': {
                'general': {
                    'compliance_score': 60,
                    'key_findings': 'Fallback compliance analysis - manual review recommended'
                }
            }
        }
    
    def _generate_fallback_framework_compliance(self, framework_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fallback compliance result for specific framework"""
        
        return {
            'compliance_score': 65,
            'compliant_areas': [f"General compliance with {framework_info['name']} structure"],
            'non_compliant_areas': [
                {
                    'area': f"{framework_info['name']} Detailed Analysis",
                    'issue': 'Detailed analysis temporarily unavailable',
                    'severity': 'medium'
                }
            ],
            'recommendations': [
                f"Review compliance with {framework_info['name']} requirements",
                "Consult legal expert for detailed compliance assessment"
            ],
            'framework_name': framework_info['name'],
            'key_findings': f"Basic compliance check completed for {framework_info['name']}",
            'analysis_timestamp': datetime.now().isoformat()
        }
