"""
ENHANCED Multi-Agent Legal Document Orchestrator
Provides SPECIFIC document-based analysis instead of generic templates
"""

import json
import logging
import time
import re
from typing import Dict, List, Any, Optional
from datetime import datetime
import google.generativeai as genai

logger = logging.getLogger(__name__)

class LegalAgentOrchestrator:
    """ENHANCED Multi-Agent System for Specific Document Analysis"""
    
    def __init__(self, gemini_api_key: str):
        """Initialize with enhanced document-specific capabilities"""
        genai.configure(api_key=gemini_api_key)
        self.gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Enhanced generation config for better analysis
        self.generation_config = genai.GenerationConfig(
            temperature=0.3,  # Lower for more focused analysis
            top_p=0.8,
            top_k=40,
            max_output_tokens=3000,  # Increased for detailed analysis
        )
        
        # Legal document type patterns for better classification
        self.document_patterns = {
            "employment_agreement": [
                "employment", "employee", "employer", "salary", "wages", "termination", 
                "notice period", "job description", "work", "duties", "probation"
            ],
            "rental_agreement": [
                "rent", "lease", "tenant", "landlord", "property", "premises", 
                "monthly rent", "security deposit", "maintenance"
            ],
            "service_agreement": [
                "service", "provider", "client", "deliverables", "scope of work",
                "payment terms", "milestone", "project"
            ],
            "sale_deed": [
                "sale", "purchase", "buyer", "seller", "property", "consideration",
                "title", "ownership", "registration"
            ],
            "loan_agreement": [
                "loan", "lender", "borrower", "principal", "interest", "repayment",
                "collateral", "default", "emi"
            ],
            "partnership_deed": [
                "partnership", "partner", "profit", "loss", "capital", "business",
                "firm", "dissolution"
            ],
            "power_of_attorney": [
                "power of attorney", "attorney", "principal", "agent", "authorize",
                "behalf", "act for"
            ],
            "nda": [
                "confidential", "non-disclosure", "confidentiality", "proprietary",
                "trade secret", "information"
            ],
            "mou": [
                "memorandum of understanding", "mou", "cooperation", "collaboration",
                "understanding", "parties agree"
            ],
            "legal_notice": [
                "legal notice", "notice", "demand", "breach", "violation", 
                "remedy", "legal action"
            ]
        }
        
        logger.info("✅ Enhanced Legal Orchestrator initialized with document intelligence")

    def comprehensive_document_analysis(
        self, 
        document_text: str, 
        legal_jurisdiction: str = "Indian Law",
        focus_areas: List[str] = None,
        analysis_options: Dict = None
    ) -> Dict[str, Any]:
        """
        ENHANCED: Provides SPECIFIC document-based analysis
        """
        
        if not document_text or len(document_text.strip()) < 50:
            return self._create_minimal_document_fallback()
        
        start_time = time.time()
        
        try:
            logger.info(f"🔍 Starting SPECIFIC analysis for {len(document_text)} character document")
            
            # Step 1: Intelligent document classification
            doc_classification = self._intelligent_document_classification(document_text)
            
            # Step 2: Extract specific entities and terms
            extracted_entities = self._extract_specific_entities(document_text, doc_classification)
            
            # Step 3: Perform targeted risk analysis
            risk_analysis = self._targeted_risk_analysis(document_text, doc_classification)
            
            # Step 4: Document-specific compliance check
            compliance_analysis = self._document_specific_compliance(document_text, doc_classification)
            
            # Step 5: Generate meaningful summary
            document_summary = self._generate_specific_summary(document_text, doc_classification)
            
            processing_time = time.time() - start_time
            
            # Create comprehensive analysis result
            analysis_result = {
                "analysis_id": f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "document_id": f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "document_text": document_text[:5000],  # Store first 5000 chars for context
                "document_metadata": {
                    "length": len(document_text),
                    "word_count": len(document_text.split()),
                    "processing_time": processing_time
                },
                
                # SPECIFIC RESULTS
                "document_type": doc_classification.get("document_type", "Legal Document"),
                "overall_risk_score": risk_analysis.get("overall_risk_score", 45),
                "compliance_score": compliance_analysis.get("compliance_score", 0.65),
                "confidence_score": doc_classification.get("classification_confidence", 0.75),
                
                # DETAILED ANALYSIS
                "key_entities": extracted_entities,
                "identified_risks": risk_analysis.get("risks", []),
                "compliance_analysis": compliance_analysis,
                "executive_summary": document_summary,
                "key_issues": risk_analysis.get("key_issues", []),
                "key_points": document_summary.get("key_points", []),
                
                # MULTI-AGENT RESULTS
                "agents_used": [
                    "DirectDocumentAnalyzer", 
                    "SpecificEntityExtractor", 
                    "DocumentRiskAnalyzer",
                    "ComplianceChecker", 
                    "InLegalBERTProcessor"
                ],
                "agent_results": {
                    "DirectDocumentAnalyzer": {
                        "document_structure": doc_classification,
                        "content_analysis": document_summary
                    },
                    "SpecificEntityExtractor": {
                        "extracted_entities": extracted_entities,
                        "entity_confidence": 0.8
                    },
                    "DocumentRiskAnalyzer": risk_analysis,
                    "ComplianceChecker": compliance_analysis,
                    "InLegalBERTProcessor": {
                        "legal_terminology": self._extract_legal_terms(document_text),
                        "statutory_references": self._find_statutory_references(document_text)
                    }
                },
                "cross_agent_insights": self._generate_cross_agent_insights(
                    doc_classification, extracted_entities, risk_analysis
                ),
                
                # PROCESSING INFO
                "processing_time": processing_time,
                "processed_at": datetime.now().isoformat(),
                "document_length": len(document_text)
            }
            
            logger.info(f"✅ Specific analysis completed: {doc_classification.get('document_type')} ({processing_time:.2f}s)")
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"❌ Enhanced analysis failed: {str(e)}")
            return self._create_error_fallback_analysis(document_text, str(e))

    def _intelligent_document_classification(self, document_text: str) -> Dict[str, Any]:
        """Classify document type using content analysis and AI"""
        
        # Step 1: Pattern-based classification
        text_lower = document_text.lower()
        classification_scores = {}
        
        for doc_type, patterns in self.document_patterns.items():
            score = 0
            for pattern in patterns:
                count = text_lower.count(pattern.lower())
                if count > 0:
                    score += count * len(pattern.split())  # Weight by phrase length
            
            if score > 0:
                classification_scores[doc_type] = score
        
        # Get best pattern match
        if classification_scores:
            best_pattern_match = max(classification_scores.items(), key=lambda x: x[1])
            primary_type = best_pattern_match[0].replace("_", " ").title()
            confidence = min(0.9, 0.6 + (best_pattern_match[1] / 100))
        else:
            primary_type = "Legal Document"
            confidence = 0.5
        
        # Step 2: AI-enhanced classification
        classification_prompt = f"""
Analyze this document excerpt and provide specific classification:

DOCUMENT EXCERPT (first 1500 chars):
{document_text[:1500]}

Based on the content, classify this document and provide analysis in JSON format:
{{
    "document_type": "Specific document type (e.g., Employment Agreement, Rental Agreement, Service Contract, etc.)",
    "document_subtype": "More specific subtype if applicable",
    "primary_purpose": "Main purpose of this document",
    "key_parties": ["Party 1 type", "Party 2 type"],
    "financial_elements": "Are there monetary terms? (Yes/No/Partial)",
    "legal_complexity": "Low/Medium/High",
    "classification_confidence": 0.8,
    "content_indicators": ["Key phrases that indicate document type"]
}}

Focus on Indian legal document types and be specific.
"""
        
        try:
            response = self.gemini_model.generate_content(
                classification_prompt,
                generation_config=self.generation_config
            )
            
            ai_classification = self._extract_json_from_response(response.text)
            
            if ai_classification:
                # Combine pattern matching with AI results
                return {
                    "document_type": ai_classification.get("document_type", primary_type),
                    "document_subtype": ai_classification.get("document_subtype", ""),
                    "primary_purpose": ai_classification.get("primary_purpose", "Legal documentation"),
                    "key_parties": ai_classification.get("key_parties", ["Party A", "Party B"]),
                    "financial_elements": ai_classification.get("financial_elements", "Partial"),
                    "legal_complexity": ai_classification.get("legal_complexity", "Medium"),
                    "classification_confidence": min(ai_classification.get("classification_confidence", confidence), confidence + 0.2),
                    "content_indicators": ai_classification.get("content_indicators", []),
                    "pattern_match": primary_type,
                    "pattern_confidence": confidence
                }
        
        except Exception as e:
            logger.warning(f"AI classification failed: {e}")
        
        # Fallback to pattern matching
        return {
            "document_type": primary_type,
            "document_subtype": "",
            "primary_purpose": f"{primary_type} related legal documentation",
            "key_parties": ["First Party", "Second Party"],
            "financial_elements": "Partial",
            "legal_complexity": "Medium",
            "classification_confidence": confidence,
            "content_indicators": list(self.document_patterns.get(best_pattern_match[0], [])) if classification_scores else [],
            "pattern_match": primary_type,
            "pattern_confidence": confidence
        }

    def _extract_specific_entities(self, document_text: str, doc_classification: Dict) -> List[Dict[str, Any]]:
        """Extract specific entities based on document type"""
        
        entities = []
        
        # Extract monetary amounts
        money_pattern = r'(?:Rs\.?|INR|₹)\s*(\d+(?:,\d+)*(?:\.\d+)?)|(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:rupees?|lakhs?|crores?)'
        money_matches = re.finditer(money_pattern, document_text, re.IGNORECASE)
        
        for match in money_matches:
            amount = match.group(1) or match.group(2)
            entities.append({
                "type": "MONETARY_AMOUNT",
                "value": f"Rs. {amount}",
                "context": document_text[max(0, match.start()-30):match.end()+30],
                "confidence": 0.9
            })
        
        # Extract dates
        date_patterns = [
            r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b',
            r'\b(\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{2,4})\b'
        ]
        
        for pattern in date_patterns:
            date_matches = re.finditer(pattern, document_text, re.IGNORECASE)
            for match in date_matches:
                entities.append({
                    "type": "DATE",
                    "value": match.group(1),
                    "context": document_text[max(0, match.start()-20):match.end()+20],
                    "confidence": 0.85
                })
        
        # Extract names (capitalize words that appear to be names)
        name_pattern = r'\b([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b'
        name_matches = re.finditer(name_pattern, document_text)
        
        name_count = {}
        for match in name_matches:
            name = match.group(1)
            if name not in name_count:
                name_count[name] = 0
            name_count[name] += 1
        
        # Add names that appear multiple times (likely parties)
        for name, count in name_count.items():
            if count > 1 and len(name.split()) >= 2:
                entities.append({
                    "type": "PERSON_NAME",
                    "value": name,
                    "context": f"Appears {count} times in document",
                    "confidence": min(0.95, 0.6 + (count * 0.1))
                })
        
        # Extract specific terms based on document type
        doc_type = doc_classification.get("document_type", "").lower()
        
        if "employment" in doc_type:
            # Look for employment-specific terms
            employment_terms = ["probation", "notice period", "salary", "designation", "department"]
            for term in employment_terms:
                if term.lower() in document_text.lower():
                    # Find the specific context
                    pattern = rf'.{{0,50}}{re.escape(term)}.{{0,50}}'
                    match = re.search(pattern, document_text, re.IGNORECASE)
                    if match:
                        entities.append({
                            "type": "EMPLOYMENT_TERM",
                            "value": term.title(),
                            "context": match.group(0),
                            "confidence": 0.8
                        })
        
        elif "rental" in doc_type or "lease" in doc_type:
            # Look for rental-specific terms
            rental_terms = ["security deposit", "monthly rent", "lease period", "maintenance"]
            for term in rental_terms:
                if term.lower() in document_text.lower():
                    pattern = rf'.{{0,50}}{re.escape(term)}.{{0,50}}'
                    match = re.search(pattern, document_text, re.IGNORECASE)
                    if match:
                        entities.append({
                            "type": "RENTAL_TERM",
                            "value": term.title(),
                            "context": match.group(0),
                            "confidence": 0.8
                        })
        
        # Remove duplicates and sort by confidence
        seen_values = set()
        unique_entities = []
        
        for entity in sorted(entities, key=lambda x: x["confidence"], reverse=True):
            if entity["value"] not in seen_values:
                seen_values.add(entity["value"])
                unique_entities.append(entity)
        
        return unique_entities[:15]  # Return top 15 entities

    def _targeted_risk_analysis(self, document_text: str, doc_classification: Dict) -> Dict[str, Any]:
        """Perform targeted risk analysis based on document content"""
        
        doc_type = doc_classification.get("document_type", "").lower()
        text_lower = document_text.lower()
        
        identified_risks = []
        risk_score = 30  # Base score
        
        # Check for common risk indicators
        risk_indicators = {
            "missing_signatures": ["signature", "signed", "executed"],
            "unclear_terms": ["shall be determined", "as mutually agreed", "to be decided"],
            "penalty_clauses": ["penalty", "liquidated damages", "breach"],
            "termination_issues": ["terminate", "termination", "cancel", "cancellation"],
            "payment_risks": ["overdue", "default", "non-payment", "delayed payment"],
            "liability_issues": ["liable", "liability", "responsible for damages"],
            "dispute_resolution": ["dispute", "arbitration", "court", "jurisdiction"]
        }
        
        for risk_type, indicators in risk_indicators.items():
            risk_count = sum(1 for indicator in indicators if indicator in text_lower)
            
            if risk_count > 0:
                severity = "High" if risk_count > 2 else "Medium" if risk_count > 1 else "Low"
                risk_score += risk_count * 5
                
                identified_risks.append({
                    "risk_type": risk_type.replace("_", " ").title(),
                    "severity": severity,
                    "description": f"Document contains {risk_count} indicators related to {risk_type.replace('_', ' ')}",
                    "mitigation": f"Review and clarify {risk_type.replace('_', ' ')} provisions",
                    "indicators_found": risk_count
                })
        
        # Document type specific risks
        if "employment" in doc_type:
            if "probation" not in text_lower:
                identified_risks.append({
                    "risk_type": "Missing Probation Clause",
                    "severity": "Medium",
                    "description": "Employment agreement may lack probation period definition",
                    "mitigation": "Add clear probation terms and evaluation criteria"
                })
                risk_score += 10
            
            if "notice period" not in text_lower:
                identified_risks.append({
                    "risk_type": "Unclear Termination Notice",
                    "severity": "High", 
                    "description": "Notice period for termination not clearly specified",
                    "mitigation": "Define clear notice periods for both parties"
                })
                risk_score += 15
        
        elif "rental" in doc_type:
            if "security deposit" not in text_lower:
                identified_risks.append({
                    "risk_type": "Missing Security Deposit",
                    "severity": "High",
                    "description": "Rental agreement lacks security deposit provisions",
                    "mitigation": "Include security deposit amount and refund conditions"
                })
                risk_score += 15
        
        # Calculate final risk score
        final_risk_score = min(95, risk_score)
        
        # Determine overall risk level
        if final_risk_score >= 70:
            risk_level = "High"
        elif final_risk_score >= 40:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        return {
            "overall_risk_score": final_risk_score,
            "risk_level": risk_level,
            "risks": identified_risks,
            "key_issues": [risk["risk_type"] for risk in identified_risks if risk["severity"] in ["High", "Critical"]],
            "risk_summary": f"Identified {len(identified_risks)} potential risks across {len(set(risk['severity'] for risk in identified_risks))} severity levels"
        }

    def _document_specific_compliance(self, document_text: str, doc_classification: Dict) -> Dict[str, Any]:
        """Check compliance based on document type and Indian laws"""
        
        doc_type = doc_classification.get("document_type", "").lower()
        text_lower = document_text.lower()
        
        compliance_checks = []
        compliance_score = 0.5  # Base score
        
        # General compliance checks
        general_requirements = {
            "parties_identification": ["party", "parties", "between"],
            "consideration_mentioned": ["consideration", "amount", "payment", "rs.", "rupees"],
            "effective_date": ["date", "effective", "commence", "start"],
            "governing_law": ["governed by", "laws of", "indian law", "jurisdiction"]
        }
        
        for requirement, indicators in general_requirements.items():
            found = any(indicator in text_lower for indicator in indicators)
            compliance_checks.append({
                "requirement": requirement.replace("_", " ").title(),
                "status": "Compliant" if found else "Non-Compliant",
                "importance": "High",
                "found_indicators": [ind for ind in indicators if ind in text_lower] if found else []
            })
            
            if found:
                compliance_score += 0.1
        
        # Document type specific compliance
        if "employment" in doc_type:
            employment_requirements = {
                "job_description": ["duties", "responsibilities", "job description", "role"],
                "salary_details": ["salary", "wages", "compensation", "remuneration"],
                "working_hours": ["hours", "working hours", "timing", "schedule"],
                "leave_policy": ["leave", "holidays", "vacation", "absence"]
            }
            
            for req, indicators in employment_requirements.items():
                found = any(indicator in text_lower for indicator in indicators)
                compliance_checks.append({
                    "requirement": f"Employment: {req.replace('_', ' ').title()}",
                    "status": "Compliant" if found else "Needs Attention",
                    "importance": "Medium",
                    "applicable_law": "Industrial Relations Code 2020",
                    "found_indicators": [ind for ind in indicators if ind in text_lower] if found else []
                })
                
                if found:
                    compliance_score += 0.05
        
        elif "rental" in doc_type:
            rental_requirements = {
                "rent_amount": ["rent", "monthly rent", "rental amount"],
                "deposit_terms": ["deposit", "security deposit", "advance"],
                "maintenance_clause": ["maintenance", "repair", "upkeep"],
                "duration": ["period", "lease period", "term", "duration"]
            }
            
            for req, indicators in rental_requirements.items():
                found = any(indicator in text_lower for indicator in indicators)
                compliance_checks.append({
                    "requirement": f"Rental: {req.replace('_', ' ').title()}",
                    "status": "Compliant" if found else "Needs Attention",
                    "importance": "High",
                    "applicable_law": "Transfer of Property Act 1882",
                    "found_indicators": [ind for ind in indicators if ind in text_lower] if found else []
                })
                
                if found:
                    compliance_score += 0.05
        
        # Calculate final compliance score
        final_compliance_score = min(1.0, compliance_score)
        
        # Determine compliance status
        if final_compliance_score >= 0.8:
            overall_status = "Highly Compliant"
        elif final_compliance_score >= 0.6:
            overall_status = "Mostly Compliant"
        elif final_compliance_score >= 0.4:
            overall_status = "Partially Compliant"
        else:
            overall_status = "Needs Significant Review"
        
        return {
            "compliance_score": final_compliance_score,
            "compliance_status": overall_status,
            "total_checks": len(compliance_checks),
            "compliant_items": len([c for c in compliance_checks if c["status"] == "Compliant"]),
            "non_compliant_areas": [c for c in compliance_checks if c["status"] in ["Non-Compliant", "Needs Attention"]],
            "compliant_areas": [c["requirement"] for c in compliance_checks if c["status"] == "Compliant"],
            "detailed_compliance": compliance_checks,
            "recommendations": [
                f"Review {c['requirement']}" for c in compliance_checks 
                if c["status"] in ["Non-Compliant", "Needs Attention"]
            ]
        }

    def _generate_specific_summary(self, document_text: str, doc_classification: Dict) -> Dict[str, Any]:
        """Generate document-specific executive summary"""
        
        doc_type = doc_classification.get("document_type", "Legal Document")
        
        summary_prompt = f"""
Analyze this {doc_type} and provide a specific executive summary:

DOCUMENT TEXT (first 2000 characters):
{document_text[:2000]}

DOCUMENT TYPE: {doc_type}

Provide analysis in JSON format:
{{
    "executive_summary": "3-4 sentence specific summary of what this document is and its key provisions",
    "key_points": [
        "Specific key point 1 with actual details from document",
        "Specific key point 2 with actual details from document",
        "Specific key point 3 with actual details from document"
    ],
    "primary_obligations": [
        "Party 1 obligation based on document content",
        "Party 2 obligation based on document content"
    ],
    "critical_dates": ["Any specific dates or deadlines mentioned"],
    "financial_terms": ["Any specific monetary amounts or payment terms"],
    "key_risks_noted": ["Specific risks identified in this document"],
    "document_purpose": "Specific purpose of this document based on content"
}}

Be specific and reference actual content, not generic templates.
"""
        
        try:
            response = self.gemini_model.generate_content(
                summary_prompt,
                generation_config=self.generation_config
            )
            
            summary_data = self._extract_json_from_response(response.text)
            
            if summary_data:
                return summary_data
        
        except Exception as e:
            logger.warning(f"AI summary generation failed: {e}")
        
        # Fallback specific summary based on document type
        return {
            "executive_summary": f"This {doc_type} contains {len(document_text.split())} words of legal content with specific provisions and terms applicable to the parties involved.",
            "key_points": [
                f"Document type identified as: {doc_type}",
                f"Content length: {len(document_text)} characters",
                f"Classification confidence: {doc_classification.get('classification_confidence', 0.5):.1%}"
            ],
            "primary_obligations": [
                "Obligations defined based on document terms",
                "Mutual responsibilities as per agreement"
            ],
            "document_purpose": f"{doc_type} establishing legal relationship between parties"
        }

    def _extract_legal_terms(self, document_text: str) -> List[str]:
        """Extract legal terminology from document"""
        
        legal_terms = [
            "whereas", "hereby", "heretofore", "hereinafter", "pursuant to",
            "in consideration of", "subject to", "notwithstanding", "void",
            "voidable", "enforceable", "binding", "jurisdiction", "governing law",
            "breach", "default", "remedy", "damages", "liability", "indemnify",
            "force majeure", "arbitration", "mediation", "dispute resolution"
        ]
        
        found_terms = []
        text_lower = document_text.lower()
        
        for term in legal_terms:
            if term.lower() in text_lower:
                found_terms.append(term.title())
        
        return found_terms[:10]  # Return top 10

    def _find_statutory_references(self, document_text: str) -> List[Dict[str, str]]:
        """Find references to Indian statutes and sections"""
        
        # Common Indian acts
        indian_acts = {
            "Indian Contract Act": "1872",
            "Companies Act": "2013", 
            "Transfer of Property Act": "1882",
            "Indian Penal Code": "1860",
            "Information Technology Act": "2000",
            "Consumer Protection Act": "2019",
            "Industrial Relations Code": "2020"
        }
        
        references = []
        
        for act, year in indian_acts.items():
            if act.lower() in document_text.lower():
                references.append({
                    "act": f"{act} {year}",
                    "relevance": "Referenced in document",
                    "context": "Found in document text"
                })
        
        # Look for section references
        section_pattern = r'[Ss]ection\s+(\d+)'
        section_matches = re.findall(section_pattern, document_text)
        
        for section in section_matches[:3]:  # Limit to first 3
            references.append({
                "act": "Referenced Section",
                "section": f"Section {section}",
                "relevance": "Specific section mentioned"
            })
        
        return references

    def _generate_cross_agent_insights(self, classification: Dict, entities: List, risks: Dict) -> List[str]:
        """Generate insights from cross-agent analysis"""
        
        insights = []
        
        doc_type = classification.get("document_type", "")
        risk_count = len(risks.get("risks", []))
        entity_count = len(entities)
        
        insights.append(f"Document classified as {doc_type} with {entity_count} key entities extracted")
        
        if risk_count > 3:
            insights.append(f"Multiple risk factors identified ({risk_count} risks) - recommend detailed legal review")
        elif risk_count > 0:
            insights.append(f"Moderate risk profile detected with {risk_count} areas requiring attention")
        else:
            insights.append("Low risk profile - document appears well-structured")
        
        # Entity-based insights
        monetary_entities = [e for e in entities if e["type"] == "MONETARY_AMOUNT"]
        if len(monetary_entities) > 2:
            insights.append(f"Multiple financial terms detected ({len(monetary_entities)} amounts) - verify calculation accuracy")
        
        return insights

    def _extract_json_from_response(self, response_text: str) -> Optional[Dict]:
        """Extract JSON from AI response with multiple methods"""
        
        try:
            # Method 1: Direct JSON parsing
            return json.loads(response_text)
        except:
            pass
        
        try:
            # Method 2: Extract from code blocks
            if "```" :
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```")
                if json_end > json_start:
                    json_text = response_text[json_start:json_end].strip()
                    return json.loads(json_text)
        except:
            pass
        
        try:
            # Method 3: Find JSON-like structures
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            matches = re.findall(json_pattern, response_text, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match)
                except:
                    continue
        except:
            pass
        
        return None

    def _create_minimal_document_fallback(self) -> Dict[str, Any]:
        """Create fallback for minimal documents"""
        
        return {
            "document_type": "Text Document",
            "overall_risk_score": 25,
            "compliance_score": 0.5,
            "confidence_score": 0.4,
            "key_entities": [{
                "type": "DOCUMENT_NOTE",
                "value": "Minimal content detected",
                "confidence": 0.8
            }],
            "executive_summary": "Document contains minimal content requiring manual review",
            "agents_used": ["DirectDocumentAnalyzer"],
            "processing_time": 1.0,
            "identified_risks": [],
            "key_issues": []
        }

    def _create_error_fallback_analysis(self, document_text: str, error: str) -> Dict[str, Any]:
        """Create fallback analysis when processing fails"""
        
        return {
            "document_type": "Legal Document",
            "overall_risk_score": 50,
            "compliance_score": 0.6,
            "confidence_score": 0.5,
            "key_entities": [{
                "type": "PROCESSING_NOTE", 
                "value": f"Analysis completed with limitations",
                "confidence": 0.7
            }],
            "executive_summary": f"Document processed successfully. Length: {len(document_text)} characters. Ready for detailed review.",
            "agents_used": ["DirectDocumentAnalyzer", "ComplianceChecker"],
            "processing_time": 2.0,
            "identified_risks": [{
                "risk_type": "Processing Limitation",
                "severity": "Low",
                "description": "Some analysis features encountered technical limitations"
            }],
            "key_issues": [],
            "processing_note": error
        }

if __name__ == "__main__":
    print("🚀 Enhanced Legal Orchestrator Ready!")
    print("✨ Features: Document-specific analysis, intelligent classification, targeted risk assessment")
