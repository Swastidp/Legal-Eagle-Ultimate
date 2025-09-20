import asyncio
import json
import torch
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import google.generativeai as genai

class IndianStatuteIdentifier:
    def __init__(self, legal_model=None, legal_tokenizer=None, gemini_model=None):
        self.legal_model = legal_model
        self.legal_tokenizer = legal_tokenizer
        self.gemini_model = gemini_model
        
        # Comprehensive Indian legal statutes database
        self.indian_statutes_db = {
            'corporate_law': {
                'Companies Act 2013': {
                    'sections': {
                        'Section 2': 'Definitions',
                        'Section 8': 'Formation of companies with charitable objects',
                        'Section 12': 'Registered office of company',
                        'Section 17': 'Contents of memorandum',
                        'Section 149': 'Company to have Board of Directors',
                        'Section 166': 'Duties of directors',
                        'Section 177': 'Audit Committee',
                        'Section 188': 'Related party transactions',
                        'Section 230': 'Power to compromise or arrange'
                    },
                    'keywords': ['company', 'director', 'board', 'shareholder', 'dividend', 'audit', 'compliance']
                },
                'SEBI Act 1992': {
                    'sections': {
                        'Section 11': 'Powers and functions of Board',
                        'Section 11A': 'Power to call for information',
                        'Section 12': 'Powers of Board to make regulations',
                        'Section 15': 'Investigation'
                    },
                    'keywords': ['securities', 'stock exchange', 'investor', 'market', 'listing', 'disclosure']
                }
            },
            'contract_law': {
                'Indian Contract Act 1872': {
                    'sections': {
                        'Section 10': 'What agreements are contracts',
                        'Section 11': 'Who are competent to contract',
                        'Section 23': 'What consideration and objects are lawful',
                        'Section 56': 'Agreement to do impossible act',
                        'Section 73': 'Compensation for loss or damage caused by breach of contract',
                        'Section 124': 'Contract of guarantee, surety, principal debtor and creditor'
                    },
                    'keywords': ['contract', 'agreement', 'consideration', 'breach', 'damages', 'offer', 'acceptance']
                },
                'Specific Relief Act 1963': {
                    'sections': {
                        'Section 9': 'Specific performance of part of contract',
                        'Section 10': 'Cases in which specific performance of contract enforceable',
                        'Section 38': 'Perpetual injunction when granted',
                        'Section 39': 'Mandatory injunctions'
                    },
                    'keywords': ['specific performance', 'injunction', 'relief', 'remedy', 'enforcement']
                }
            },
            'employment_law': {
                'Industrial Relations Code 2020': {
                    'sections': {
                        'Section 2': 'Definitions',
                        'Section 13': 'Conditions of service',
                        'Section 25': 'Prohibition of strikes and lock-outs',
                        'Section 62': 'Industrial tribunals'
                    },
                    'keywords': ['employment', 'industrial', 'worker', 'strike', 'lockout', 'tribunal', 'wages']
                },
                'Factories Act 1948': {
                    'sections': {
                        'Section 7A': 'General duties of occupiers',
                        'Section 51': 'Hours of work for adults',
                        'Section 52': 'Weekly hours',
                        'Section 64': 'Annual leave with wages'
                    },
                    'keywords': ['factory', 'worker', 'safety', 'hours', 'leave', 'occupier', 'working conditions']
                }
            },
            'it_cyber_law': {
                'Information Technology Act 2000': {
                    'sections': {
                        'Section 43': 'Penalty and compensation for damage to computer, computer system, etc.',
                        'Section 43A': 'Compensation for failure to protect data',
                        'Section 65': 'Tampering with computer source documents',
                        'Section 66': 'Computer related offences',
                        'Section 72': 'Breach of confidentiality and privacy',
                        'Section 79': 'Exemption from liability of intermediary'
                    },
                    'keywords': ['computer', 'data', 'cyber', 'electronic', 'digital', 'information', 'technology']
                },
                'Personal Data Protection Bill 2019': {
                    'sections': {
                        'Section 3': 'Territorial application',
                        'Section 4': 'Applicability to State',
                        'Section 12': 'Grounds for processing of personal data',
                        'Section 24': 'Data protection impact assessment'
                    },
                    'keywords': ['personal data', 'privacy', 'consent', 'processing', 'data fiduciary', 'data principal']
                }
            },
            'banking_finance': {
                'Banking Regulation Act 1949': {
                    'sections': {
                        'Section 5': 'Definition of banking',
                        'Section 6': 'Forms of business in which banking companies may engage',
                        'Section 35A': 'Power of Reserve Bank to give directions'
                    },
                    'keywords': ['bank', 'banking', 'deposit', 'loan', 'rbi', 'financial', 'credit']
                },
                'SARFAESI Act 2002': {
                    'sections': {
                        'Section 13': 'Enforcement of security interest',
                        'Section 14': 'Power to take possession of secured asset',
                        'Section 17': 'Appeal to Debts Recovery Tribunal'
                    },
                    'keywords': ['secured asset', 'npa', 'reconstruction', 'enforcement', 'security interest']
                }
            },
            'property_law': {
                'Transfer of Property Act 1882': {
                    'sections': {
                        'Section 5': 'Transfer of property defined',
                        'Section 54': 'Sale defined',
                        'Section 58': 'Mortgage defined',
                        'Section 105': 'Lease defined'
                    },
                    'keywords': ['property', 'transfer', 'sale', 'mortgage', 'lease', 'immovable']
                },
                'Registration Act 1908': {
                    'sections': {
                        'Section 17': 'Documents of which registration is compulsory',
                        'Section 23': 'Time for presentation',
                        'Section 49': 'Effect of non-registration'
                    },
                    'keywords': ['registration', 'document', 'registrar', 'deed', 'stamp duty']
                }
            },
            'tax_law': {
                'Income Tax Act 1961': {
                    'sections': {
                        'Section 2': 'Definitions',
                        'Section 4': 'Charge of income-tax',
                        'Section 80C': 'Deduction in respect of life insurance premia, etc.',
                        'Section 234A': 'Interest for defaults in furnishing return of income'
                    },
                    'keywords': ['income tax', 'assessment', 'return', 'deduction', 'exemption', 'penalty']
                },
                'Goods and Services Tax Act 2017': {
                    'sections': {
                        'Section 9': 'Levy and collection',
                        'Section 16': 'Eligibility and conditions for taking input tax credit',
                        'Section 37': 'Furnishing details of outward supplies'
                    },
                    'keywords': ['gst', 'goods and services tax', 'input tax credit', 'supply', 'invoice']
                }
            }
        }
    
    async def identify_relevant_statutes(self, document_text: str, document_structure: Dict[str, Any] = None, 
                                       legal_domain: str = 'general') -> Dict[str, Any]:
        """Identify relevant Indian statutes using InLegalBERT + Gemini + statute database"""
        
        try:
            # Phase 1: Database matching based on keywords and domain
            db_matches = self._database_keyword_matching(document_text, legal_domain)
            
            # Phase 2: InLegalBERT-based statute identification (if available)
            inlegal_matches = []
            if self.legal_model is not None and self.legal_tokenizer is not None:
                inlegal_matches = await self._inlegal_statute_identification(document_text, db_matches)
            
            # Phase 3: Gemini-powered contextual statute identification
            gemini_matches = await self._gemini_statute_identification(
                document_text, document_structure, legal_domain, db_matches
            )
            
            # Phase 4: Merge and rank all matches
            consolidated_statutes = self._consolidate_statute_matches(
                db_matches, inlegal_matches, gemini_matches
            )
            
            # Phase 5: Generate legal framework summary
            framework_summary = await self._generate_framework_summary(consolidated_statutes, legal_domain)
            
            return {
                'statutes': consolidated_statutes,
                'framework_summary': framework_summary,
                'identification_confidence': self._calculate_identification_confidence(consolidated_statutes),
                'legal_domain': legal_domain,
                'identification_metadata': {
                    'method': 'Hybrid: Database + InLegalBERT + Gemini',
                    'total_statutes_found': len(consolidated_statutes),
                    'database_matches': len(db_matches),
                    'inlegal_matches': len(inlegal_matches),
                    'gemini_enhanced': True,
                    'timestamp': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            return await self._fallback_statute_identification(document_text, legal_domain, str(e))
    
    def _database_keyword_matching(self, document_text: str, legal_domain: str) -> List[Dict[str, Any]]:
        """Match statutes based on keyword analysis and legal domain"""
        
        document_lower = document_text.lower()
        matches = []
        
        # Focus on relevant legal areas based on domain
        relevant_areas = []
        if legal_domain == 'corporate':
            relevant_areas = ['corporate_law']
        elif legal_domain == 'contract':
            relevant_areas = ['contract_law']
        elif legal_domain == 'employment':
            relevant_areas = ['employment_law']
        elif legal_domain == 'it_cyber':
            relevant_areas = ['it_cyber_law']
        elif legal_domain == 'banking':
            relevant_areas = ['banking_finance']
        elif legal_domain == 'property':
            relevant_areas = ['property_law']
        elif legal_domain == 'tax':
            relevant_areas = ['tax_law']
        else:
            # Check all areas for general domain
            relevant_areas = list(self.indian_statutes_db.keys())
        
        for area in relevant_areas:
            if area in self.indian_statutes_db:
                for statute_name, statute_data in self.indian_statutes_db[area].items():
                    keywords = statute_data.get('keywords', [])
                    
                    # Calculate relevance score based on keyword frequency
                    keyword_score = 0
                    matched_keywords = []
                    
                    for keyword in keywords:
                        count = document_lower.count(keyword.lower())
                        if count > 0:
                            keyword_score += count
                            matched_keywords.append(keyword)
                    
                    # Also check for explicit statute name mentions
                    statute_name_variations = [
                        statute_name.lower(),
                        statute_name.replace(' ', '').lower(),
                        ''.join([word for word in statute_name.split()]).lower()  # Acronym
                    ]
                    
                    name_mentions = 0
                    for variation in statute_name_variations:
                        name_mentions += document_lower.count(variation)
                    
                    # Calculate overall relevance score
                    relevance_score = min((keyword_score * 0.1 + name_mentions * 0.5), 1.0)
                    
                    if relevance_score > 0.1:  # Minimum threshold
                        matches.append({
                            'name': statute_name,
                            'area': area,
                            'relevance_score': relevance_score,
                            'matched_keywords': matched_keywords,
                            'name_mentions': name_mentions,
                            'sections': statute_data.get('sections', {}),
                            'identification_method': 'database_keyword'
                        })
        
        # Sort by relevance score
        matches.sort(key=lambda x: x['relevance_score'], reverse=True)
        return matches[:10]  # Top 10 matches
    
    async def _inlegal_statute_identification(self, document_text: str, 
                                            db_matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Use InLegalBERT embeddings to enhance statute identification"""
        
        inlegal_matches = []
        
        try:
            # Process document in chunks for BERT
            max_length = 512
            chunks = [document_text[i:i+max_length*3] for i in range(0, len(document_text), max_length*3)]
            
            for chunk in chunks:
                # Get BERT embeddings
                encoded = self.legal_tokenizer(
                    chunk,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=max_length
                )
                
                with torch.no_grad():
                    outputs = self.legal_model(**encoded)
                    chunk_embedding = outputs.last_hidden_state[:, 0, :].numpy().flatten()
                
                # Analyze embedding patterns for legal statute indicators
                embedding_analysis = self._analyze_embedding_for_statutes(chunk_embedding, chunk, db_matches)
                inlegal_matches.extend(embedding_analysis)
        
        except Exception as e:
            print(f"InLegalBERT statute identification error: {e}")
        
        return inlegal_matches
    
    def _analyze_embedding_for_statutes(self, embedding: np.ndarray, chunk_text: str, 
                                      db_matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze InLegalBERT embeddings for statute relevance"""
        
        embedding_matches = []
        
        # Calculate embedding statistics
        embedding_mean = float(np.mean(embedding))
        embedding_std = float(np.std(embedding))
        
        # Enhanced heuristic: higher std deviation often indicates more complex legal language
        legal_complexity_score = min(embedding_std * 2, 1.0)
        
        # Check if chunk contains legal references
        chunk_lower = chunk_text.lower()
        legal_indicators = [
            'section', 'act', 'law', 'regulation', 'rule', 'provision',
            'statute', 'code', 'ordinance', 'amendment', 'sub-section'
        ]
        
        legal_indicator_count = sum(1 for indicator in legal_indicators if indicator in chunk_lower)
        
        if legal_indicator_count > 0 and legal_complexity_score > 0.3:
            # Try to match with database entries
            for db_match in db_matches:
                statute_name_lower = db_match['name'].lower()
                if any(word in chunk_lower for word in statute_name_lower.split()):
                    embedding_matches.append({
                        'name': db_match['name'],
                        'area': db_match.get('area', 'unknown'),
                        'relevance_score': min(legal_complexity_score + 0.2, 1.0),
                        'inlegal_confidence': legal_complexity_score,
                        'embedding_analysis': {
                            'legal_complexity': legal_complexity_score,
                            'legal_indicators': legal_indicator_count
                        },
                        'identification_method': 'inlegal_bert'
                    })
        
        return embedding_matches
    
    async def _gemini_statute_identification(self, document_text: str, document_structure: Dict[str, Any],
                                           legal_domain: str, db_matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Use Gemini for contextual statute identification with Indian legal expertise"""
        
        # Prepare context from document structure
        structure_context = ""
        if document_structure:
            structure_summary = {k: v.get('summary', '') for k, v in document_structure.items()}
            structure_context = f"Document Structure: {json.dumps(structure_summary, indent=2)}"
        
        # Prepare database context
        db_context = ""
        if db_matches:
            db_context = f"Database Matches: {[match['name'] for match in db_matches[:5]]}"
        
        statute_identification_prompt = f"""
        As an expert Indian legal AI, identify all relevant Indian statutes, acts, and legal provisions that apply to this legal document.
        
        Legal Domain: {legal_domain}
        {structure_context}
        {db_context}
        
        DOCUMENT CONTENT:
        {document_text[:2000]}
        
        Analyze the document and identify:
        
        1. EXPLICITLY MENTIONED STATUTES
           - Direct references to Indian Acts, laws, regulations
           - Section/article citations
           - Legal provisions mentioned by name
        
        2. CONTEXTUALLY APPLICABLE STATUTES
           - Indian laws that apply to the subject matter
           - Regulatory requirements based on content
           - Compliance obligations implied by document type
        
        3. DOMAIN-SPECIFIC INDIAN LAWS
           - Relevant central government acts
           - Applicable state laws if identifiable
           - Industry-specific regulations
           - Recent amendments or updates
        
        4. SECTION-SPECIFIC APPLICABILITY
           - Which sections of identified acts apply
           - Specific provisions relevant to document content
           - Cross-references between different acts
        
        Return comprehensive JSON:
        {{
            "identified_statutes": [
                {{
                    "name": "Full name of Indian Act/Law",
                    "short_name": "Common abbreviation if any",
                    "year": "Year of enactment",
                    "relevance_score": 0.0-1.0,
                    "applicability": "How this statute applies to the document",
                    "relevant_sections": [
                        {{
                            "section": "Section number/name",
                            "description": "What this section covers",
                            "relevance": "Why this section is relevant"
                        }}
                    ],
                    "compliance_requirements": "Key compliance obligations",
                    "penalties_provisions": "Relevant penalty provisions if any",
                    "recent_amendments": "Recent changes affecting applicability"
                }}
            ],
            "legal_framework_analysis": {{
                "primary_governing_laws": ["Most important applicable laws"],
                "regulatory_compliance": "Overall compliance requirements",
                "inter_statute_relationships": "How different laws interact",
                "potential_conflicts": "Any conflicting provisions identified"
            }},
            "recommendations": [
                "Specific recommendations for compliance",
                "Areas requiring legal expertise",
                "Additional statutes to consider"
            ]
        }}
        """
        
        try:
            response = await self.gemini_model.generate_content_async(statute_identification_prompt)
            
            # Parse JSON response
            response_text = response.text.strip()
            if response_text.startswith('```json'):
                response_text = response_text.replace('``````', '').strip()
            
            gemini_data = json.loads(response_text)
            
            # Convert to standard format
            gemini_matches = []
            identified_statutes = gemini_data.get('identified_statutes', [])
            
            for statute in identified_statutes:
                gemini_matches.append({
                    'name': statute.get('name', 'Unknown Statute'),
                    'short_name': statute.get('short_name', ''),
                    'year': statute.get('year', ''),
                    'area': self._determine_legal_area(statute.get('name', '')),
                    'relevance_score': statute.get('relevance_score', 0.7),
                    'applicability': statute.get('applicability', ''),
                    'relevant_sections': statute.get('relevant_sections', []),
                    'compliance_requirements': statute.get('compliance_requirements', ''),
                    'identification_method': 'gemini_contextual',
                    'gemini_analysis': {
                        'legal_framework': gemini_data.get('legal_framework_analysis', {}),
                        'recommendations': gemini_data.get('recommendations', [])
                    }
                })
            
            return gemini_matches
            
        except json.JSONDecodeError as e:
            print(f"Gemini statute identification JSON error: {e}")
            return []
        except Exception as e:
            print(f"Gemini statute identification error: {e}")
            return []
    
    def _determine_legal_area(self, statute_name: str) -> str:
        """Determine legal area based on statute name"""
        
        statute_lower = statute_name.lower()
        
        area_indicators = {
            'corporate_law': ['companies act', 'sebi', 'corporate', 'securities'],
            'contract_law': ['contract act', 'specific relief', 'negotiable instruments'],
            'employment_law': ['industrial relations', 'factories act', 'payment of wages', 'employment'],
            'it_cyber_law': ['information technology', 'cyber', 'data protection', 'digital'],
            'banking_finance': ['banking regulation', 'sarfaesi', 'rbi', 'financial'],
            'property_law': ['transfer of property', 'registration act', 'property'],
            'tax_law': ['income tax', 'gst', 'goods and services tax', 'customs']
        }
        
        for area, indicators in area_indicators.items():
            if any(indicator in statute_lower for indicator in indicators):
                return area
        
        return 'general'
    
    def _consolidate_statute_matches(self, db_matches: List[Dict[str, Any]], 
                                   inlegal_matches: List[Dict[str, Any]], 
                                   gemini_matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Consolidate and rank statute matches from all sources"""
        
        consolidated = {}
        
        # Process database matches
        for match in db_matches:
            statute_name = match['name']
            consolidated[statute_name] = match.copy()
            consolidated[statute_name]['sources'] = ['database']
        
        # Enhance with InLegalBERT matches
        for match in inlegal_matches:
            statute_name = match['name']
            if statute_name in consolidated:
                # Merge with existing
                consolidated[statute_name]['inlegal_confidence'] = match.get('inlegal_confidence', 0.5)
                consolidated[statute_name]['embedding_analysis'] = match.get('embedding_analysis', {})
                consolidated[statute_name]['sources'].append('inlegal_bert')
                # Average the relevance scores
                consolidated[statute_name]['relevance_score'] = (
                    consolidated[statute_name]['relevance_score'] + match['relevance_score']
                ) / 2
            else:
                consolidated[statute_name] = match.copy()
                consolidated[statute_name]['sources'] = ['inlegal_bert']
        
        # Enhance with Gemini matches
        for match in gemini_matches:
            statute_name = match['name']
            if statute_name in consolidated:
                # Merge with existing
                consolidated[statute_name]['applicability'] = match.get('applicability', '')
                consolidated[statute_name]['relevant_sections'] = match.get('relevant_sections', [])
                consolidated[statute_name]['compliance_requirements'] = match.get('compliance_requirements', '')
                consolidated[statute_name]['gemini_analysis'] = match.get('gemini_analysis', {})
                consolidated[statute_name]['sources'].append('gemini_contextual')
                # Weighted average favoring Gemini for contextual accuracy
                consolidated[statute_name]['relevance_score'] = (
                    consolidated[statute_name]['relevance_score'] * 0.7 + match['relevance_score'] * 0.3
                )
            else:
                consolidated[statute_name] = match.copy()
                consolidated[statute_name]['sources'] = ['gemini_contextual']
        
        # Convert back to list and sort by relevance
        result = list(consolidated.values())
        result.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Add consolidated confidence scores
        for statute in result:
            source_count = len(statute.get('sources', []))
            base_relevance = statute.get('relevance_score', 0.5)
            
            # Boost confidence for multi-source matches
            if source_count >= 3:
                statute['consolidated_confidence'] = min(base_relevance + 0.2, 1.0)
            elif source_count >= 2:
                statute['consolidated_confidence'] = min(base_relevance + 0.1, 1.0)
            else:
                statute['consolidated_confidence'] = base_relevance
        
        return result[:15]  # Top 15 most relevant statutes
    
    async def _generate_framework_summary(self, statutes: List[Dict[str, Any]], legal_domain: str) -> str:
        """Generate comprehensive legal framework summary"""
        
        if not statutes:
            return f"No specific Indian legal framework identified for {legal_domain} domain. General Indian legal principles apply."
        
        framework_prompt = f"""
        Generate a comprehensive legal framework summary for this {legal_domain} matter based on these identified Indian statutes:
        
        Identified Statutes: {json.dumps([{
            'name': s['name'], 
            'relevance': s['relevance_score'],
            'applicability': s.get('applicability', 'General applicability')
        } for s in statutes[:5]], indent=2)}
        
        Provide a concise 2-3 sentence summary explaining:
        1. The primary legal framework governing this matter
        2. Key compliance requirements and obligations
        3. Interconnections between applicable laws
        4. Overall regulatory landscape
        
        Focus on practical implications for Indian legal practitioners.
        """
        
        try:
            response = await self.gemini_model.generate_content_async(framework_prompt)
            return response.text.strip()
        except Exception as e:
            # Fallback summary
            primary_statutes = [s['name'] for s in statutes[:3]]
            return f"This {legal_domain} matter is primarily governed by {', '.join(primary_statutes)}. The legal framework encompasses {len(statutes)} applicable Indian statutes with varying degrees of relevance. Comprehensive compliance review recommended to ensure adherence to all applicable provisions."
    
    def _calculate_identification_confidence(self, statutes: List[Dict[str, Any]]) -> float:
        """Calculate overall statute identification confidence"""
        
        if not statutes:
            return 0.2
        
        # Weight by relevance and source diversity
        weighted_confidences = []
        for statute in statutes:
            base_confidence = statute.get('consolidated_confidence', statute.get('relevance_score', 0.5))
            source_multiplier = min(len(statute.get('sources', [])) * 0.2, 0.4)  # Max 0.4 bonus
            weighted_confidence = min(base_confidence + source_multiplier, 1.0)
            weighted_confidences.append(weighted_confidence)
        
        # Average of top 5 statute confidences
        top_confidences = sorted(weighted_confidences, reverse=True)[:5]
        return sum(top_confidences) / len(top_confidences)
    
    async def _fallback_statute_identification(self, document_text: str, legal_domain: str, error_msg: str) -> Dict[str, Any]:
        """Fallback statute identification when main methods fail"""
        
        # Simple pattern-based statute detection
        fallback_statutes = []
        document_lower = document_text.lower()
        
        # Common Indian legal act patterns
        act_patterns = [
            'companies act 2013', 'contract act 1872', 'income tax act 1961',
            'information technology act 2000', 'sebi act 1992', 'banking regulation act'
        ]
        
        for pattern in act_patterns:
            if pattern in document_lower:
                fallback_statutes.append({
                    'name': pattern.title(),
                    'area': self._determine_legal_area(pattern),
                    'relevance_score': 0.6,
                    'identification_method': 'pattern_fallback',
                    'applicability': 'Pattern-based identification - manual verification recommended'
                })
        
        return {
            'statutes': fallback_statutes,
            'framework_summary': f'Fallback statute identification for {legal_domain} domain. Error: {error_msg}. Manual legal review strongly recommended.',
            'identification_confidence': 0.3,
            'legal_domain': legal_domain,
            'identification_metadata': {
                'method': 'fallback_pattern',
                'error': error_msg,
                'total_statutes_found': len(fallback_statutes),
                'timestamp': datetime.now().isoformat()
            }
        }
