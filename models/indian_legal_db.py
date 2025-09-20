import json
import os
from typing import Dict, List, Any, Optional
import streamlit as st

class IndianLegalDB:
    """Comprehensive Indian Legal Database Manager"""
    
    def __init__(self):
        self.statutes_data = None
        self.db_path = "data/indian_statutes.json"
        self.load_database()
    
    @st.cache_data
    def load_database(_self):
        """Load Indian statutes database with caching"""
        try:
            with open(_self.db_path, 'r', encoding='utf-8') as f:
                _self.statutes_data = json.load(f)
            return True
        except FileNotFoundError:
            # Create default database if file doesn't exist
            _self.statutes_data = _self._create_default_database()
            _self._save_database()
            return True
        except Exception as e:
            print(f"Error loading database: {e}")
            _self.statutes_data = _self._create_default_database()
            return False
    
    def get_all_statutes(self) -> List[Dict[str, Any]]:
        """Get all statutes as a flat list for processing"""
        if not self.statutes_data:
            return []
        
        statutes_list = []
        
        for act_key, act_data in self.statutes_data.items():
            if 'sections' in act_data:
                for section_num, section_data in act_data['sections'].items():
                    statutes_list.append({
                        'act': act_key,
                        'full_name': act_data['full_name'],
                        'section': section_num,
                        'title': section_data['title'],
                        'description': section_data['description'],
                        'keywords': section_data.get('keywords', [])
                    })
            elif isinstance(act_data, dict) and 'full_name' in act_data:
                # Handle nested structures like labour_codes
                for sub_key, sub_data in act_data.items():
                    if sub_key != 'full_name' and isinstance(sub_data, dict) and 'sections' in sub_data:
                        for section_num, section_data in sub_data['sections'].items():
                            statutes_list.append({
                                'act': f"{act_key}.{sub_key}",
                                'full_name': f"{act_data['full_name']} - {sub_key.replace('_', ' ').title()}",
                                'section': section_num,
                                'title': section_data['title'],
                                'description': section_data['description'],
                                'keywords': section_data.get('keywords', [])
                            })
        
        return statutes_list
    
    def search_statutes(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search statutes by query"""
        query_lower = query.lower()
        results = []
        
        for statute in self.get_all_statutes():
            score = 0
            
            # Check title match
            if query_lower in statute['title'].lower():
                score += 3
            
            # Check description match
            if query_lower in statute['description'].lower():
                score += 2
            
            # Check keyword match
            for keyword in statute['keywords']:
                if query_lower in keyword.lower() or keyword.lower() in query_lower:
                    score += 1
            
            if score > 0:
                statute['relevance_score'] = score
                results.append(statute)
        
        # Sort by relevance and return top results
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        return results[:max_results]
    
    def get_statute_by_reference(self, act: str, section: str = None) -> Optional[Dict[str, Any]]:
        """Get specific statute by act and section"""
        if not self.statutes_data or act not in self.statutes_data:
            return None
        
        act_data = self.statutes_data[act]
        
        if section and 'sections' in act_data and section in act_data['sections']:
            section_data = act_data['sections'][section]
            return {
                'act': act,
                'full_name': act_data['full_name'],
                'section': section,
                'title': section_data['title'],
                'description': section_data['description'],
                'keywords': section_data.get('keywords', [])
            }
        
        return {
            'act': act,
            'full_name': act_data.get('full_name', act),
            'description': f"General provisions of {act_data.get('full_name', act)}"
        }
    
    def get_related_statutes(self, keywords: List[str], max_results: int = 5) -> List[Dict[str, Any]]:
        """Get statutes related to given keywords"""
        results = []
        
        for statute in self.get_all_statutes():
            relevance_score = 0
            
            for keyword in keywords:
                for statute_keyword in statute['keywords']:
                    if keyword.lower() in statute_keyword.lower():
                        relevance_score += 1
                
                if keyword.lower() in statute['title'].lower():
                    relevance_score += 2
                
                if keyword.lower() in statute['description'].lower():
                    relevance_score += 1
            
            if relevance_score > 0:
                statute['relevance_score'] = relevance_score
                results.append(statute)
        
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        return results[:max_results]
    
    def _create_default_database(self) -> Dict[str, Any]:
        """Create a default database structure"""
        return {
            "companies_act_2013": {
                "full_name": "The Companies Act, 2013",
                "sections": {
                    "2": {
                        "title": "Definitions",
                        "description": "Key definitions for company law terms",
                        "keywords": ["company", "director", "member", "promoter"]
                    },
                    "166": {
                        "title": "Duties of directors",
                        "description": "Fiduciary duties and responsibilities of directors",
                        "keywords": ["fiduciary duty", "care", "diligence", "conflict of interest"]
                    }
                }
            },
            "it_act_2000": {
                "full_name": "Information Technology Act, 2000",
                "sections": {
                    "43A": {
                        "title": "Compensation for failure to protect data",
                        "description": "Corporate liability for data breaches and privacy violations",
                        "keywords": ["data protection", "privacy", "sensitive personal data", "compensation"]
                    }
                }
            }
        }
    
    def _save_database(self):
        """Save database to file"""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            with open(self.db_path, 'w', encoding='utf-8') as f:
                json.dump(self.statutes_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving database: {e}")
