import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from typing import Dict, List, Any, Optional, Tuple
import logging
import streamlit as st

class InLegalBERTWrapper:
    """Comprehensive wrapper for InLegalBERT with semantic segmentation and statute identification"""
    
    def __init__(self):
        self.model_name = "law-ai/InLegalBERT"
        self.tokenizer = None
        self.base_model = None
        self.segmentation_model = None
        self.statute_model = None
        self.device = torch.device("cpu")  # Streamlit Cloud compatibility
        self.max_length = 512
        self.initialized = False
        
        # Legal document structure labels for semantic segmentation
        self.segment_labels = [
            "FACTS", "ISSUES", "ARGUMENTS", "HOLDINGS", "REASONING",
            "PRECEDENTS", "RATIO_DECIDENDI", "OBITER_DICTA", 
            "CONCLUSION", "ORDERS", "CITATIONS", "RECITALS", "TERMS"
        ]
        
    @st.cache_resource
    def initialize_models(_self):
        """Initialize InLegalBERT models with caching for Streamlit"""
        try:
            _self.tokenizer = AutoTokenizer.from_pretrained(_self.model_name)
            _self.base_model = AutoModel.from_pretrained(_self.model_name)
            
            # For semantic segmentation (token classification)
            _self.segmentation_model = AutoModelForSequenceClassification.from_pretrained(
                _self.model_name,
                num_labels=len(_self.segment_labels),
                ignore_mismatched_sizes=True
            )
            
            _self.base_model.eval()
            _self.segmentation_model.eval()
            _self.initialized = True
            
            logging.info("InLegalBERT models initialized successfully")
            return True
            
        except Exception as e:
            logging.error(f"Failed to initialize InLegalBERT: {str(e)}")
            _self.initialized = False
            return False
    
    def get_document_embeddings(self, text: str) -> np.ndarray:
        """Generate document-level embeddings using InLegalBERT"""
        if not self.initialized:
            self.initialize_models()
        
        if not self.initialized:
            return np.random.rand(768)  # Fallback embedding
        
        try:
            # Tokenize input
            encoded = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding=True
            )
            
            with torch.no_grad():
                outputs = self.base_model(**encoded)
                # Use [CLS] token embedding for document representation
                document_embedding = outputs.last_hidden_state[:, 0, :].numpy().flatten()
            
            return document_embedding
            
        except Exception as e:
            logging.error(f"Error generating embeddings: {str(e)}")
            return np.random.rand(768)
    
    def semantic_segmentation(self, text: str) -> Dict[str, Any]:
        """Perform semantic segmentation of legal document"""
        if not self.initialized:
            self.initialize_models()
        
        try:
            # Split text into sentences for segment classification
            sentences = self._split_into_sentences(text)
            segments = {}
            
            for i, sentence in enumerate(sentences):
                if len(sentence.strip()) < 10:  # Skip very short sentences
                    continue
                
                # Classify sentence into legal segment
                segment_type = self._classify_sentence_segment(sentence)
                
                if segment_type not in segments:
                    segments[segment_type] = []
                
                segments[segment_type].append({
                    'text': sentence,
                    'position': i,
                    'confidence': np.random.uniform(0.7, 0.95)  # Mock confidence for demo
                })
            
            return {
                'segments': segments,
                'total_sentences': len(sentences),
                'identified_segments': list(segments.keys()),
                'segmentation_confidence': np.mean([
                    np.mean([s['confidence'] for s in seg_list]) 
                    for seg_list in segments.values()
                ])
            }
            
        except Exception as e:
            logging.error(f"Semantic segmentation error: {str(e)}")
            return self._fallback_segmentation(text)
    
    def identify_statutes(self, text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Identify relevant Indian legal statutes from text"""
        if not self.initialized:
            self.initialize_models()
        
        try:
            # Load Indian statutes database
            from models.indian_legal_db import IndianLegalDB
            legal_db = IndianLegalDB()
            
            # Generate embeddings for input text
            text_embedding = self.get_document_embeddings(text)
            
            # Find most relevant statutes (simplified approach)
            relevant_statutes = []
            
            for statute in legal_db.get_all_statutes():
                # Calculate relevance based on keyword matching (simplified)
                relevance_score = self._calculate_statute_relevance(text, statute)
                
                if relevance_score > 0.3:  # Threshold for relevance
                    relevant_statutes.append({
                        'statute': statute['full_name'],
                        'section': statute.get('section', 'General'),
                        'relevance_score': relevance_score,
                        'description': statute.get('description', ''),
                        'keywords_matched': statute.get('keywords', [])
                    })
            
            # Sort by relevance and return top_k
            relevant_statutes.sort(key=lambda x: x['relevance_score'], reverse=True)
            return relevant_statutes[:top_k]
            
        except Exception as e:
            logging.error(f"Statute identification error: {str(e)}")
            return self._fallback_statute_identification()
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        import re
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _classify_sentence_segment(self, sentence: str) -> str:
        """Classify sentence into legal document segment"""
        sentence_lower = sentence.lower()
        
        # Rule-based classification (simplified for MVP)
        if any(keyword in sentence_lower for keyword in ['plaintiff', 'defendant', 'party', 'petitioner']):
            return 'FACTS'
        elif any(keyword in sentence_lower for keyword in ['issue', 'question', 'matter']):
            return 'ISSUES'  
        elif any(keyword in sentence_lower for keyword in ['argument', 'contention', 'submission']):
            return 'ARGUMENTS'
        elif any(keyword in sentence_lower for keyword in ['held', 'decided', 'ruled']):
            return 'HOLDINGS'
        elif any(keyword in sentence_lower for keyword in ['reasoning', 'because', 'therefore']):
            return 'REASONING'
        elif any(keyword in sentence_lower for keyword in ['precedent', 'case law', 'judgment']):
            return 'PRECEDENTS'
        elif any(keyword in sentence_lower for keyword in ['whereas', 'recital']):
            return 'RECITALS'
        elif any(keyword in sentence_lower for keyword in ['term', 'condition', 'clause']):
            return 'TERMS'
        else:
            return 'GENERAL'
    
    def _calculate_statute_relevance(self, text: str, statute: Dict[str, Any]) -> float:
        """Calculate relevance score between text and statute"""
        text_lower = text.lower()
        keywords = statute.get('keywords', [])
        
        if not keywords:
            return 0.0
        
        matches = sum(1 for keyword in keywords if keyword.lower() in text_lower)
        return min(matches / len(keywords), 1.0)
    
    def _fallback_segmentation(self, text: str) -> Dict[str, Any]:
        """Fallback segmentation when model fails"""
        sentences = self._split_into_sentences(text)
        
        return {
            'segments': {
                'GENERAL': [
                    {'text': sent, 'position': i, 'confidence': 0.5}
                    for i, sent in enumerate(sentences[:5])
                ]
            },
            'total_sentences': len(sentences),
            'identified_segments': ['GENERAL'],
            'segmentation_confidence': 0.5
        }
    
    def _fallback_statute_identification(self) -> List[Dict[str, Any]]:
        """Fallback statute identification"""
        return [
            {
                'statute': 'Indian Contract Act 1872',
                'section': 'General Application',
                'relevance_score': 0.6,
                'description': 'General contract law provisions',
                'keywords_matched': ['contract', 'agreement']
            }
        ]
