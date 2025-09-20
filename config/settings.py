import os
import streamlit as st
from typing import Dict, Any, Optional, List
import toml
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Config:
    """Comprehensive configuration for Legal Eagle MVP with robust secret handling"""
    
    def __init__(self):
        # Initialize with multiple fallback methods for API keys
        self.gemini_api_key = self._get_api_key()
        self.google_cloud_project = self._get_secret('GOOGLE_CLOUD_PROJECT')
        self.google_credentials_path = self._get_secret('GOOGLE_APPLICATION_CREDENTIALS')
        
        # Validate critical configurations
        self._validate_critical_config()
        
        # Model configurations
        self.model_configs = {
            'inlegal_bert': {
                'model_name': 'law-ai/InLegalBERT',
                'max_length': 512,
                'batch_size': 8,
                'device': 'cpu'  # Streamlit Cloud compatibility
            },
            'gemini': {
                'model_name': 'gemini-2.0-flash-exp',
                'temperature': 0.3,
                'max_tokens': 2048,
                'top_p': 0.8,
                'top_k': 40
            },
            'gemini_pro': {
                'model_name': 'gemini-2.0-flash-thinking-exp',
                'temperature': 0.2,
                'max_tokens': 4096,
                'top_p': 0.9,
                'top_k': 40
            }
        }
        
        # Enhanced document processing settings
        self.document_processing = {
            'max_file_size_mb': 50,
            'supported_formats': ['.pdf', '.docx', '.txt', '.doc'],
            'chunk_size': 1000,
            'chunk_overlap': 100,
            'max_chunks_per_document': 50,
            'ocr_enabled': True,
            'extract_metadata': True,
            'extract_tables': True,
            'extract_images': False  # Disabled for MVP
        }
        
        # Enhanced legal domain settings for Indian law
        self.legal_domains = {
            'primary_acts': [
                'Indian Contract Act, 1872',
                'Companies Act, 2013', 
                'Information Technology Act, 2000',
                'Consumer Protection Act, 2019',
                'Arbitration and Conciliation Act, 2015',
                'Indian Evidence Act, 1872',
                'Code of Civil Procedure, 1908',
                'Code of Criminal Procedure, 1973',
                'Indian Penal Code, 1860',
                'Constitution of India, 1950'
            ],
            'regulatory_bodies': [
                'Securities and Exchange Board of India (SEBI)',
                'Reserve Bank of India (RBI)', 
                'Ministry of Corporate Affairs (MCA)',
                'Competition Commission of India (CCI)',
                'Telecom Regulatory Authority of India (TRAI)'
            ],
            'court_hierarchy': [
                'Supreme Court of India',
                'High Courts',
                'District Courts',  
                'Sessions Courts',
                'Magistrate Courts'
            ]
        }
        
        # Enhanced application settings - FIXED the debug_mode issue
        self.app_settings = {
            'app_name': 'Legal Eagle - AI Legal Intelligence',
            'version': '1.0.0-MVP',
            'environment': self._get_environment(),
            'debug_mode': self._get_debug_mode(),  # FIXED: separate method
            'session_timeout_minutes': 60,
            'max_chat_history': 50,
            'enable_analytics': True,
            'enable_feedback': True,
            'default_language': 'en',
            'supported_languages': ['en', 'hi']
        }
        
        # Enhanced security settings
        self.security_settings = {
            'enable_rate_limiting': True,
            'max_requests_per_minute': 30,
            'enable_input_sanitization': True,
            'enable_output_filtering': True,
            'max_input_length': 10000,
            'allowed_file_types': ['pdf', 'docx', 'txt', 'doc'],
            'scan_for_pii': True,
            'enable_audit_logging': True
        }
        
        logger.info(f"✅ Configuration initialized successfully")
        logger.info(f"Environment: {self.app_settings['environment']}")
        logger.info(f"Debug mode: {self.app_settings['debug_mode']}")
    
    def _get_api_key(self) -> Optional[str]:
        """Get Gemini API key with multiple fallback methods"""
        # Method 1: Streamlit secrets (recommended for deployment)
        try:
            if hasattr(st, 'secrets') and 'GEMINI_API_KEY' in st.secrets:
                api_key = st.secrets['GEMINI_API_KEY']
                logger.info("✅ Using Gemini API key from Streamlit secrets")
                return api_key
        except Exception as e:
            logger.debug(f"Streamlit secrets not available: {e}")
        
        # Method 2: Environment variable
        api_key = os.getenv('GEMINI_API_KEY')
        if api_key:
            logger.info("✅ Using Gemini API key from environment variable")
            return api_key
        
        # Method 3: .env file
        try:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.getenv('GEMINI_API_KEY')
            if api_key:
                logger.info("✅ Using Gemini API key from .env file")
                return api_key
        except ImportError:
            pass
        
        logger.warning("⚠️ GEMINI_API_KEY not found in any configuration method")
        return None
    
    def _get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get secret value with fallback methods"""
        # Streamlit secrets
        try:
            if hasattr(st, 'secrets') and key in st.secrets:
                return str(st.secrets[key])
        except:
            pass
        
        # Environment variable
        value = os.getenv(key, default)
        if value:
            return str(value)
        
        return default
    
    def _get_debug_mode(self) -> bool:
        """Get debug mode setting with proper type handling"""
        debug_value = self._get_secret('DEBUG', 'false')
        
        # Handle different possible values
        if isinstance(debug_value, bool):
            return debug_value
        elif isinstance(debug_value, str):
            return debug_value.lower() in ('true', '1', 'yes', 'on')
        else:
            return False
    
    def _get_environment(self) -> str:
        """Detect the current environment"""
        if os.getenv('STREAMLIT_SHARING') == '1':
            return 'streamlit_cloud'
        elif os.getenv('RAILWAY_ENVIRONMENT'):
            return 'railway'
        elif os.getenv('HEROKU_APP_NAME'):
            return 'heroku'
        else:
            return 'local'
    
    def _validate_critical_config(self):
        """Validate critical configuration settings"""
        issues = []
        
        # Check API key
        if not self.gemini_api_key:
            issues.append("❌ GEMINI_API_KEY is required but not found")
        elif len(self.gemini_api_key) < 10:
            issues.append("❌ GEMINI_API_KEY appears to be invalid (too short)")
        
        # Log validation results
        if issues:
            logger.warning("Configuration validation issues found:")
            for issue in issues:
                logger.warning(f"  {issue}")
        else:
            logger.info("✅ All critical configurations validated successfully")
        
        self.validation_issues = issues
    
    def validate_configuration(self) -> Dict[str, bool]:
        """Validate all configurations and return status"""
        return {
            'gemini_api_key': bool(self.gemini_api_key and len(self.gemini_api_key) > 20),
            'model_configs': bool(self.model_configs),
            'app_settings': bool(self.app_settings),
            'legal_domains': bool(self.legal_domains),
            'security_settings': bool(self.security_settings)
        }
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get configuration for a specific model"""
        return self.model_configs.get(model_name, {})
    
    def is_valid(self) -> bool:
        """Check if configuration is valid for application startup"""
        return len(getattr(self, 'validation_issues', [])) == 0
    
    def get_legal_context(self, domain: str = None) -> Dict[str, List[str]]:
        """Get legal context for specific domain or all domains"""
        if domain and domain in self.legal_domains:
            return {domain: self.legal_domains[domain]}
        return self.legal_domains

# Singleton pattern
_config_instance = None

def get_config() -> Config:
    """Get singleton configuration instance"""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance

def validate_setup() -> bool:
    """Quick validation check"""
    config = get_config()
    validation = config.validate_configuration()
    return validation.get('gemini_api_key', False)

if __name__ == "__main__":
    config = Config()
    print("✅ Configuration Ready!" if config.is_valid() else "❌ Setup Required")
