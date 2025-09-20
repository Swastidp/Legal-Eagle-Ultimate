"""
Cloud-safe configuration management
"""
import os
import logging
from typing import Dict, Any
import streamlit as st

logger = logging.getLogger(__name__)

class Config:
    """Cloud-safe configuration class"""
    
    def __init__(self):
        # Get Gemini API key from secrets or environment
        self.gemini_api_key = self._get_secret('GEMINI_API_KEY')
        
        # Optional Google Cloud configuration
        self.google_cloud_project = self._get_secret('GOOGLE_CLOUD_PROJECT', required=False)
        
        logger.info("Configuration loaded for cloud deployment")
    
    def _get_secret(self, key: str, required: bool = True) -> str:
        """Get secret from Streamlit secrets or environment"""
        try:
            # Try Streamlit secrets first (for cloud deployment)
            if hasattr(st, 'secrets'):
                try:
                    value = st.secrets.get(key)
                    if value:
                        return value
                except:
                    pass
            
            # Try environment variable
            value = os.getenv(key)
            if value:
                return value
            
            # If required and not found, raise error
            if required:
                raise ValueError(f"Required secret '{key}' not found in Streamlit secrets or environment variables")
            
            return None
            
        except Exception as e:
            if required:
                logger.error(f"Failed to get required secret '{key}': {e}")
                raise
            else:
                logger.debug(f"Optional secret '{key}' not found: {e}")
                return None
    
    def validate_configuration(self) -> Dict[str, bool]:
        """Validate configuration - cloud safe"""
        validation = {
            'gemini_api_key': bool(self.gemini_api_key),
            'google_cloud_project': bool(self.google_cloud_project)
        }
        
        logger.info(f"Configuration validation: {validation}")
        return validation

def get_config() -> Config:
    """Get configuration instance"""
    return Config()

def validate_setup() -> Dict[str, Any]:
    """Validate setup for cloud deployment"""
    try:
        config = get_config()
        validation = config.validate_configuration()
        
        return {
            'valid': validation['gemini_api_key'],  # Only Gemini is required
            'details': validation,
            'deployment_ready': True
        }
        
    except Exception as e:
        return {
            'valid': False,
            'error': str(e),
            'deployment_ready': False
        }
