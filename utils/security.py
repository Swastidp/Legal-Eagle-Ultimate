"""
Simple security manager for Legal Eagle MVP
"""
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class SecurityManager:
    """Simple security manager for demo"""
    
    def __init__(self):
        self.session_active = True
        logger.info("Simple security manager initialized")
    
    def validate_session(self) -> bool:
        """Simple session validation"""
        return self.session_active
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get security status"""
        return {
            'session_active': self.session_active,
            'security_level': 'basic',
            'encryption_enabled': False
        }
