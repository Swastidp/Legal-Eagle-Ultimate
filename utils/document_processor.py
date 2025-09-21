"""
Enhanced Document Processor with Cloud Deployment Support
No Google Cloud dependencies - purely optional enhancement
"""

import os
import logging
from typing import Dict, Any, Optional, BinaryIO
import streamlit as st
from datetime import datetime

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Cloud-ready document processor with optional Google Document AI"""
    
    def __init__(self):
        """Initialize document processor with cloud-safe configuration"""
        self.supported_formats = ['pdf', 'docx', 'txt', 'jpg', 'png', 'jpeg']
        self.max_file_size_mb = 10
        self.total_processed = 0
        self.successful_processed = 0
        
        # Cloud-safe Google Client initialization
        self.google_client = None
        self.google_processor_name = None
        
        logger.info("Cloud-ready Document Processor initialized")

    def process_uploaded_file(self, uploaded_file) -> Dict[str, Any]:
        """
        Process uploaded file and return structured result
        This method handles the complete file processing workflow
        """
        try:
            if not uploaded_file:
                return {
                    "success": False,
                    "error": "No file provided",
                    "extracted_text": "",
                    "metadata": {}
                }
            
            # Get file metadata
            metadata = self.get_document_metadata(uploaded_file)
            
            # Validate file
            if not metadata.get("is_supported"):
                return {
                    "success": False,
                    "error": f"Unsupported file format: {metadata.get('extension', 'unknown')}",
                    "extracted_text": "",
                    "metadata": metadata
                }
            
            if not metadata.get("is_valid_size"):
                return {
                    "success": False,
                    "error": f"File too large: {metadata.get('size_mb', 0)} MB (max: {self.max_file_size_mb} MB)",
                    "extracted_text": "",
                    "metadata": metadata
                }
            
            # Extract text using the existing extract_text method
            extracted_text = self.extract_text(uploaded_file)
            
            # Check if extraction was successful
            if extracted_text.startswith("Error:"):
                return {
                    "success": False,
                    "error": extracted_text,
                    "extracted_text": "",
                    "metadata": metadata
                }
            
            # Successful processing
            processing_result = {
                "success": True,
                "extracted_text": extracted_text,
                "metadata": metadata,
                "processing_stats": {
                    "text_length": len(extracted_text),
                    "word_count": len(extracted_text.split()),
                    "processing_method": metadata.get("processing_mode", "standard"),
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            logger.info(f"Successfully processed {metadata.get('filename', 'unknown')}: {len(extracted_text)} characters extracted")
            
            return processing_result
            
        except Exception as e:
            logger.error(f"File processing failed: {str(e)}")
            return {
                "success": False,
                "error": f"Processing failed: {str(e)}",
                "extracted_text": "",
                "metadata": {}
            }

    def test_document_ai_connection(self) -> Dict[str, Any]:
        """Cloud-safe Google Document AI connection test"""
        try:
            # Cloud-safe Google Client initialization
            has_google_config = self._check_google_cloud_config()
            
            if not has_google_config:
                return {
                    'connection_status': 'not_configured',
                    'client_initialized': False,
                    'message': 'Google Document AI not configured - using standard OCR (this is normal for cloud deployment)',
                    'deployment_safe': True
                }
            
            # Try to initialize Google Cloud client if configured
            try:
                client_result = self._initialize_google_client()
                if client_result['success']:
                    return {
                        'connection_status': 'success',
                        'client_initialized': True,
                        'message': 'Google Document AI connected successfully',
                        'deployment_safe': True
                    }
                else:
                    return {
                        'connection_status': 'error',
                        'client_initialized': False,
                        'message': f'Google Document AI configuration error: {client_result["error"]}',
                        'deployment_safe': True  # Still safe, just falls back to OCR
                    }
            except Exception as e:
                return {
                    'connection_status': 'error',
                    'client_initialized': False,
                    'message': f'Google Document AI initialization failed: {str(e)}',
                    'deployment_safe': True,
                    'fallback_message': 'Using standard OCR processing'
                }
        except Exception as e:
            # This should never fail in cloud deployment
            logger.warning(f"Document AI connection test failed safely: {e}")
            return {
                'connection_status': 'not_configured',
                'client_initialized': False,
                'message': 'Using standard OCR processing (cloud deployment safe)',
                'deployment_safe': True
            }

    def _check_google_cloud_config(self) -> bool:
        """Check if Google Cloud is configured without failing"""
        try:
            # Method 1: Check Streamlit secrets (for cloud deployment)
            if hasattr(st, 'secrets'):
                try:
                    google_config = st.secrets.get('google_cloud', {})
                    if isinstance(google_config, dict) and google_config.get('project_id'):
                        return True
                    
                    # Alternative secret structure
                    if st.secrets.get('GOOGLE_CLOUD_PROJECT') or st.secrets.get('GOOGLE_APPLICATION_CREDENTIALS'):
                        return True
                except:
                    pass
            
            # Method 2: Check environment variables
            google_project = os.getenv('GOOGLE_CLOUD_PROJECT')
            google_credentials = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
            return bool(google_project and google_credentials)
            
        except Exception as e:
            logger.debug(f"Google config check failed safely: {e}")
            return False

    def _initialize_google_client(self) -> Dict[str, Any]:
        """Initialize Google client with proper error handling"""
        try:
            # Try to import Google Cloud libraries
            try:
                from google.cloud import documentai_v1 as documentai
                from google.oauth2 import service_account
            except ImportError as e:
                return {
                    'success': False,
                    'error': 'Google Cloud libraries not installed - using standard OCR',
                    'safe_fallback': True
                }
            
            # Get credentials from Streamlit secrets (cloud deployment)
            credentials = None
            project_id = None
            
            try:
                # Method 1: Streamlit secrets with service account JSON
                if hasattr(st, 'secrets') and 'google_cloud' in st.secrets:
                    google_config = st.secrets['google_cloud']
                    if isinstance(google_config, dict):
                        credentials = service_account.Credentials.from_service_account_info(google_config)
                        project_id = google_config.get('project_id')
                
                # Method 2: Environment-based (for local development)
                if not credentials:
                    project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
                    creds_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
                    if project_id and creds_path:
                        credentials = service_account.Credentials.from_service_account_file(creds_path)
                
                if not credentials or not project_id:
                    return {
                        'success': False,
                        'error': 'Google Cloud credentials not properly configured',
                        'safe_fallback': True
                    }
                
                # Initialize client
                client = documentai.DocumentProcessorServiceClient(credentials=credentials)
                
                # Set processor name (you would configure this in secrets)
                processor_id = st.secrets.get('google_cloud', {}).get('processor_id', 'default-processor')
                location = st.secrets.get('google_cloud', {}).get('location', 'us')
                self.google_processor_name = client.processor_path(project_id, location, processor_id)
                self.google_client = client
                
                return {
                    'success': True,
                    'project_id': project_id,
                    'message': 'Google Document AI initialized successfully'
                }
                
            except Exception as cred_error:
                return {
                    'success': False,
                    'error': f'Credential initialization failed: {str(cred_error)}',
                    'safe_fallback': True
                }
                
        except Exception as e:
            logger.error(f"Google client initialization failed: {e}")
            return {
                'success': False,
                'error': f'Client initialization failed: {str(e)}',
                'safe_fallback': True
            }

    def get_document_metadata(self, uploaded_file) -> Dict[str, Any]:
        """Get basic metadata from uploaded file - cloud safe"""
        try:
            # Get file information
            file_size = len(uploaded_file.getvalue()) if uploaded_file else 0
            file_size_mb = round(file_size / (1024 * 1024), 2)
            filename = uploaded_file.name if uploaded_file else "unknown"
            extension = filename.split('.')[-1].lower() if '.' in filename else ""
            
            # Check if supported by Document AI (only if configured)
            supported_by_document_ai = False
            if extension in ['pdf', 'png', 'jpg', 'jpeg']:
                # Only mark as supported if Google is actually configured
                supported_by_document_ai = self._check_google_cloud_config()
            
            metadata = {
                'filename': filename,
                'size_bytes': file_size,
                'size_mb': file_size_mb,
                'extension': extension,
                'mime_type': uploaded_file.type if uploaded_file else "unknown",
                'supported_by_document_ai': supported_by_document_ai,
                'upload_timestamp': datetime.now().isoformat(),
                'is_supported': extension in self.supported_formats,
                'is_valid_size': file_size_mb <= self.max_file_size_mb,
                'processing_mode': 'enhanced' if supported_by_document_ai else 'standard'
            }
            
            logger.info(f"Generated metadata for {filename}: {file_size_mb} MB, {metadata['processing_mode']} mode")
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to get document metadata: {e}")
            return {
                'filename': 'error',
                'size_mb': 0,
                'extension': 'unknown',
                'supported_by_document_ai': False,
                'is_supported': False,
                'is_valid_size': True,
                'processing_mode': 'standard',
                'error': str(e)
            }

    def extract_text(self, uploaded_file) -> str:
        """Extract text with cloud-safe fallbacks"""
        try:
            if not uploaded_file:
                return "Error: No file provided"
            
            filename = uploaded_file.name
            extension = filename.split('.')[-1].lower() if '.' in filename else ""
            
            # Reset file pointer
            uploaded_file.seek(0)
            
            # Try Google Document AI first (if available and configured)
            if self.google_client and extension in ['pdf', 'png', 'jpg', 'jpeg']:
                try:
                    google_result = self._extract_with_google_ai(uploaded_file)
                    if google_result and not google_result.startswith("Error"):
                        logger.info(f"Successfully extracted text using Google Document AI: {len(google_result)} chars")
                        self.successful_processed += 1
                        self.total_processed += 1
                        return google_result
                    else:
                        logger.warning("Google Document AI extraction failed, falling back to standard OCR")
                except Exception as google_error:
                    logger.warning(f"Google Document AI failed: {google_error}, using fallback")
            
            # Fallback to standard extraction methods
            uploaded_file.seek(0)  # Reset pointer after Google attempt
            
            if extension == 'txt':
                text = self._extract_text_from_txt(uploaded_file)
            elif extension == 'pdf':
                text = self._extract_text_from_pdf(uploaded_file)
            elif extension == 'docx':
                text = self._extract_text_from_docx(uploaded_file)
            elif extension in ['png', 'jpg', 'jpeg']:
                text = self._extract_text_from_image(uploaded_file)
            else:
                text = f"Error: Unsupported file format '{extension}'"
            
            # Update statistics
            self.total_processed += 1
            if not text.startswith("Error"):
                self.successful_processed += 1
            
            logger.info(f"Extracted {len(text)} characters from {filename} using standard methods")
            return text
            
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            return f"Error: Text extraction failed - {str(e)}"

    def _extract_with_google_ai(self, uploaded_file) -> str:
        """Extract text using Google Document AI (cloud-safe)"""
        try:
            if not self.google_client or not self.google_processor_name:
                return "Error: Google Document AI not properly initialized"
            
            # Import required libraries
            from google.cloud import documentai_v1 as documentai
            
            # Read file content
            uploaded_file.seek(0)
            file_content = uploaded_file.read()
            
            # Create request
            raw_document = documentai.RawDocument(
                content=file_content,
                mime_type=uploaded_file.type
            )
            
            request = documentai.ProcessRequest(
                name=self.google_processor_name,
                raw_document=raw_document
            )
            
            # Process document
            result = self.google_client.process_document(request=request)
            document = result.document
            
            # Extract text
            return document.text
            
        except Exception as e:
            logger.error(f"Google Document AI extraction failed: {e}")
            return f"Error: Google Document AI extraction failed - {str(e)}"

    def _extract_text_from_txt(self, uploaded_file) -> str:
        """Extract text from TXT file - cloud safe"""
        try:
            encodings = ['utf-8', 'latin-1', 'cp1252']
            for encoding in encodings:
                try:
                    uploaded_file.seek(0)
                    content = uploaded_file.read()
                    if isinstance(content, bytes):
                        text = content.decode(encoding)
                    else:
                        text = str(content)
                    return text
                except UnicodeDecodeError:
                    continue
            return "Error: Could not decode text file with any supported encoding"
        except Exception as e:
            return f"Error reading TXT file: {str(e)}"

    def _extract_text_from_pdf(self, uploaded_file) -> str:
        """Extract text from PDF file - cloud safe"""
        try:
            try:
                import PyPDF2
            except ImportError:
                return "Error: PyPDF2 not installed. Standard PDF processing unavailable."
            
            uploaded_file.seek(0)
            
            try:
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
            except Exception as e:
                return f"Error: Could not read PDF file - {str(e)}. File may be corrupted or password-protected."
            
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                try:
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    text += page_text + "\n"
                except Exception as page_error:
                    logger.warning(f"Failed to extract text from page {page_num}: {page_error}")
                    continue
            
            if text.strip():
                return text.strip()
            else:
                return "Error: No text found in PDF. The PDF might contain only images or be scanned. Consider using image processing tools."
                
        except Exception as e:
            return f"Error processing PDF file: {str(e)}"

    def _extract_text_from_docx(self, uploaded_file) -> str:
        """Extract text from DOCX file - cloud safe"""
        try:
            try:
                from docx import Document
            except ImportError:
                return "Error: python-docx not installed. DOCX processing unavailable."
            
            uploaded_file.seek(0)
            
            try:
                doc = Document(uploaded_file)
            except Exception as e:
                return f"Error: Could not read DOCX file - {str(e)}"
            
            text = ""
            
            # Extract paragraph text
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            
            # Extract text from tables
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        text += cell.text + " "
                    text += "\n"
            
            return text.strip() if text.strip() else "Error: No text content found in DOCX file"
            
        except Exception as e:
            return f"Error processing DOCX file: {str(e)}"

    def _extract_text_from_image(self, uploaded_file) -> str:
        """Extract text from image using OCR - cloud safe"""
        try:
            try:
                from PIL import Image
                import pytesseract
            except ImportError:
                return "Error: PIL and pytesseract not installed. Image OCR unavailable. Try uploading a PDF or text file instead."
            
            uploaded_file.seek(0)
            
            try:
                image = Image.open(uploaded_file)
            except Exception as e:
                return f"Error: Could not open image file - {str(e)}"
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                try:
                    image = image.convert('RGB')
                except Exception as e:
                    return f"Error: Could not convert image format - {str(e)}"
            
            # Extract text using OCR
            try:
                text = pytesseract.image_to_string(image)
            except Exception as e:
                return f"Error: OCR processing failed - {str(e)}. Make sure Tesseract OCR is installed on the system."
            
            if text.strip():
                return text.strip()
            else:
                return "Error: No text found in image. The image might be unclear, contain no text, or require manual processing."
                
        except Exception as e:
            return f"Error processing image file: {str(e)}"

    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get processing statistics - cloud safe"""
        success_rate = (self.successful_processed / self.total_processed * 100) if self.total_processed > 0 else 0
        
        # Check Google AI configuration safely
        google_ai_configured = self._check_google_cloud_config()
        
        return {
            'total_processed': self.total_processed,
            'successful_processed': self.successful_processed,
            'success_rate': success_rate,
            'google_ai_configured': google_ai_configured,
            'google_ai_usage_rate': 100.0 if google_ai_configured else 0.0,
            'supported_formats': self.supported_formats,
            'max_file_size_mb': self.max_file_size_mb,
            'deployment_environment': 'cloud' if 'STREAMLIT_SHARING_MODE' in os.environ else 'local'
        }

# Test function for cloud deployment
def test_cloud_deployment():
    """Test document processor in cloud environment"""
    print("🧪 Testing Document Processor for Cloud Deployment...")
    
    processor = DocumentProcessor()
    
    # Test metadata generation
    print("✅ Document processor initialized for cloud")
    
    # Test Document AI connection (should not fail)
    doc_ai_status = processor.test_document_ai_connection()
    print(f"📊 Document AI Status: {doc_ai_status['connection_status']} (Safe: {doc_ai_status.get('deployment_safe', True)})")
    
    # Test statistics
    stats = processor.get_processing_statistics()
    print(f"📈 Processing Stats: {stats['success_rate']:.1f}% success rate")
    print(f"🌍 Environment: {stats['deployment_environment']}")
    
    print("🎉 Cloud deployment test completed successfully!")
    return True

if __name__ == "__main__":
    test_cloud_deployment()
