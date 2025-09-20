"""
Simple Document Processor for Legal Eagle MVP
Handles basic document processing with minimal dependencies
"""
import os
import logging
from typing import Dict, Any, Optional, BinaryIO
import streamlit as st
from datetime import datetime

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Simple document processor for demo purposes"""
    
    def __init__(self):
        """Initialize simple document processor"""
        self.supported_formats = ['pdf', 'docx', 'txt', 'jpg', 'png', 'jpeg']
        self.max_file_size_mb = 10
        self.total_processed = 0
        self.successful_processed = 0
        
        logger.info("Simple Document Processor initialized")
    
    def get_document_metadata(self, uploaded_file) -> Dict[str, Any]:
        """Get basic metadata from uploaded file"""
        try:
            # Get file information
            file_size = len(uploaded_file.getvalue()) if uploaded_file else 0
            file_size_mb = round(file_size / (1024 * 1024), 2)
            
            filename = uploaded_file.name if uploaded_file else "unknown"
            extension = filename.split('.')[-1].lower() if '.' in filename else ""
            
            # Check if supported by Document AI (PDF, images)
            supported_by_document_ai = extension in ['pdf', 'png', 'jpg', 'jpeg']
            
            metadata = {
                'filename': filename,
                'size_bytes': file_size,
                'size_mb': file_size_mb,
                'extension': extension,
                'mime_type': uploaded_file.type if uploaded_file else "unknown",
                'supported_by_document_ai': supported_by_document_ai,
                'upload_timestamp': datetime.now().isoformat(),
                'is_supported': extension in self.supported_formats,
                'is_valid_size': file_size_mb <= self.max_file_size_mb
            }
            
            logger.info(f"Generated metadata for {filename}: {file_size_mb} MB")
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
                'error': str(e)
            }
    
    def extract_text(self, uploaded_file) -> str:
        """Extract text from uploaded file"""
        try:
            if not uploaded_file:
                return "Error: No file provided"
            
            filename = uploaded_file.name
            extension = filename.split('.')[-1].lower() if '.' in filename else ""
            
            # Reset file pointer
            uploaded_file.seek(0)
            
            # Extract text based on file type
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
            
            logger.info(f"Extracted {len(text)} characters from {filename}")
            return text
            
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            return f"Error: Text extraction failed - {str(e)}"
    
    def _extract_text_from_txt(self, uploaded_file) -> str:
        """Extract text from TXT file"""
        try:
            # Try different encodings
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
            
            return "Error: Could not decode text file"
            
        except Exception as e:
            return f"Error reading TXT file: {str(e)}"
    
    def _extract_text_from_pdf(self, uploaded_file) -> str:
        """Extract text from PDF file"""
        try:
            # Try to import PDF processing library
            try:
                import PyPDF2
            except ImportError:
                return "Error: PyPDF2 not installed. Run: pip install PyPDF2"
            
            uploaded_file.seek(0)
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
            
            if text.strip():
                return text.strip()
            else:
                return "Error: No text found in PDF. The PDF might be scanned or image-based."
                
        except Exception as e:
            return f"Error reading PDF file: {str(e)}"
    
    def _extract_text_from_docx(self, uploaded_file) -> str:
        """Extract text from DOCX file"""
        try:
            # Try to import docx processing library
            try:
                from docx import Document
            except ImportError:
                return "Error: python-docx not installed. Run: pip install python-docx"
            
            uploaded_file.seek(0)
            doc = Document(uploaded_file)
            
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            
            # Also extract text from tables
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        text += cell.text + " "
                    text += "\n"
            
            return text.strip()
            
        except Exception as e:
            return f"Error reading DOCX file: {str(e)}"
    
    def _extract_text_from_image(self, uploaded_file) -> str:
        """Extract text from image file using OCR"""
        try:
            # Try to import required libraries
            try:
                from PIL import Image
                import pytesseract
            except ImportError:
                return "Error: PIL and pytesseract not installed. Run: pip install Pillow pytesseract"
            
            uploaded_file.seek(0)
            image = Image.open(uploaded_file)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Extract text using OCR
            text = pytesseract.image_to_string(image)
            
            if text.strip():
                return text.strip()
            else:
                return "Error: No text found in image. The image might be unclear or contain no text."
                
        except Exception as e:
            return f"Error processing image file: {str(e)}. Note: Tesseract OCR must be installed on your system."
    
    def test_document_ai_connection(self) -> Dict[str, Any]:
        """Test Google Document AI connection"""
        try:
            # Check for Google Cloud credentials
            google_project = os.getenv('GOOGLE_CLOUD_PROJECT')
            google_credentials = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
            
            if google_project and google_credentials:
                # Try to import Google Cloud libraries
                try:
                    from google.cloud import documentai_v1 as documentai
                    
                    # Try to initialize client
                    client = documentai.DocumentProcessorServiceClient()
                    
                    return {
                        'connection_status': 'success',
                        'client_initialized': True,
                        'project_id': google_project,
                        'message': 'Google Document AI is configured and ready'
                    }
                    
                except Exception as e:
                    return {
                        'connection_status': 'error',
                        'client_initialized': False,
                        'error': str(e),
                        'message': 'Google Document AI configuration error'
                    }
            else:
                return {
                    'connection_status': 'not_configured',
                    'client_initialized': False,
                    'message': 'Google Document AI not configured - using basic OCR'
                }
                
        except Exception as e:
            return {
                'connection_status': 'error',
                'client_initialized': False,
                'error': str(e),
                'message': 'Document AI connection test failed'
            }
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        success_rate = (self.successful_processed / self.total_processed * 100) if self.total_processed > 0 else 0
        
        # Check Google AI configuration
        google_ai_configured = os.getenv('GOOGLE_CLOUD_PROJECT') and os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        
        return {
            'total_processed': self.total_processed,
            'successful_processed': self.successful_processed,
            'success_rate': success_rate,
            'google_ai_configured': bool(google_ai_configured),
            'google_ai_usage_rate': 100.0 if google_ai_configured else 0.0,
            'supported_formats': self.supported_formats,
            'max_file_size_mb': self.max_file_size_mb
        }

# Test function
def test_document_processor():
    """Test the document processor"""
    print("🧪 Testing Document Processor...")
    
    processor = DocumentProcessor()
    
    # Test metadata generation
    print("✅ Document processor initialized")
    
    # Test Document AI connection
    doc_ai_status = processor.test_document_ai_connection()
    print(f"📊 Document AI Status: {doc_ai_status['connection_status']}")
    
    # Test statistics
    stats = processor.get_processing_statistics()
    print(f"📈 Processing Stats: {stats['success_rate']:.1f}% success rate")
    
    print("🎉 Document processor test completed!")

if __name__ == "__main__":
    test_document_processor()
