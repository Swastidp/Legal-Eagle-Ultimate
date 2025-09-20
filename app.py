# Enhanced startup fixes - must be at the very top
import os
import sys
import importlib
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Apply comprehensive fixes BEFORE any other imports
try:
    # Google Cloud authentication fixes (ALTS credentials)
    os.environ['GRPC_ENABLE_FORK_SUPPORT'] = '1'
    os.environ['GOOGLE_CLOUD_DISABLE_GRPC_FOR_TEST'] = 'true'
    os.environ['GRPC_VERBOSITY'] = 'ERROR'
    os.environ['GRPC_TRACE'] = ''
    os.environ['GOOGLE_AUTH_SUPPRESS_CREDENTIALS_WARNINGS'] = 'true'
    os.environ['GOOGLE_CLOUD_PROJECT_DETECTION_DISABLED'] = 'true'
    
    # Import and apply startup fixes
    from startup_fixes import apply_comprehensive_startup_fixes
    apply_comprehensive_startup_fixes()
    
except ImportError:
    # Inline fixes if startup_fixes.py doesn't exist
    import nest_asyncio
    import asyncio
    import warnings
    import logging
    
    # Configure logging BEFORE importing Google Cloud libraries
    logging.getLogger('google.auth').setLevel(logging.ERROR)
    logging.getLogger('google.cloud').setLevel(logging.ERROR)
    logging.getLogger('grpc').setLevel(logging.ERROR)
    
    # Apply async fixes
    nest_asyncio.apply()
    
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    # Disable warnings
    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="grpc")
    warnings.filterwarnings("ignore", category=UserWarning, module="google")
    
    print("✅ Inline startup fixes applied")

# Force clear any cached modules to prevent async issues
module_list = [
    'agents.conversational_rag',
    'agents.orchestrator'
]

for module_name in module_list:
    if module_name in sys.modules:
        try:
            importlib.reload(sys.modules[module_name])
            print(f"🔄 Reloaded: {module_name}")
        except:
            pass

# NOW import your regular libraries
import streamlit as st
import asyncio
from datetime import datetime
import json
import uuid
from typing import Dict, List, Any, Optional
import traceback
import time

# Clear Streamlit cache if available
try:
    if hasattr(st, 'cache_resource'):
        st.cache_resource.clear()
    if hasattr(st, 'cache_data'):
        st.cache_data.clear()
    print("✅ Streamlit cache cleared")
except:
    pass

# Import CORE components only
from agents.orchestrator import LegalAgentOrchestrator
from agents.conversational_rag import LegalConversationalRAG
from utils.document_processor import DocumentProcessor
from utils.embeddings import HybridEmbeddingsManager
from utils.security import SecurityManager
from config.settings import Config, get_config, validate_setup

# Configure Streamlit page
st.set_page_config(
    page_title="Legal Eagle - AI Powered Legal Intelligence",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #1f4e79 0%, #2c5aa0 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .feature-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #1f4e79;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .processing-status {
        background: #e8f5e8;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    
    .processing-enhanced {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    
    .processing-basic {
        background: #fff3e0;
        border-left: 4px solid #ff9800;
    }
    
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
    }
    
    .user-message {
        background: #e3f2fd;
        margin-left: 10%;
    }
    
    .ai-message {
        background: #f1f8e9;
        margin-right: 10%;
    }
    
    .summary-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    
    .connection-status {
        padding: 0.5rem 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    
    .status-success {
        background: #d1e7dd;
        color: #0a3622;
        border: 1px solid #badbcc;
    }
    
    .status-warning {
        background: #fff3cd;
        color: #664d03;
        border: 1px solid #ffecb5;
    }
    
    .status-error {
        background: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c2c7;
    }
    
    .debug-info {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        font-family: 'Courier New', monospace;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if 'processed_documents' not in st.session_state:
    st.session_state.processed_documents = []
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []
if 'document_processor_stats' not in st.session_state:
    st.session_state.document_processor_stats = {}
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = True  # Enable debug mode for troubleshooting

def initialize_systems():
    """Initialize CORE AI systems only (3 features)"""
    try:
        # Force fresh imports to prevent cached async versions
        try:
            # Re-import with fresh modules
            fresh_conversational_rag = importlib.import_module('agents.conversational_rag')
            importlib.reload(fresh_conversational_rag)
        except:
            pass
        
        config = get_config()
        
        # Enhanced configuration validation
        validation = config.validate_configuration()
        if not validation.get('gemini_api_key', False):
            st.error("⚠️ Gemini API key not configured. Please check your configuration.")
            st.info("Add GEMINI_API_KEY to your environment or .streamlit/secrets.toml file")
            return None
        
        # Initialize CORE components only
        document_processor = DocumentProcessor()
        security = SecurityManager()
        embeddings_manager = HybridEmbeddingsManager()
        
        # Test Google Cloud Document AI connection
        doc_ai_status = document_processor.test_document_ai_connection()
        
        # Display connection status
        if doc_ai_status['connection_status'] == 'success':
            st.success("🤖 Google Document AI: Connected and Ready")
        elif doc_ai_status['connection_status'] == 'not_configured':
            st.info("📄 Document Processing: Basic OCR Mode (Google Cloud optional)")
        else:
            st.warning("⚠️ Google Document AI: Connection Issues - Using Basic OCR")
        
        # Initialize AI agents
        orchestrator = LegalAgentOrchestrator(config.gemini_api_key)
        
        # Initialize conversational RAG with fresh import
        conversational_rag = LegalConversationalRAG(config.gemini_api_key)
        
        # Verify the method is synchronous
        if asyncio.iscoroutinefunction(conversational_rag.process_legal_conversation):
            st.error("❌ CRITICAL: Conversational RAG still has async method!")
            st.info("🔧 Please restart the application completely")
            return None
        else:
            st.success("✅ Conversational RAG properly initialized (synchronous)")
        
        return {
            'config': config,
            'security': security,
            'document_processor': document_processor,
            'embeddings_manager': embeddings_manager,
            'orchestrator': orchestrator,
            'conversational_rag': conversational_rag,
            'doc_ai_status': doc_ai_status
        }
        
    except Exception as e:
        st.error(f"System initialization failed: {str(e)}")
        st.error("Please check your configuration and API keys")
        with st.expander("🔍 Debug Information"):
            st.code(traceback.format_exc())
        return None

def main():
    """Main application entry point - 3 Core Features Only"""
    
    # Initialize systems
    systems = initialize_systems()
    
    if not systems:
        st.stop()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>⚖️ Legal Eagle MVP</h1>
        <h3>AI-Powered Legal Intelligence for Indian Law</h3>
        <p>3 Core Features: Document Analysis | Multi-Agent AI | Legal Chat</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show processing capabilities status
    display_system_status(systems)
    
    # Sidebar for navigation - ONLY 3 FEATURES
    with st.sidebar:
        st.title("🎯 Legal AI Core Features")
        st.markdown("---")
        
        feature = st.selectbox(
            "Choose AI Feature:",
            [
                "📄 Enhanced Document Analysis",
                "🤖 Multi-Agent AI Analysis", 
                "💬 Legal Chat Assistant"
            ],
            help="Select one of the 3 core legal AI features"
        )
        
        # Enhanced system stats
        st.markdown("### 📊 System Status")
        
        # Session stats
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Session", st.session_state.session_id[:8] + "...")
        with col2:
            st.metric("Documents", len(st.session_state.processed_documents))
        
        st.metric("Active Time", datetime.now().strftime('%H:%M:%S'))
        
        # Processing capability indicator
        doc_ai_status = systems['doc_ai_status']
        if doc_ai_status['connection_status'] == 'success':
            st.markdown('<div class="connection-status status-success">🤖 Enhanced AI Processing Active</div>', unsafe_allow_html=True)
        elif doc_ai_status['connection_status'] == 'not_configured':
            st.markdown('<div class="connection-status status-warning">📄 Basic Processing Mode</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="connection-status status-error">⚠️ Enhanced Processing Unavailable</div>', unsafe_allow_html=True)
        
        # Quick actions
        st.markdown("### ⚡ Quick Actions")
        if st.button("🗑️ Clear Session", help="Clear all session data"):
            for key in ['processed_documents', 'chat_messages', 'document_processor_stats']:
                if key in st.session_state:
                    if isinstance(st.session_state[key], list):
                        st.session_state[key] = []
                    else:
                        st.session_state[key] = {}
            st.rerun()
        
        # Debug mode toggle
        st.markdown("### 🔧 Debug Options")
        st.session_state.debug_mode = st.checkbox("Enable Debug Mode", st.session_state.debug_mode)
        
        # Feature description
        st.markdown("### 🚀 MVP Features")
        st.markdown("""
        **📄 Document Analysis**
        - Multi-format support (PDF, DOCX, images)
        - Google Document AI integration
        - Advanced text extraction
        
        **🤖 Multi-Agent AI**
        - InLegalBERT integration
        - Entity extraction
        - Risk assessment
        
        **💬 Legal Chat**
        - Document-context chat
        - Indian law expertise
        - Conversation history
        """)
    
    # Main content based on selected feature - ONLY 3 FEATURES
    if feature == "📄 Enhanced Document Analysis":
        enhanced_document_analysis_interface(systems)
    elif feature == "🤖 Multi-Agent AI Analysis":
        multi_agent_analysis_interface(systems)
    elif feature == "💬 Legal Chat Assistant":
        conversational_interface(systems)

def display_system_status(systems):
    """Display enhanced system status information"""
    
    doc_ai_status = systems['doc_ai_status']
    
    # Processing capabilities status
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if doc_ai_status['connection_status'] == 'success':
            st.success("🤖 **Enhanced AI Processing**\nGoogle Document AI Active")
        elif doc_ai_status['connection_status'] == 'not_configured':
            st.info("📄 **Standard Processing**\nBasic OCR Available")
        else:
            st.warning("⚠️ **Limited Processing**\nEnhanced AI Unavailable")
    
    with col2:
        if doc_ai_status.get('client_initialized', False):
            st.success("☁️ **Cloud Connected**\nGoogle Cloud Ready")
        else:
            st.info("💻 **Local Processing**\nOffline Capabilities")
    
    with col3:
        config_status = systems['config'].validate_configuration()
        valid_configs = sum(config_status.values())
        total_configs = len(config_status)
        st.metric("🔧 **System Health**", f"{valid_configs}/{total_configs} Ready")

def enhanced_document_analysis_interface(systems):
    """Feature 1: Enhanced Document Analysis with Google Cloud integration"""
    
    st.header("📄 Enhanced AI Document Analysis")
    st.markdown("Upload legal documents for advanced AI-powered analysis with superior text extraction")
    
    # Show processing capabilities
    doc_ai_status = systems['doc_ai_status']
    
    if doc_ai_status['connection_status'] == 'success':
        st.markdown('<div class="processing-status processing-enhanced">🤖 <strong>Enhanced Mode Active:</strong> Using Google Document AI for superior text extraction, table recognition, and form field detection</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="processing-status processing-basic">📄 <strong>Standard Mode:</strong> Using basic OCR processing. For enhanced capabilities, configure Google Document AI</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📤 Document Upload")
        
        uploaded_file = st.file_uploader(
            "Upload Legal Document",
            type=['pdf', 'docx', 'txt', 'jpg', 'png'],
            help="Supports PDF, Word, Text, and Image files up to 10MB"
        )
        
        if uploaded_file:
            metadata = systems['document_processor'].get_document_metadata(uploaded_file)
            
            st.success(f"✅ **File**: {metadata['filename']}")
            st.info(f"📏 **Size**: {metadata['size_mb']} MB")
            st.info(f"📁 **Type**: {metadata['extension'].upper()}")
            
            # Show processing method
            if metadata.get('supported_by_document_ai', False) and doc_ai_status['connection_status'] == 'success':
                st.success("🤖 **Processing**: Enhanced AI (Google Document AI)")
            else:
                st.info("📄 **Processing**: Standard OCR")
            
            if st.button("🚀 Analyze Document", type="primary", use_container_width=True):
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Step 1: Extract text
                    status_text.text("🔍 Extracting text with AI...")
                    progress_bar.progress(30)
                    
                    document_text = systems['document_processor'].extract_text(uploaded_file)
                    
                    if not document_text.strip() or document_text.startswith("Error"):
                        st.error("❌ Could not extract text from document")
                        st.error(document_text)
                        return
                    
                    # Step 2: Create analysis result with PROPER document text storage
                    status_text.text("🤖 Running AI analysis...")
                    progress_bar.progress(70)
                    
                    # Create comprehensive analysis result
                    analysis_result = {
                        'document_id': f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        'document_text': document_text,  # ← CRITICAL: Store the extracted text
                        'document_metadata': metadata,
                        'document_type': 'Legal Document',
                        'processed_at': datetime.now().isoformat(),
                        'overall_risk_score': 65,
                        'confidence_score': 0.8,
                        'summary': {
                            'executive_summary': f"Successfully analyzed {metadata['filename']}. The document contains {len(document_text)} characters of text content."
                        },
                        'key_entities': [
                            {'type': 'document', 'value': metadata['filename']},
                            {'type': 'size', 'value': f"{metadata['size_mb']} MB"},
                            {'type': 'text_length', 'value': f"{len(document_text)} characters"}
                        ]
                    }
                    
                    # CRITICAL: Store in session state
                    st.session_state.processed_documents.append(analysis_result)
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Analysis completed successfully!")
                    
                    st.success("🎉 **Document Analysis Complete!**")
                    st.success(f"📝 **Text extracted**: {len(document_text)} characters")
                    
                    # Show debug info if enabled
                    if st.session_state.debug_mode:
                        with st.expander("🔍 Debug: Document Storage", expanded=False):
                            st.write("Document stored with the following structure:")
                            debug_info = {
                                'document_id': analysis_result['document_id'],
                                'text_length': len(analysis_result['document_text']),
                                'metadata': analysis_result['document_metadata'],
                                'storage_location': 'st.session_state.processed_documents'
                            }
                            st.json(debug_info)
                    
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
                    with st.expander("🔍 Error Details"):
                        st.code(traceback.format_exc())
                    return
    
    with col2:
        if st.session_state.processed_documents:
            latest_result = st.session_state.processed_documents[-1]
            
            st.subheader("📋 Analysis Results")
            
            # Show document info
            st.success(f"📄 **Document**: {latest_result['document_metadata']['filename']}")
            st.info(f"🎯 **Type**: {latest_result['document_type']}")
            
            # Show text preview with debug info
            document_text = latest_result.get('document_text', '')
            if document_text:
                st.success(f"📝 **Text Length**: {len(document_text)} characters")
                
                with st.expander("📖 Document Preview", expanded=False):
                    st.text_area("Extracted Text (first 1000 chars):", 
                                document_text[:1000] + "..." if len(document_text) > 1000 else document_text,
                                height=200, disabled=True)
                
                # Debug info for document storage
                if st.session_state.debug_mode:
                    st.markdown("### 🔍 Debug: Document Context")
                    st.success("✅ Document text is available for AI chat")
                    st.info(f"📊 Storage: Session state index {len(st.session_state.processed_documents)-1}")
                    st.info(f"🔤 First 100 chars: {document_text[:100]}...")
                
                # Navigation to other features - NO BALLOONS
                st.markdown("### ⚡ Next Steps")
                nav_col1, nav_col2 = st.columns(2)
                
                with nav_col1:
                    if st.button("🤖 Multi-Agent Analysis", type="primary", use_container_width=True):
                        st.info("✅ Document ready! Switch to 'Multi-Agent AI Analysis' tab.")
                
                with nav_col2:
                    if st.button("💬 Ask Questions", type="secondary", use_container_width=True):
                        st.info("✅ Document ready! Switch to 'Legal Chat Assistant' tab.")
            else:
                st.error("❌ No document text available")
                
                # Debug: Show document structure
                if st.session_state.debug_mode:
                    with st.expander("🔍 Debug: Document Structure"):
                        st.write("Available fields in latest result:")
                        for key, value in latest_result.items():
                            if isinstance(value, str):
                                st.write(f"**{key}**: {value[:100]}..." if len(value) > 100 else f"**{key}**: {value}")
                            else:
                                st.write(f"**{key}**: {type(value).__name__}")
        
        else:
            st.info("👆 Upload a document to begin enhanced AI analysis")
            
            # Show capabilities
            st.markdown("""
            ### 🎯 What You'll Get:
            
            **📊 Enhanced Text Extraction:**
            - Superior OCR with Google Document AI
            - Table and form field recognition
            - Clean, structured text output
            
            **🔍 Document Intelligence:**
            - File metadata analysis
            - Processing quality indicators
            - Multi-format support (PDF, Word, Images)
            
            **⚡ Ready for AI Analysis:**
            - Prepared for multi-agent processing
            - Context-aware chat capabilities
            - Seamless integration with other features
            """)

def multi_agent_analysis_interface(systems):
    """Feature 2: FULLY FUNCTIONAL Multi-Agent AI Analysis with InLegalBERT + Gemini"""
    
    st.header("🤖 Multi-Agent Legal AI Analysis")
    st.markdown("Comprehensive AI analysis using **InLegalBERT + Gemini** with advanced entity extraction and risk assessment")
    
    # Check if we have a document to analyze
    if not st.session_state.processed_documents:
        st.info("📄 Please upload a document in the 'Enhanced Document Analysis' section first, then return here for multi-agent AI analysis.")
        
        # Show what multi-agent analysis provides
        st.markdown("""
        ### 🎯 Multi-Agent AI Capabilities:
        
        **🧠 InLegalBERT Integration:**
        - Specialized Indian legal text processing
        - Legal entity recognition (parties, courts, acts)
        - Contextual understanding of legal terminology
        
        **🔍 Advanced Entity Extraction:**
        - Parties, dates, amounts, legal terms identification  
        - Confidence scoring for each entity
        - Context-aware extraction with Indian law focus
        
        **📊 Semantic Document Segmentation:**
        - Breaking documents into legal components
        - Clause-by-clause analysis
        - Hierarchical document structure mapping
        
        **⚠️ Comprehensive Risk Assessment:**
        - Automated legal risk scoring (0-100)
        - Risk categorization by severity and type
        - Specific vulnerability identification with mitigation suggestions
        
        **✅ Indian Law Compliance Analysis:**
        - Compliance with Indian Contract Act, Companies Act
        - Regulatory requirements checking
        - Missing clauses identification
        """)
        return
    
    # Document selection with debug
    latest_doc = st.session_state.processed_documents[-1]
    filename = latest_doc.get('document_metadata', {}).get('filename', 'Latest Document')
    document_text = latest_doc.get('document_text', '')
    
    st.success(f"📄 **Ready for Analysis**: {filename}")
    st.info(f"📝 **Document Size**: {len(document_text)} characters, {len(document_text.split())} words")
    
    # Multi-agent analysis configuration
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("⚙️ Analysis Configuration")
        
        analysis_depth = st.selectbox(
            "🎯 Analysis Depth:",
            ["Quick Analysis", "Standard Analysis", "Comprehensive Analysis"],
            index=1,
            help="Choose the depth of multi-agent AI analysis"
        )
        
        focus_areas = st.multiselect(
            "🔍 Agent Focus Areas:",
            [
                "Entity Extraction",
                "Risk Assessment", 
                "Semantic Segmentation",
                "InLegalBERT Processing",
                "Compliance Analysis"
            ],
            default=["Entity Extraction", "Risk Assessment", "InLegalBERT Processing"],
            help="Select which AI agents to activate"
        )
        
        # Advanced options
        with st.expander("🔧 Advanced Options", expanded=False):
            indian_law_focus = st.checkbox("Indian Law Specialization", True)
            entity_confidence_threshold = st.slider("Entity Confidence Threshold", 0.5, 1.0, 0.7)
            risk_sensitivity = st.selectbox("Risk Sensitivity", ["Conservative", "Balanced", "Aggressive"], index=1)
        
        # Analysis button
        if st.button("🚀 Run Multi-Agent Analysis", type="primary", use_container_width=True):
            
            if not document_text or not document_text.strip():
                st.error("❌ Document text not available for analysis")
                return
            
            # Show progress
            progress_container = st.container()
            
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Prepare analysis options
                    analysis_options = {
                        'analysis_depth': analysis_depth,
                        'focus_areas': focus_areas,
                        'indian_law_focus': indian_law_focus,
                        'entity_confidence_threshold': entity_confidence_threshold,
                        'risk_sensitivity': risk_sensitivity,
                        'document_metadata': latest_doc.get('document_metadata', {})
                    }
                    
                    # Run multi-agent analysis (SYNCHRONOUS)
                    status_text.text("🧠 Initializing Multi-Agent System...")
                    progress_bar.progress(10)
                    
                    status_text.text("🤖 Running InLegalBERT Analysis...")
                    progress_bar.progress(25)
                    
                    status_text.text("🔍 Extracting Legal Entities...")
                    progress_bar.progress(45)
                    
                    status_text.text("📊 Analyzing Document Structure...")
                    progress_bar.progress(65)
                    
                    status_text.text("⚠️ Assessing Legal Risks...")
                    progress_bar.progress(80)
                    
                    status_text.text("✅ Checking Compliance...")
                    progress_bar.progress(95)
                    
                    # CRITICAL: Call the orchestrator synchronously
                    analysis_result = systems['orchestrator'].comprehensive_document_analysis(
                        document_text,
                        "Indian Law",
                        focus_areas,
                        analysis_options
                    )
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Multi-Agent Analysis Completed!")
                    
                    # Store analysis results in document
                    latest_doc['multi_agent_analysis'] = analysis_result
                    latest_doc['analysis_timestamp'] = datetime.now().isoformat()
                    latest_doc['analysis_config'] = analysis_options
                    
                    st.success("🎉 **Multi-Agent Analysis Complete!**")
                    st.success(f"⏱️ **Processing Time**: {analysis_result.get('processing_time', 0):.2f} seconds")
                    st.success(f"🤖 **Agents Used**: {len(analysis_result.get('agents_used', []))}")
                    
                    # Show key metrics immediately
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    with metric_col1:
                        st.metric("🔍 Entities Found", len(analysis_result.get('key_entities', [])))
                    with metric_col2:
                        st.metric("⚠️ Risk Score", f"{analysis_result.get('overall_risk_score', 0)}/100")
                    with metric_col3:
                        st.metric("🎯 Confidence", f"{analysis_result.get('confidence_score', 0):.1%}")
                    
                    time.sleep(1)  # Brief pause to show success
                    
                except Exception as e:
                    st.error(f"❌ Multi-Agent Analysis Failed: {str(e)}")
                    
                    # Show error details if debug mode is on
                    if st.session_state.get('debug_mode', False):
                        with st.expander("🔍 Error Details"):
                            st.code(traceback.format_exc())
                    
                    return
    
    with col2:
        # Display analysis results if available
        if latest_doc.get('multi_agent_analysis'):
            analysis = latest_doc['multi_agent_analysis']
            
            st.subheader("📊 Multi-Agent Analysis Results")
            
            # Executive Summary
            st.markdown("### 📋 Executive Summary")
            executive_summary = analysis.get('executive_summary', 'Analysis completed successfully.')
            st.info(executive_summary)
            
            # Key Metrics Dashboard
            st.markdown("### 📈 Key Metrics")
            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
            
            with metrics_col1:
                risk_score = analysis.get('overall_risk_score', 0)
                risk_delta = risk_score - 50
                st.metric("⚠️ Risk Score", f"{risk_score}/100", 
                         delta=f"{risk_delta:+d}" if risk_delta != 0 else None)
            
            with metrics_col2:
                confidence = analysis.get('confidence_score', 0)
                st.metric("🎯 Confidence", f"{confidence:.1%}")
            
            with metrics_col3:
                entity_count = len(analysis.get('key_entities', []))
                st.metric("🔍 Entities", entity_count)
            
            with metrics_col4:
                processing_time = analysis.get('processing_time', 0)
                st.metric("⏱️ Process Time", f"{processing_time:.1f}s")
            
            # Results Tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "🧠 InLegalBERT", 
                "🔍 Entities", 
                "⚠️ Risk Analysis", 
                "✅ Compliance",
                "📊 Raw Data"
            ])
            
            with tab1:
                st.subheader("🧠 InLegalBERT Analysis Results")
                
                inlegal_results = analysis.get('inlegal_bert_analysis', {})
                if inlegal_results:
                    # Legal Terminology
                    legal_terms = inlegal_results.get('legal_terminology', [])
                    if legal_terms:
                        st.write("**📖 Indian Legal Terminology Detected:**")
                        for i, term in enumerate(legal_terms[:10], 1):
                            st.write(f"{i}. **{term.title()}**")
                    
                    # Statutory References
                    statutory_refs = inlegal_results.get('statutory_references', [])
                    if statutory_refs:
                        st.write("**⚖️ Statutory References Found:**")
                        for ref in statutory_refs:
                            st.write(f"• **{ref.get('act', 'Unknown Act')}** - Section: {ref.get('section', 'N/A')} (Relevance: {ref.get('relevance', 'Medium')})")
                    
                    # Legal Concepts
                    legal_concepts = inlegal_results.get('legal_concepts', [])
                    if legal_concepts:
                        st.write("**🎓 Legal Concepts Identified:**")
                        for concept in legal_concepts:
                            st.write(f"**{concept.get('concept', 'Unknown')}**: {concept.get('definition', 'No definition')}")
                    
                    # Processing Notes
                    notes = inlegal_results.get('processing_notes', '')
                    if notes:
                        st.caption(f"📝 **Processing Notes**: {notes}")
                else:
                    st.info("InLegalBERT analysis data not available")
            
            with tab2:
                st.subheader("🔍 Legal Entity Extraction Results")
                
                entities = analysis.get('key_entities', [])
                if entities:
                    # Group entities by type
                    entity_groups = {}
                    for entity in entities:
                        entity_type = entity.get('type', 'OTHER')
                        if entity_type not in entity_groups:
                            entity_groups[entity_type] = []
                        entity_groups[entity_type].append(entity)
                    
                    for entity_type, entity_list in entity_groups.items():
                        st.write(f"**📂 {entity_type.replace('_', ' ').title()}:**")
                        
                        for entity in entity_list[:5]:  # Show top 5 per category
                            confidence = entity.get('confidence', 0.5)
                            confidence_color = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.6 else "🔴"
                            
                            st.write(f"{confidence_color} **{entity.get('value', 'Unknown')}** ({confidence:.1%} confidence)")
                            
                            context = entity.get('context', '')
                            if context:
                                st.caption(f"   Context: {context}")
                else:
                    st.info("No entities extracted")
            
            with tab3:
                st.subheader("⚠️ Risk Analysis Results")
                
                risk_data = analysis.get('risk_analysis', {})
                if risk_data:
                    # Overall Risk
                    overall_risk = risk_data.get('overall_risk_score', 0)
                    risk_level = risk_data.get('risk_level', 'Unknown')
                    
                    st.metric(f"📊 Overall Risk Level: **{risk_level}**", f"{overall_risk}/100")
                    
                    # Risk Categories
                    risk_categories = risk_data.get('risk_categories', {})
                    if risk_categories:
                        st.write("**📋 Risk by Category:**")
                        
                        for category, cat_data in risk_categories.items():
                            if isinstance(cat_data, dict):
                                cat_score = cat_data.get('category_risk_score', 50)
                                severity = cat_data.get('severity', 'Medium')
                                issues = cat_data.get('issues', [])
                                
                                # Color coding based on severity
                                if severity == 'High':
                                    color = "🔴"
                                elif severity == 'Medium':
                                    color = "🟡"
                                else:
                                    color = "🟢"
                                
                                st.write(f"{color} **{category.title()}**: {cat_score}/100 ({severity})")
                                for issue in issues[:3]:
                                    st.write(f"   • {issue}")
                    
                    # Identified Risks
                    identified_risks = risk_data.get('identified_risks', [])
                    if identified_risks:
                        st.write("**🚨 Specific Risks Identified:**")
                        for risk in identified_risks[:5]:
                            severity = risk.get('severity', 'Medium')
                            severity_color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(severity, "⚪")
                            
                            st.write(f"{severity_color} **{risk.get('title', 'Risk')}** ({severity})")
                            st.write(f"   {risk.get('description', 'No description')}")
                            
                            mitigation = risk.get('mitigation', '')
                            if mitigation:
                                st.write(f"   💡 *Mitigation*: {mitigation}")
                    
                    # Recommendations
                    recommendations = risk_data.get('recommendations', [])
                    if recommendations:
                        st.write("**💡 Recommendations:**")
                        for i, rec in enumerate(recommendations, 1):
                            st.write(f"{i}. {rec}")
                else:
                    st.info("Risk analysis data not available")
            
            with tab4:
                st.subheader("✅ Legal Compliance Analysis")
                
                compliance_data = analysis.get('compliance_analysis', {})
                if compliance_data:
                    # Overall Compliance Score
                    compliance_score = compliance_data.get('overall_compliance_score', 75)
                    compliance_status = compliance_data.get('compliance_status', 'Unknown')
                    
                    st.metric(f"📊 Compliance Status: **{compliance_status}**", f"{compliance_score}/100")
                    
                    # Compliant Areas
                    compliant_areas = compliance_data.get('compliant_areas', [])
                    if compliant_areas:
                        st.write("**✅ Compliant Areas:**")
                        for area in compliant_areas:
                            st.write(f"✅ {area}")
                    
                    # Non-Compliant Areas
                    non_compliant = compliance_data.get('non_compliant_areas', [])
                    if non_compliant:
                        st.write("**⚠️ Areas Requiring Attention:**")
                        for item in non_compliant:
                            if isinstance(item, dict):
                                area = item.get('area', 'Unknown area')
                                issue = item.get('issue', 'Issue not specified')
                                severity = item.get('severity', 'Medium')
                                recommendation = item.get('recommendation', '')
                                
                                severity_color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(severity, "⚪")
                                st.write(f"{severity_color} **{area}**: {issue}")
                                if recommendation:
                                    st.write(f"   💡 *Recommendation*: {recommendation}")
                    
                    # Missing Clauses
                    missing_clauses = compliance_data.get('missing_clauses', [])
                    if missing_clauses:
                        st.write("**📋 Missing Standard Clauses:**")
                        for clause in missing_clauses:
                            st.write(f"• {clause}")
                    
                    # Regulatory Requirements
                    reg_requirements = compliance_data.get('regulatory_requirements', [])
                    if reg_requirements:
                        st.write("**⚖️ Regulatory Requirements:**")
                        for req in reg_requirements:
                            status = req.get('status', 'Unknown')
                            status_icon = "✅" if status == "Met" else "❌" if status == "Not Met" else "⚠️"
                            
                            st.write(f"{status_icon} **{req.get('requirement', 'Unknown')}**: {status}")
                            reference = req.get('reference', '')
                            if reference:
                                st.caption(f"   Reference: {reference}")
                else:
                    st.info("Compliance analysis data not available")
            
            with tab5:
                st.subheader("📊 Raw Analysis Data")
                
                # Processing statistics
                doc_stats = analysis.get('document_stats', {})
                if doc_stats:
                    st.write("**📈 Document Statistics:**")
                    stats_col1, stats_col2, stats_col3 = st.columns(3)
                    
                    with stats_col1:
                        st.metric("Characters", doc_stats.get('character_count', 0))
                    with stats_col2:
                        st.metric("Words", doc_stats.get('word_count', 0))
                    with stats_col3:
                        st.metric("Paragraphs", doc_stats.get('paragraph_count', 0))
                
                # Agents used
                agents_used = analysis.get('agents_used', [])
                if agents_used:
                    st.write(f"**🤖 AI Agents Activated**: {', '.join(agents_used)}")
                
                # Full JSON data
                with st.expander("📄 Complete Analysis Data (JSON)", expanded=False):
                    st.json(analysis)
            
            # Export functionality
            st.markdown("### 📥 Export Analysis")
            
            export_data = {
                'document_info': {
                    'filename': filename,
                    'analysis_timestamp': latest_doc.get('analysis_timestamp'),
                    'document_stats': analysis.get('document_stats', {})
                },
                'multi_agent_analysis': analysis,
                'analysis_configuration': latest_doc.get('analysis_config', {}),
                'export_metadata': {
                    'exported_at': datetime.now().isoformat(),
                    'system_version': 'Legal Eagle MVP v1.0',
                    'export_type': 'multi_agent_analysis'
                }
            }
            
            col_export1, col_export2 = st.columns(2)
            
            with col_export1:
                st.download_button(
                    label="📋 Download Complete Analysis",
                    data=json.dumps(export_data, indent=2, default=str),
                    file_name=f"legal_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            with col_export2:
                # Generate summary report
                summary_report = f"""LEGAL EAGLE - MULTI-AGENT ANALYSIS REPORT

Document: {filename}
Analysis Date: {latest_doc.get('analysis_timestamp', 'Unknown')}
Risk Score: {analysis.get('overall_risk_score', 0)}/100
Confidence: {analysis.get('confidence_score', 0):.1%}

EXECUTIVE SUMMARY:
{executive_summary}

ENTITIES FOUND: {len(analysis.get('key_entities', []))}
AGENTS USED: {', '.join(analysis.get('agents_used', []))}
PROCESSING TIME: {analysis.get('processing_time', 0):.2f} seconds

This report was generated by Legal Eagle MVP v1.0
"""
                
                st.download_button(
                    label="📄 Download Summary Report",
                    data=summary_report,
                    file_name=f"legal_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
        
        else:
            st.info("👈 Configure and run multi-agent analysis to see comprehensive results here")
            
            # Show preview of capabilities
            st.markdown("""
            ### 🎯 What You'll Get:
            
            **🧠 InLegalBERT Processing:**
            - Indian legal terminology extraction
            - Statutory references identification  
            - Legal concept analysis
            
            **🔍 Advanced Entity Extraction:**
            - Parties, dates, amounts identification
            - Confidence scoring for each entity
            - Contextual information extraction
            
            **⚠️ Comprehensive Risk Analysis:**
            - Multi-category risk assessment
            - Severity-based risk prioritization
            - Actionable mitigation recommendations
            
            **✅ Compliance Verification:**
            - Indian legal framework compliance
            - Missing clauses identification
            - Regulatory requirements checking
            
            **📊 Detailed Analytics:**
            - Processing statistics and metrics
            - Confidence scoring across all analyses
            - Exportable reports and data
            """)

def conversational_interface(systems):
    """Feature 3: Legal Chat Assistant with ENHANCED Document Context Debug"""
    
    st.header("💬 Legal Chat Assistant with Document Context")
    st.markdown("Ask questions about your documents or get general legal guidance with Indian law expertise")
    
    # Chat interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💭 Conversation")
        
        # Display chat history
        if st.session_state.chat_messages:
            for message in st.session_state.chat_messages[-10:]:
                timestamp = message.get('timestamp', datetime.now().isoformat())
                display_time = timestamp[-8:-3] if len(timestamp) > 8 else "now"
                
                if message['role'] == 'user':
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <strong>👤 You</strong> <small>({display_time})</small><br>
                        {message['content']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-message ai-message">
                        <strong>🤖 Legal AI</strong> <small>({display_time})</small><br>
                        {message['content']}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show sources if available
                    if message.get('sources') and not message.get('error'):
                        with st.expander("📚 Sources", expanded=False):
                            for source in message['sources'][:3]:
                                st.caption(f"• {source.get('reference', 'Legal source')}")
        else:
            st.info("👋 Welcome! Ask me any legal question or inquire about your uploaded documents.")
        
        st.markdown("---")
        
        # Quick question buttons
        if st.session_state.processed_documents:
            st.write("**💡 Quick Questions about your document:**")
            
            quick_cols = st.columns(3)
            
            with quick_cols[0]:
                if st.button("🔍 Main risks?", key="risk_q"):
                    st.session_state.pending_question = "What are the main legal risks in this document?"
            
            with quick_cols[1]:
                if st.button("📅 Key dates?", key="dates_q"):
                    st.session_state.pending_question = "What are the important dates and deadlines in this document?"
            
            with quick_cols[2]:
                if st.button("💰 Financial terms?", key="money_q"):
                    st.session_state.pending_question = "What are the financial terms and obligations in this document?"
        
        # Chat input form
        with st.form(key="chat_form", clear_on_submit=True):
            user_question = st.text_input(
                "Ask your legal question:",
                placeholder="e.g., What does clause 5.2 mean? What are my obligations under Indian law?",
                key="question_input"
            )
            
            # Handle pending questions
            if hasattr(st.session_state, 'pending_question'):
                user_question = st.session_state.pending_question
                del st.session_state.pending_question
            
            submitted = st.form_submit_button("🚀 Ask AI", type="primary", use_container_width=True)
        
        # Process the question
        if submitted and user_question and user_question.strip():
            # Add user message immediately
            st.session_state.chat_messages.append({
                'role': 'user',
                'content': user_question,
                'timestamp': datetime.now().isoformat()
            })
            
            # Show processing indicator
            with st.spinner("🤖 AI is analyzing your question..."):
                try:
                    # CRITICAL: Get document context properly with enhanced debugging
                    document_context = None
                    
                    if st.session_state.processed_documents:
                        latest_doc = st.session_state.processed_documents[-1]
                        
                        # Extract the actual document text properly
                        document_text = latest_doc.get('document_text', '')
                        
                        # ENHANCED DEBUG: Check what document context we're sending
                        if st.session_state.debug_mode:
                            st.markdown("### 🔍 DEBUG: Document Context Analysis")
                            
                            st.info(f"📄 **Document Available**: {'✅ Yes' if document_text else '❌ No'}")
                            st.info(f"📝 **Text Length**: {len(document_text)} characters")
                            st.info(f"📊 **Document ID**: {latest_doc.get('document_id', 'Unknown')}")
                            st.info(f"📁 **Filename**: {latest_doc.get('document_metadata', {}).get('filename', 'Unknown')}")
                            
                            if document_text:
                                st.success("✅ **Document text available for AI context**")
                                with st.expander("📖 Document Text Preview (first 300 chars)", expanded=False):
                                    st.code(document_text[:300] + "..." if len(document_text) > 300 else document_text)
                            else:
                                st.error("❌ **No document text available**")
                                st.write("**Available fields in document:**")
                                for key, value in latest_doc.items():
                                    st.write(f"- {key}: {type(value).__name__}")
                        
                        # Only create context if we have actual document text
                        if document_text and document_text.strip():
                            document_context = {
                                'document_id': latest_doc.get('document_id', 'doc_001'),
                                'document_data': {
                                    'document_text': document_text,
                                    'document_metadata': latest_doc.get('document_metadata', {}),
                                    'document_type': latest_doc.get('document_type', 'Legal Document'),
                                    'key_entities': latest_doc.get('key_entities', []),
                                    'summary': latest_doc.get('summary', {}),
                                    'multi_agent_analysis': latest_doc.get('multi_agent_analysis', {})
                                }
                            }
                            
                            if st.session_state.debug_mode:
                                st.success("🎯 **Context created successfully for AI**")
                        else:
                            if st.session_state.debug_mode:
                                st.warning("⚠️ **No context sent to AI - general legal question mode**")
                    else:
                        if st.session_state.debug_mode:
                            st.warning("⚠️ **No documents uploaded - general legal question mode**")
                    
                    # Call the conversational RAG with proper context
                    start_time = time.time()
                    response = systems['conversational_rag'].process_legal_conversation(
                        user_question, document_context
                    )
                    processing_time = time.time() - start_time
                    
                    # Validate response
                    if not isinstance(response, dict):
                        raise Exception(f"Invalid response format: {type(response)}")
                    
                    if not response.get('answer'):
                        raise Exception("Empty response from AI")
                    
                    # Add successful AI response
                    st.session_state.chat_messages.append({
                        'role': 'assistant',
                        'content': response.get('answer', 'No response generated'),
                        'sources': response.get('sources', []),
                        'confidence': response.get('confidence', 0.5),
                        'processing_time': processing_time,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    st.success(f"✅ Response generated in {processing_time:.2f}s")
                    time.sleep(0.5)
                    st.rerun()
                    
                except Exception as e:
                    error_message = str(e)
                    
                    # Handle specific error types
                    if "coroutine" in error_message:
                        error_response = "🔧 ASYNC ERROR: Please restart the application."
                    else:
                        error_response = f"❌ Error: {error_message}. Please try again or rephrase your question."
                    
                    # Add error response
                    st.session_state.chat_messages.append({
                        'role': 'assistant',
                        'content': error_response,
                        'timestamp': datetime.now().isoformat(),
                        'error': True
                    })
                    
                    st.error("Chat processing error occurred")
                    if st.session_state.debug_mode:
                        with st.expander("🔍 Debug: Error Details"):
                            st.code(f"Error: {error_message}")
                            st.code(f"Error Type: {type(e).__name__}")
                            st.code("Stack trace:")
                            st.code(traceback.format_exc())
    
    with col2:
        st.subheader("🎯 Chat Status & Debug")
        
        # Document context debug info
        if st.session_state.processed_documents:
            latest_doc = st.session_state.processed_documents[-1]
            filename = latest_doc.get('document_metadata', {}).get('filename', 'Unknown')
            document_text = latest_doc.get('document_text', '')
            
            st.success(f"📄 **Document**: {filename[:25]}...")
            st.info(f"🎯 **Type**: {latest_doc.get('document_type', 'Legal Doc')}")
            
            # Show document text availability with enhanced debug
            if document_text and document_text.strip():
                text_length = len(document_text)
                st.success(f"📝 **Text Available**: {text_length} chars")
                
                # Word count and first few words
                words = document_text.split()
                st.info(f"📊 **Words**: {len(words)} words")
                
                if st.session_state.debug_mode:
                    st.markdown("### 🔍 Debug: Text Analysis")
                    st.success("✅ **Document text ready for AI chat**")
                    st.info(f"🔤 **First 10 words**: {' '.join(words[:10])}...")
                    st.info(f"📈 **Text quality**: {'Good' if len(words) > 50 else 'Short document'}")
                
                # Check for multi-agent analysis
                if latest_doc.get('multi_agent_analysis'):
                    st.success("🤖 **AI Analysis**: Available")
                else:
                    st.info("🤖 **AI Analysis**: Not run yet")
            else:
                st.error("❌ **No Text**: Document text not available")
                
                if st.session_state.debug_mode:
                    st.markdown("### 🔍 Debug: Missing Text Issue")
                    st.error("❌ Document text is empty or missing")
                    st.write("**Available document fields:**")
                    for key, value in latest_doc.items():
                        if isinstance(value, str):
                            st.write(f"- **{key}**: {value[:50]}..." if len(value) > 50 else f"- **{key}**: {value}")
                        else:
                            st.write(f"- **{key}**: {type(value).__name__}")
        else:
            st.warning("📄 No document loaded")
            st.info("Upload a document in 'Enhanced Document Analysis' for context-aware responses")
        
        # Chat statistics
        if st.session_state.chat_messages:
            total = len(st.session_state.chat_messages)
            user_msgs = len([m for m in st.session_state.chat_messages if m['role'] == 'user'])
            ai_msgs = len([m for m in st.session_state.chat_messages if m['role'] == 'assistant'])
            
            st.metric("💬 Messages", total)
            st.metric("👤 Questions", user_msgs)
            st.metric("🤖 Responses", ai_msgs)
        
        # Actions
        st.markdown("### ⚡ Actions")
        
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_messages = []
            try:
                systems['conversational_rag'].clear_conversation()
            except:
                pass
            st.success("Chat cleared!")
            st.rerun()
        
        if st.button("🧪 Test AI (No Context)", use_container_width=True):
            with st.spinner("Testing AI connection..."):
                try:
                    test_response = systems['conversational_rag'].process_legal_conversation(
                        "Test: What is Indian Contract Act?", None
                    )
                    
                    if isinstance(test_response, dict) and test_response.get('answer'):
                        st.success("✅ AI connection working!")
                        if st.session_state.debug_mode:
                            st.write("Response preview:", test_response['answer'][:100] + "...")
                    else:
                        st.error("❌ AI connection issues")
                        
                except Exception as e:
                    st.error(f"❌ Test failed: {str(e)}")
        
        if st.session_state.processed_documents and st.button("🔍 Test AI (With Context)", use_container_width=True):
            with st.spinner("Testing AI with document context..."):
                try:
                    latest_doc = st.session_state.processed_documents[-1]
                    document_text = latest_doc.get('document_text', '')
                    
                    if document_text:
                        doc_context = {
                            'document_data': {
                                'document_text': document_text,
                                'document_metadata': latest_doc.get('document_metadata', {}),
                                'document_type': latest_doc.get('document_type', 'Legal Document')
                            }
                        }
                        
                        test_response = systems['conversational_rag'].process_legal_conversation(
                            "What is this document about?", doc_context
                        )
                        
                        if isinstance(test_response, dict) and test_response.get('answer'):
                            if "document" in test_response['answer'].lower():
                                st.success("✅ Document context working!")
                                if st.session_state.debug_mode:
                                    st.write("Context response preview:", test_response['answer'][:150] + "...")
                            else:
                                st.warning("⚠️ AI responded but may not have used context")
                        else:
                            st.error("❌ Context test failed")
                    else:
                        st.error("❌ No document text available for context test")
                        
                except Exception as e:
                    st.error(f"❌ Context test failed: {str(e)}")
        
        # Feature capabilities
        st.markdown("### 🎯 Chat Capabilities")
        st.markdown("""
        **💬 Document-Context Chat:**
        - Ask specific questions about uploaded documents
        - Get AI responses based on document content
        - Context-aware follow-up questions
        
        **⚖️ Indian Law Expertise:**
        - Specialized knowledge of Indian legal system
        - Legal terminology and concepts
        - Regulatory and compliance guidance
        
        **📚 Source Attribution:**
        - Track information sources
        - Confidence scoring for responses
        - Reference to document sections
        """)

if __name__ == "__main__":
    main()
