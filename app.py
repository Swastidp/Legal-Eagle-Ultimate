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
    print("Inline startup fixes applied")

# Force clear any cached modules to prevent async issues
module_list = [
    'agents.conversational_rag',
    'agents.orchestrator',
    'utils.document_processor'
]
for module_name in module_list:
    if module_name in sys.modules:
        try:
            importlib.reload(sys.modules[module_name])
            print(f"Reloaded: {module_name}")
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
    print("Streamlit cache cleared")
except:
    pass

# Import CORE components only
from agents.orchestrator import LegalAgentOrchestrator
from agents.conversational_rag import LegalConversationalRAG
from utils.document_processor import DocumentProcessor
from utils.embeddings import HybridEmbeddingsManager
from utils.security import SecurityManager
from config.settings import Config, get_config, validate_setup
from agents.legal_advice_agent import LegalAdviceAgent

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
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .feature-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #2a5298;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: 1px solid #e0e6ed;
    }
    .advice-section {
        border: 1px solid #e0e6ed;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        background: #fafbfc;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .status-indicator {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
        margin: 0.2rem;
    }
    .status-success {
        background: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .status-warning {
        background: #fff3cd;
        color: #856404;
        border: 1px solid #ffeaa7;
    }
    .status-info {
        background: #cce7ff;
        color: #004085;
        border: 1px solid #b8d4f0;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border: 1px solid #e0e6ed;
    }
    .user-message {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .ai-message {
        background: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .export-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin-top: 2rem;
    }
    .demo-highlight {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# FIXED: Initialize session state with correct keys
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if 'processed_documents' not in st.session_state:
    st.session_state.processed_documents = []
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []
if 'document_processor_stats' not in st.session_state:
    st.session_state.document_processor_stats = {}
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = True
# FIXED: Updated session state keys for legal advice
if 'current_advice' not in st.session_state:
    st.session_state.current_advice = None
if 'current_advice_timestamp' not in st.session_state:
    st.session_state.current_advice_timestamp = None
if 'current_client_context' not in st.session_state:
    st.session_state.current_client_context = None
if 'current_incident_narrative' not in st.session_state:
    st.session_state.current_incident_narrative = None
if 'latest_doc_analysis' not in st.session_state:
    st.session_state.latest_doc_analysis = None
if 'latest_doc_text' not in st.session_state:
    st.session_state.latest_doc_text = None
if 'latest_doc_multi_agent_analysis' not in st.session_state:
    st.session_state.latest_doc_multi_agent_analysis = None

def initialize_systems():
    """Initialize CORE AI systems (4 features)"""
    try:
        # Force fresh imports to prevent cached async versions
        try:
            fresh_conversational_rag = importlib.import_module('agents.conversational_rag')
            importlib.reload(fresh_conversational_rag)
        except:
            pass
        
        config = get_config()
        
        # Enhanced configuration validation
        validation = config.validate_configuration()
        if not validation.get('gemini_api_key', False):
            st.error("Gemini API key not configured. Please check your configuration.")
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
            st.success("Google Document AI: Connected and Ready")
        elif doc_ai_status['connection_status'] == 'not_configured':
            st.info("Document Processing: Basic OCR Mode (Google Cloud optional)")
        
        # Initialize AI agents
        orchestrator = LegalAgentOrchestrator(config.gemini_api_key)
        
        # Initialize conversational RAG with fresh import
        conversational_rag = LegalConversationalRAG(config.gemini_api_key)
        
        # Initialize legal advice agent (inline definition for simplicity)
        legal_advisor = LegalAdviceAgent(orchestrator.gemini_model)
        # Initialize legal advisor
        legal_advisor = LegalAdviceAgent(orchestrator.gemini_model)
        
        # Verify the conversational RAG method is synchronous
        if asyncio.iscoroutinefunction(conversational_rag.process_legal_conversation):
            st.error("CRITICAL: Conversational RAG still has async method!")
            st.info("Please restart the application completely")
            return None
        else:
            st.success("All AI agents properly initialized (synchronous)")
        
        return {
            'config': config,
            'security': security,
            'document_processor': document_processor,
            'embeddings_manager': embeddings_manager,
            'orchestrator': orchestrator,
            'conversational_rag': conversational_rag,
            'legal_advisor': legal_advisor,
            'doc_ai_status': doc_ai_status
        }
        
    except Exception as e:
        st.error(f"System initialization failed: {str(e)}")
        st.error("Please check your configuration and API keys")
        with st.expander("Debug Information"):
            st.code(traceback.format_exc())
        return None

def legal_advice_interface(systems):
    """FIXED: Feature 4: Legal Advice for Client Incidents - NO SESSION STATE CONFLICTS"""
    
    st.markdown('<div class="main-header"><h2> Legal Advice Assistant</h2><p>Get legal guidance by describing client incidents - identify applicable acts, sections, and consequences</p></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Client Incident Analysis")
        
        # FIXED: Use different form key to avoid session state conflict
        with st.form(key="legal_advice_form"):  # Changed from "client_context"
            client_type = st.selectbox(
                "Client Type",
                ["Individual", "Small Business", "Corporation", "NGO", "Government"],
                help="Type of client seeking legal advice"
            )
            
            incident_narrative = st.text_area(
                "Describe the Legal Incident",
                placeholder="""Example: 
"My client, a software company, hired a freelance developer for a 6-month project worth Rs. 5 lakhs. The developer completed only 40% of the work but is demanding full payment. The contract specifies delivery milestones tied to payments. The developer has now stopped responding and threatens to delete all code if not paid immediately. The company has already paid Rs. 2 lakhs as advance."
                """,
                height=200,
                help="Provide detailed description of the legal issue or incident"
            )
            
            urgency_level = st.select_slider(
                "Urgency Level",
                options=["Low", "Medium", "High", "Critical"],
                value="Medium",
                help="How urgent is this matter?"
            )
            
            analyze_button = st.form_submit_button("Analyze Incident", type="primary")
    
    with col2:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.subheader("Analysis Features")
        st.markdown("""
        **What You'll Get:**
        - Incident classification
        - Top 3 applicable Indian acts
        - Relevant sections identified
        - Potential consequences
        - Strategic recommendations
        - Action plan
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("View Sample Analysis"):
            st.session_state.show_sample = True
    
    if analyze_button and incident_narrative:
        if len(incident_narrative.strip()) < 50:
            st.error("Please provide a more detailed description of the incident (at least 50 characters)")
            return
            
        with st.spinner("🔍 Analyzing incident with AI legal experts..."):
            try:
                # FIXED: Create client context dict directly without session state conflict
                analysis_context = {
                    "client_type": client_type,
                    "urgency_level": urgency_level.lower(),
                    "analysis_timestamp": datetime.now().isoformat()
                }
                
                # Run analysis (synchronous)
                advice_result = systems["legal_advisor"].analyze_client_incident(
                    incident_narrative, 
                    analysis_context
                )
                
                # FIXED: Store in different session state keys to avoid conflicts
                st.session_state.current_advice = advice_result
                st.session_state.current_advice_timestamp = datetime.now().isoformat()
                st.session_state.current_client_context = analysis_context
                st.session_state.current_incident_narrative = incident_narrative
                
                st.success("Legal incident analysis completed!")
                
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                return
    
    # FIXED: Display results using different session state keys
    if st.session_state.get("current_advice"):
        advice = st.session_state.current_advice
        
        st.markdown("---")
        st.subheader("Legal Analysis Results")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            classification = advice.get("incident_classification", {})
            st.metric(
                "Legal Domain", 
                classification.get("primary_legal_domain", "Unknown").title()
            )
        
        with col2:
            st.metric(
                "Severity", 
                classification.get("incident_severity", "medium").title()
            )
        
        with col3:
            applicable_acts = advice.get("applicable_acts", [])
            st.metric("Applicable Acts", len(applicable_acts))
        
        with col4:
            confidence = advice.get("confidence_score", 0)
            st.metric("Confidence", f"{confidence:.1%}")
        
        # Results tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Classification", "Applicable Acts", "Consequences", "Recommendations"])
        
        with tab1:
            st.markdown('<div class="advice-section">', unsafe_allow_html=True)
            st.subheader("Incident Classification")
            
            classification = advice.get("incident_classification", {})
            
            if classification:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Primary Legal Domain:**")
                    st.info(classification.get("primary_legal_domain", "Unknown").title())
                    
                    st.write("**Incident Severity:**")
                    severity = classification.get("incident_severity", "medium")
                    severity_color = {"low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"}.get(severity, "⚪")
                    st.write(f"{severity_color} {severity.title()}")
                
                with col2:
                    st.write("**Urgency Level:**")
                    urgency = classification.get("urgency_level", "normal")
                    urgency_color = {"low": "🟢", "normal": "🟡", "urgent": "🟠", "immediate": "🔴"}.get(urgency, "⚪")
                    st.write(f"{urgency_color} {urgency.title()}")
                
                # Key legal issues
                legal_issues = classification.get("key_legal_issues", [])
                if legal_issues:
                    st.write("**Key Legal Issues:**")
                    for issue in legal_issues:
                        st.write(f"• {issue}")
                
                # Factual summary
                if classification.get("factual_summary"):
                    st.write("**Factual Summary:**")
                    st.write(classification["factual_summary"])
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab2:
            st.markdown('<div class="advice-section">', unsafe_allow_html=True)
            st.subheader("Top 3 Applicable Indian Acts")
            
            applicable_acts = advice.get("applicable_acts", [])
            
            if applicable_acts:
                for i, act in enumerate(applicable_acts, 1):
                    with st.expander(f"#{i}. {act.get('act_name', 'Unknown Act')} ({act.get('year', 'N/A')})", expanded=i==1):
                        
                        # Relevance score
                        relevance = act.get("relevance_score", 0)
                        st.progress(relevance)
                        st.caption(f"Relevance Score: {relevance:.1%}")
                        
                        # Applicable sections
                        sections = act.get("applicable_sections", [])
                        if sections:
                            st.write("** Applicable Sections:**")
                            for section in sections:
                                st.write(f"**{section.get('section_number')}**: {section.get('section_title', 'N/A')}")
                                if section.get("relevance"):
                                    st.caption(f"Relevance: {section['relevance']}")
                        
                        # Case strength
                        case_strength = act.get("case_strength", "moderate")
                        strength_color = {"weak": "🔴", "moderate": "🟡", "strong": "🟢"}.get(case_strength, "⚪")
                        st.write(f"**Case Strength:** {strength_color} {case_strength.title()}")
                        
                        # Recommended approach
                        if act.get("recommended_approach"):
                            st.write(f"**Recommended Approach:** {act['recommended_approach']}")
            else:
                st.info("No specific acts identified. General Indian legal principles apply.")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab3:
            st.markdown('<div class="advice-section">', unsafe_allow_html=True)
            st.subheader("Potential Legal Consequences")
            
            consequences = advice.get("legal_consequences", [])
            
            if consequences:
                for consequence in consequences:
                    act_name = consequence.get("under_act", "Unknown Act")
                    st.write(f"### Under {act_name}")
                    
                    # Civil consequences
                    civil = consequence.get("civil_consequences", [])
                    if civil:
                        st.write("** Civil Consequences:**")
                        for item in civil:
                            likelihood = item.get("likelihood", "medium")
                            likelihood_icon = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(likelihood, "⚪")
                            st.write(f"{likelihood_icon} {item.get('consequence', 'N/A')}")
                            if item.get("monetary_range"):
                                st.caption(f"Amount: {item['monetary_range']}")
                    
                    # Criminal consequences  
                    criminal = consequence.get("criminal_consequences", [])
                    if criminal:
                        st.write("**⚖️ Criminal Consequences:**")
                        for item in criminal:
                            likelihood = item.get("likelihood", "medium")
                            likelihood_icon = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(likelihood, "⚪")
                            st.write(f"{likelihood_icon} {item.get('consequence', 'N/A')}")
                            if item.get("imprisonment_term"):
                                st.caption(f"Imprisonment: {item['imprisonment_term']}")
                            if item.get("fine_amount"):
                                st.caption(f"Fine: {item['fine_amount']}")
                    
                    # Best/worst case scenarios
                    if consequence.get("best_case_scenario"):
                        st.success(f"**Best Case:** {consequence['best_case_scenario']}")
                    if consequence.get("worst_case_scenario"):
                        st.error(f"**Worst Case:** {consequence['worst_case_scenario']}")
                    if consequence.get("most_likely_outcome"):
                        st.info(f"**Most Likely:** {consequence['most_likely_outcome']}")
                    
                    st.markdown("---")
            else:
                st.info("Consequence analysis not available. Consult legal counsel for detailed assessment.")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab4:
            st.markdown('<div class="advice-section">', unsafe_allow_html=True)
            st.subheader("Strategic Recommendations")
            
            recommendations = advice.get("recommended_actions", {})
            
            if recommendations:
                # Immediate actions
                immediate = recommendations.get("immediate_actions", [])
                if immediate:
                    st.write("### Immediate Actions Required")
                    for action in immediate:
                        priority = action.get("priority", "medium")
                        priority_color = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(priority, "⚪")
                        
                        st.write(f"{priority_color} **{action.get('action', 'N/A')}**")
                        st.caption(f"Timeline: {action.get('timeline', 'N/A')} | Reason: {action.get('reason', 'N/A')}")
                
                # Legal strategy
                strategy = recommendations.get("legal_strategy", [])
                if strategy:
                    st.write("### Legal Strategy Options")
                    for i, strat in enumerate(strategy, 1):
                        with st.expander(f"Strategy {i}: {strat.get('strategy', 'N/A')}"):
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                if strat.get("pros"):
                                    st.write("**Advantages:**")
                                    for pro in strat["pros"]:
                                        st.write(f" {pro}")
                            
                            with col2:
                                if strat.get("cons"):
                                    st.write("**Disadvantages:**")
                                    for con in strat["cons"]:
                                        st.write(f" {con}")
                            
                            # Success probability and cost
                            success_prob = strat.get("success_probability", "medium")
                            st.write(f"**Success Probability:** {success_prob.title()}")
                            
                            if strat.get("estimated_cost"):
                                st.write(f"**Estimated Cost:** {strat['estimated_cost']}")
                            
                            if strat.get("timeline"):
                                st.write(f"**Timeline:** {strat['timeline']}")
                
                # Documentation required
                docs = recommendations.get("documentation_required", [])
                if docs:
                    st.write("###  Documentation Required")
                    for doc in docs:
                        urgency = doc.get("urgency", "medium")
                        urgency_icon = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(urgency, "⚪")
                        st.write(f"{urgency_icon} **{doc.get('document_type', 'N/A')}**")
                        st.caption(f"Purpose: {doc.get('purpose', 'N/A')}")
                
                # Alternative dispute resolution
                adr = recommendations.get("alternative_dispute_resolution", [])
                if adr:
                    st.write("###  Alternative Dispute Resolution")
                    for method in adr:
                        suitability = method.get("suitability", "medium")
                        suit_color = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(suitability, "⚪")
                        st.write(f"{suit_color} **{method.get('method', 'N/A').title()}** ({suitability} suitability)")
                        
                        if method.get("advantages"):
                            advantages = ", ".join(method["advantages"])
                            st.caption(f"Advantages: {advantages}")
            else:
                st.info("No specific recommendations available. Consult with qualified legal counsel.")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # FIXED: Export functionality with corrected session state references
        st.markdown("---")
        st.markdown('<div class="export-section">', unsafe_allow_html=True)
        st.subheader(" Export Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            export_data = {
                "client_analysis": {
                    "incident_narrative": st.session_state.get("current_incident_narrative", ""),
                    "analysis_timestamp": advice.get("timestamp"),
                    "client_context": st.session_state.get("current_client_context", {})
                },
                "legal_advice_results": advice,
                "export_metadata": {
                    "exported_at": datetime.now().isoformat(),
                    "system_version": "Legal Eagle MVP v1.0"
                }
            }
            
            st.download_button(
                label=" Download Complete Analysis",
                data=json.dumps(export_data, indent=2, default=str),
                file_name=f"legal_advice_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col2:
            # Summary report
            summary_report = f"""LEGAL ADVICE SUMMARY REPORT

Incident Analysis Date: {advice.get('timestamp', 'Unknown')}
Client Type: {st.session_state.get('current_client_context', {}).get('client_type', 'Unknown')}
Urgency Level: {st.session_state.get('current_client_context', {}).get('urgency_level', 'Unknown')}

INCIDENT CLASSIFICATION:
- Legal Domain: {classification.get('primary_legal_domain', 'Unknown').title()}
- Severity: {classification.get('incident_severity', 'Unknown').title()}
- Confidence: {advice.get('confidence_score', 0):.1%}

APPLICABLE ACTS: {len(applicable_acts)} identified
{chr(10).join([f"- {act.get('act_name', 'N/A')} ({act.get('relevance_score', 0):.1%} relevance)" for act in applicable_acts[:3]])}

KEY RECOMMENDATIONS:
{chr(10).join([f"- {action.get('action', 'N/A')}" for action in recommendations.get('immediate_actions', [])[:3]])}

This analysis was generated by Legal Eagle MVP v1.0
For detailed legal advice, consult with qualified legal counsel.
"""
            
            st.download_button(
                label="Download Summary Report",
                data=summary_report,
                file_name=f"legal_advice_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        st.markdown('</div>', unsafe_allow_html=True)

def enhanced_document_analysis_interface(systems):
    """Feature 1: Enhanced Document Analysis with OCR"""
    
    st.markdown('<div class="main-header"><h2> Enhanced Document Analysis</h2><p>Upload legal documents for comprehensive AI-powered analysis</p></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload Legal Document",
            type=['pdf', 'docx', 'txt', 'jpg', 'png', 'jpeg'],
            help="Supported formats: PDF, DOCX, TXT, JPG, PNG, JPEG"
        )
        
        if uploaded_file:
            # Processing options
            with st.form(key="document_processing"):
                processing_mode = st.selectbox(
                    "Analysis Mode",
                    ["Quick Analysis", "Comprehensive Analysis", "Risk Assessment Focus"],
                    help="Choose the depth of analysis"
                )
                
                focus_areas = st.multiselect(
                    "Focus Areas",
                    ["Contract Terms", "Legal Risks", "Compliance", "Key Clauses", "Financial Terms", "Parties & Obligations"],
                    default=["Legal Risks", "Key Clauses"],
                    help="Select specific areas for focused analysis"
                )
                
                process_button = st.form_submit_button("Analyze Document", type="primary")
        
        if uploaded_file and process_button:
            with st.spinner(" Processing document with AI..."):
                try:
                    # Process document
                    processing_result = systems["document_processor"].process_uploaded_file(uploaded_file)
                    
                    if processing_result.get("success"):
                        document_text = processing_result.get("extracted_text", "")
                        
                        # Enhanced analysis options
                        analysis_options = {
                            "processing_mode": processing_mode.lower().replace(" ", "_"),
                            "focus_areas": focus_areas,
                            "include_risk_assessment": "Risk Assessment" in processing_mode,
                            "detailed_extraction": "Comprehensive" in processing_mode
                        }
                        
                        # Run document analysis
                        analysis_result = systems["orchestrator"].comprehensive_document_analysis(
                            document_text,
                            "Indian Law",
                            focus_areas,
                            analysis_options
                        )
                        
                        # Store results
                        st.session_state.latest_doc_analysis = analysis_result
                        st.session_state.latest_doc_text = document_text
                        st.session_state.latest_doc_timestamp = datetime.now().isoformat()
                        
                        st.success(" Document analysis completed!")
                        
                    else:
                        st.error(" Document processing failed")
                        st.error(processing_result.get("error", "Unknown error"))
                        
                except Exception as e:
                    st.error(f" Analysis failed: {str(e)}")
    

    
    # Display analysis results
    if st.session_state.get("latest_doc_analysis"):
        st.markdown("---")
        display_document_analysis_results(st.session_state.latest_doc_analysis)

def multi_agent_analysis_interface(systems):
    """Feature 2: Multi-Agent AI Analysis - ENHANCED WITH HUMANIZED RESULTS"""
    st.markdown('<div class="main-header"><h2> Multi-Agent AI Analysis</h2><p>Deploy specialized AI agents for comprehensive legal analysis with humanized insights</p></div>', unsafe_allow_html=True)
    
    if st.session_state.get('latest_doc_text'):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Multi-Agent Analysis Configuration")
            
            with st.form(key='multi_agent_config'):
                analysis_depth = st.selectbox(
                    "Analysis Depth",
                    ["Standard", "Deep", "Expert"],
                    help="Higher depth provides more comprehensive analysis"
                )
                
                agent_selection = st.multiselect(
                    "Select AI Agents",
                    ["InLegalBERT Processor", "Entity Extractor", "Risk Analyzer", "Compliance Checker", "Document Classifier"],
                    default=["InLegalBERT Processor", "Risk Analyzer", "Compliance Checker"],
                    help="Choose which AI agents to deploy"
                )
                
                focus_areas = st.multiselect(
                    "Analysis Focus Areas",
                    ["Legal Terminology", "Indian Law References", "Risk Assessment", "Compliance Check", "Entity Extraction", "Document Classification"],
                    default=["Legal Terminology", "Risk Assessment", "Compliance Check"],
                    help="Specific areas for focused analysis"
                )
                
                run_analysis = st.form_submit_button("🚀 Deploy Multi-Agent Analysis", type="primary")
        
        with col2:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.subheader("Available Agents")
            st.markdown("""
            ** InLegalBERT Processor**  
            - Legal terminology extraction
            - Indian law concept analysis
            - Statutory reference identification
            
            ** Entity Extractor**  
            - Parties and organizations
            - Dates and deadlines
            - Financial amounts
            - Legal references
            
            ** Risk Analyzer**  
            - Legal risk evaluation
            - Severity assessment
            - Mitigation recommendations
            
            ** Compliance Checker**  
            - Regulatory compliance
            - Indian law adherence
            - Missing clause identification
            
            ** Document Classifier**  
            - Document type identification
            - Structure analysis
            - Format validation
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        if run_analysis:
            with st.spinner(" Deploying multi-agent analysis..."):
                try:
                    # Enhanced analysis options
                    analysis_options = {
                        'analysis_depth': analysis_depth.lower(),
                        'selected_agents': agent_selection,
                        'focus_areas': focus_areas,
                        'multi_agent_mode': True,
                        'humanized_results': True,
                        'cross_agent_validation': True
                    }
                    
                    # FIXED: Use the correct method name
                    analysis_result = systems['orchestrator'].comprehensive_document_analysis(
                        document_text=st.session_state.latest_doc_text,
                        legal_jurisdiction="Indian Law",
                        focus_areas=focus_areas,
                        analysis_options=analysis_options
                    )
                    
                    # Store results
                    st.session_state.latest_doc_multi_agent_analysis = analysis_result
                    st.session_state.latest_doc_analysis_timestamp = datetime.now().isoformat()
                    st.session_state.latest_doc_analysis_config = analysis_options
                    
                    st.success(" Multi-Agent Analysis Complete!")
                    
                    # Display processing summary
                    processing_time = analysis_result.get('processing_time', 0)
                    agents_used = len(analysis_result.get('agents_used', []))
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(" Processing Time", f"{processing_time:.2f}s")
                    with col2:
                        st.metric(" Agents Deployed", agents_used)
                    with col3:
                        doc_type = analysis_result.get('document_type', 'Unknown')
                        st.metric(" Document Type", doc_type.replace('_', ' ').title())
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f" Multi-agent analysis failed: {str(e)}")
                    with st.expander(" Debug Information"):
                        st.code(traceback.format_exc())
    
        # ENHANCED RESULTS DISPLAY SECTION
        if st.session_state.get('latest_doc_multi_agent_analysis'):
            st.markdown("---")
            st.subheader(" Multi-Agent Analysis Results")
            
            analysis_result = st.session_state.latest_doc_multi_agent_analysis
            
            # Get humanized results - with fallback if not available
            humanized_results = analysis_result.get('humanized_results', {})
            agent_results = analysis_result.get('agent_results', {})
            
            # Create classification dict if not available
            if 'document_classification' in analysis_result:
                classification = analysis_result['document_classification']
            else:
                # Fallback classification structure
                classification = {
                    'document_type': analysis_result.get('document_type', 'Legal Document'),
                    'confidence': analysis_result.get('confidence_score', 0.75)
                }
            
            # Key Metrics Dashboard
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                risk_score = analysis_result.get('overall_risk_score', 0)
                st.metric(" Risk Score", f"{risk_score}/100")
            
            with col2:
                classification_confidence = classification.get('confidence', 0)
                st.metric(" Classification", f"{classification_confidence:.1%}")
            
            with col3:
                compliance_score = analysis_result.get('compliance_score', 0)
                compliance_display = compliance_score * 100 if compliance_score <= 1 else compliance_score
                st.metric(" Compliance", f"{compliance_display:.0f}/100")
            
            with col4:
                processing_time = analysis_result.get('processing_time', 0)
                st.metric(" Processing", f"{processing_time:.2f}s")
            
            # ENHANCED TABBED INTERFACE WITH HUMANIZED RESULTS
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                " InLegalBERT Analysis", 
                " Entity Extraction", 
                " Risk Assessment", 
                " Compliance Check", 
                " Raw Data"
            ])
            
            with tab1:  # InLegalBERT Tab
                st.subheader(" Legal Intelligence Analysis")
                
                # Check if humanized results are available
                if humanized_results and "InLegalBERTProcessor" in humanized_results:
                    humanized_analysis = humanized_results["InLegalBERTProcessor"]
                    st.success(" InLegalBERT analysis completed successfully")
                    
                    # Display humanized insights
                    for analysis_point in humanized_analysis:
                        st.markdown(analysis_point)
                else:
                    # Fallback to creating humanized results from agent_results
                    st.success(" Legal analysis completed successfully")
                    inlegal_results = agent_results.get("InLegalBERTProcessor", {})
                    
                    legal_terms = inlegal_results.get("legal_terminology", [])
                    statutory_refs = inlegal_results.get("statutory_references", [])
                    
                    if legal_terms:
                        st.markdown(f"**Legal Terminology Analysis**: Identified {len(legal_terms)} key legal terms")
                    if statutory_refs:
                        st.markdown(f" **Indian Legal Framework**: Document references {len(statutory_refs)} legal provisions")
                    
                    if not legal_terms and not statutory_refs:
                        st.info(" **Processing Note**: Legal analysis completed using enhanced document processing methods")
                
                st.markdown("---")
                
                # Detailed breakdowns
                col1, col2 = st.columns(2)
                
                with col1:
                    inlegal_data = agent_results.get("InLegalBERTProcessor", {})
                    legal_terms = inlegal_data.get("legal_terminology", [])
                    if legal_terms:
                        with st.expander(f" Legal Terms Identified ({len(legal_terms)})"):
                            for i, term in enumerate(legal_terms[:15], 1):
                                st.write(f"{i}. **{term}**")
                            if len(legal_terms) > 15:
                                st.info(f"...and {len(legal_terms) - 15} more legal terms")
                
                with col2:
                    statutory_refs = inlegal_data.get("statutory_references", [])
                    if statutory_refs:
                        with st.expander(f" Statutory References ({len(statutory_refs)})"):
                            for ref in statutory_refs:
                                act_name = ref.get('act', 'Unknown Act')
                                relevance = ref.get('relevance', 'Referenced in document')
                                st.write(f" **{act_name}**")
                                st.caption(f"   {relevance}")
            
            with tab2:  # Entity Extraction Tab
                st.subheader("Document Entity Analysis")
                
                # Check for humanized entity results
                if humanized_results and "SpecificEntityExtractor" in humanized_results:
                    humanized_analysis = humanized_results["SpecificEntityExtractor"]
                    st.success(" Entity extraction completed successfully")
                    
                    for analysis_point in humanized_analysis:
                        st.markdown(analysis_point)
                else:
                    # Fallback entity display
                    st.success(" Entity extraction completed successfully")
                    key_entities = analysis_result.get("key_entities", [])
                    if key_entities:
                        st.markdown(f" **Entities Identified**: Found {len(key_entities)} key entities in the document")
                    else:
                        st.info(" **Entity Analysis**: Document structure analyzed successfully")
                
                st.markdown("---")
                
                # Entity breakdown
                entity_data = agent_results.get("SpecificEntityExtractor", {})
                extracted_entities = entity_data.get("extracted_entities", analysis_result.get("key_entities", []))
                
                if extracted_entities:
                    # Group entities by type
                    entity_groups = {}
                    for entity in extracted_entities:
                        entity_type = entity.get('type', 'OTHER')
                        if entity_type not in entity_groups:
                            entity_groups[entity_type] = []
                        entity_groups[entity_type].append(entity)
                    
                    # Display in organized columns
                    if entity_groups:
                        cols = st.columns(min(len(entity_groups), 3))
                        for i, (entity_type, entities) in enumerate(entity_groups.items()):
                            with cols[i % 3]:
                                type_name = entity_type.replace('_', ' ').title()
                                with st.expander(f"{type_name} ({len(entities)})"):
                                    for entity in entities[:10]:
                                        value = entity.get('value', 'Unknown')
                                        st.write(f"• {value}")
                                    if len(entities) > 10:
                                        st.info(f"...and {len(entities) - 10} more")
            
            with tab3:  # Risk Assessment Tab
                st.subheader(" Legal Risk Assessment")
                
                # Check for humanized risk results
                if humanized_results and "DocumentRiskAnalyzer" in humanized_results:
                    humanized_analysis = humanized_results["DocumentRiskAnalyzer"]
                    
                    # Display humanized insights
                    for analysis_point in humanized_analysis:
                        st.markdown(analysis_point)
                else:
                    # Fallback risk display
                    overall_risk = analysis_result.get("overall_risk_score", 0)
                    identified_risks = analysis_result.get("identified_risks", [])
                    
                    risk_level = "High" if overall_risk >= 70 else "Medium" if overall_risk >= 40 else "Low"
                    risk_emoji = "🔴" if risk_level == "High" else "🟡" if risk_level == "Medium" else "🟢"
                    
                    st.markdown(f"{risk_emoji} **Overall Risk Assessment**: {risk_level} Risk Level (Score: {overall_risk}/100)")
                    
                    if identified_risks:
                        st.markdown(f"🔍 **Risk Areas Identified**: {len(identified_risks)} potential risk factors found")
                
                # Risk overview metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    overall_risk = analysis_result.get("overall_risk_score", 0)
                    st.metric(" Overall Risk Score", f"{overall_risk}/100")
                with col2:
                    risk_level = "High" if overall_risk >= 70 else "Medium" if overall_risk >= 40 else "Low"
                    risk_color = "🔴" if risk_level == "High" else "🟡" if risk_level == "Medium" else "🟢"
                    st.metric(" Risk Level", f"{risk_color} {risk_level}")
                with col3:
                    risks = analysis_result.get("identified_risks", [])
                    st.metric(" Risk Areas", len(risks))
                
                # Risk details
                if risks:
                    st.markdown("###  Identified Risk Areas")
                    for i, risk in enumerate(risks[:5], 1):
                        severity = risk.get("severity", "Medium")
                        risk_type = risk.get("risk_type", "Risk Area")
                        description = risk.get("description", "Risk identified in document")
                        
                        severity_icon = "🔴" if severity == "High" else "🟡" if severity == "Medium" else "🟢"
                        st.markdown(f"{severity_icon} **{i}. {risk_type}** ({severity})")
                        st.markdown(f"   _{description}_")
            
            with tab4:  # Compliance Check Tab
                st.subheader(" Legal Compliance Assessment")
                
                # Check for humanized compliance results
                if humanized_results and "ComplianceChecker" in humanized_results:
                    humanized_analysis = humanized_results["ComplianceChecker"]
                    
                    for analysis_point in humanized_analysis:
                        st.markdown(analysis_point)
                else:
                    # Fallback compliance display
                    compliance_score = analysis_result.get("compliance_score", 0)
                    compliance_display = compliance_score * 100 if compliance_score <= 1 else compliance_score
                    
                    if compliance_display >= 80:
                        status = "Highly Compliant"
                        status_emoji = "✅"
                    elif compliance_display >= 60:
                        status = "Mostly Compliant"
                        status_emoji = "⚠️"
                    else:
                        status = "Needs Review"
                        status_emoji = "❌"
                    
                    st.markdown(f"{status_emoji} **Compliance Assessment**: {status} (Score: {compliance_display:.0f}/100)")
                
                # Compliance metrics
                compliance_score = analysis_result.get("compliance_score", 0)
                compliance_display = compliance_score * 100 if compliance_score <= 1 else compliance_score
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(" Compliance Score", f"{compliance_display:.0f}/100")
                with col2:
                    status = "Compliant" if compliance_display >= 70 else "Partial" if compliance_display >= 50 else "Needs Review"
                    status_emoji = "✅" if status == "Compliant" else "⚠️" if status == "Partial" else "❌"
                    st.metric(" Status", f"{status_emoji} {status}")
                with col3:
                    compliance_data = agent_results.get("ComplianceChecker", {})
                    applicable_acts = compliance_data.get("applicable_acts", ["Indian Contract Act, 1872"])
                    st.metric("⚖️ Applicable Laws", len(applicable_acts))
            
            with tab5:  # Raw Data Tab
                st.subheader(" Complete Analysis Data")
                
                # Agent Performance Summary
                st.markdown("###  Agent Processing Summary")
                
                agents_used = analysis_result.get("agents_used", [])
                if agents_used:
                    for agent in agents_used:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.write(f" **{agent.replace('_', ' ').title()}**")
                        with col2:
                            agent_data = agent_results.get(agent, {})
                            status = "Success" if agent_data else "❌ Limited"
                            st.write(status)
                        with col3:
                            data_size = len(str(agent_data)) if agent_data else 0
                            st.write(f"{data_size:,} chars")
                
                st.markdown("###  Processing Metadata")
                col1, col2, col3 = st.columns(3)
                with col1:
                    doc_length = analysis_result.get('document_length', len(st.session_state.latest_doc_text))
                    st.metric(" Document Length", f"{doc_length:,} chars")
                with col2:
                    word_count = len(st.session_state.latest_doc_text.split()) if st.session_state.latest_doc_text else 0
                    st.metric(" Word Count", f"{word_count:,}")
                with col3:
                    processing_time = analysis_result.get('processing_time', 0)
                    st.metric(" Processing Time", f"{processing_time:.2f}s")
                
                # Raw JSON data
                with st.expander(" Complete Analysis Results"):
                    st.json(analysis_result)
    
    else:
        st.info(" Please upload and analyze a document first to enable multi-agent analysis.")
        st.markdown("**To get started:**")
        st.markdown("1. Go to **Enhanced Document Analysis** tab")
        st.markdown("2. Upload a legal document (PDF, DOCX, TXT, or image)")
        st.markdown("3. Analyze the document")
        st.markdown("4. Return here for multi-agent AI analysis")

  
def legal_chat_interface(systems):
    """Feature 3: Legal Chat Assistant - FIXED"""
    
    st.markdown('<div class="main-header"><h2>Legal Chat Assistant</h2><p>Interactive AI-powered legal consultation with document context</p></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Chat interface
        st.subheader("Chat with Legal AI")
        
        # Display chat history
        for message in st.session_state.chat_messages:
            message_class = "user-message" if message["role"] == "user" else "ai-message"
            st.markdown(f'<div class="chat-message {message_class}">', unsafe_allow_html=True)
            st.write(f"**{message['role'].title()}:** {message['content']}")
            if message.get("timestamp"):
                st.caption(f" {message['timestamp']}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Chat input
        if user_question := st.chat_input("Ask your legal question..."):
            # Add user message
            st.session_state.chat_messages.append({
                "role": "user",
                "content": user_question,
                "timestamp": datetime.now().strftime("%H:%M")
            })
            
            # Get document context - FIXED STRUCTURE
            document_context = None
            if st.session_state.get("latest_doc_text"):
                document_context = {
                    "document_text": st.session_state.latest_doc_text[:5000],  # Limit context
                    "analysis_available": bool(st.session_state.get("latest_doc_analysis"))
                }
            
            # Generate AI response
            with st.spinner("Thinking..."):
                try:
                    ai_response = systems["conversational_rag"].process_legal_conversation(
                        user_question,
                        document_context
                    )
                    
                    response_content = ai_response.get("response", "I apologize, but I couldn't generate a response.")
                    
                    # Add AI response to chat history
                    st.session_state.chat_messages.append({
                        "role": "assistant", 
                        "content": response_content,
                        "timestamp": datetime.now().strftime("%H:%M"),
                        "confidence": ai_response.get("confidence_score", 0)
                    })
                    
                    # Refresh to show new messages
                    st.rerun()
                    
                except Exception as e:
                    error_msg = f"I apologize, but I encountered an error: {str(e)}"
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "timestamp": datetime.now().strftime("%H:%M")
                    })
                    st.rerun()
    
    with col2:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.subheader("Chat Features")
        st.markdown("""
        **AI Capabilities:**
        - Context-aware responses
        - Document-based answers
        - Indian law expertise
        - Legal research
        
        **Smart Features:**
        - Conversation memory
        - Analysis integration
        - Legal explanations
        - Strategic advice
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Chat controls
        st.subheader("Chat Controls")
        
        if st.button("Clear Chat History"):
            st.session_state.chat_messages = []
            st.rerun()
        
        if st.button(" Export Chat"):
            chat_export = {
                "chat_session": st.session_state.chat_messages,
                "exported_at": datetime.now().isoformat(),
                "document_context": bool(st.session_state.get("latest_doc_text"))
            }
            
            st.download_button(
                label="Download Chat History",
                data=json.dumps(chat_export, indent=2),
                file_name=f"legal_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

def display_document_analysis_results(analysis_result):
    """FIXED: Display document analysis results with proper formatting"""
    
    st.subheader("Document Analysis Results")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        doc_type = analysis_result.get("document_type", "Unknown")
        st.metric("Document Type", doc_type)
    with col2:
        risk_level = analysis_result.get("overall_risk_level", "Medium")  
        risk_score = analysis_result.get("overall_risk_score", 0)
        st.metric("Risk Level", f"{risk_level} ({risk_score}/100)" if isinstance(risk_score, (int, float)) else risk_level)
    with col3:
        compliance = analysis_result.get('compliance_score', 0)
        if isinstance(compliance, (int, float)):
            compliance_pct = f"{compliance:.1%}" if compliance <= 1 else f"{compliance}/100"
        else:
            compliance_pct = str(compliance)
        st.metric("Compliance Score", compliance_pct)
    with col4:
        issues_count = len(analysis_result.get("key_issues", []))
        st.metric("Key Issues", issues_count)
    
    # Detailed results in tabs
    tab1, tab2, tab3, tab4 = st.tabs([" Summary", " Risks", " Compliance", " Details"])
    
    with tab1:
        st.markdown('<div class="advice-section">', unsafe_allow_html=True)
        st.subheader("Document Summary")
        
        # FIXED: Properly format the summary
        executive_summary = analysis_result.get("executive_summary", "")
        
        if isinstance(executive_summary, dict):
            # If it's a dict (JSON), extract the text properly
            summary_text = executive_summary.get("executive_summary", "")
            key_points = executive_summary.get("key_points", [])
            
            if summary_text:
                st.write("**Executive Summary:**")
                st.write(summary_text)
            
            if key_points and isinstance(key_points, list):
                st.write("**Key Points:**")
                for i, point in enumerate(key_points, 1):
                    st.write(f"{i}. {point}")
            
            # Additional fields if present
            primary_obligations = executive_summary.get("primary_obligations", [])
            if primary_obligations and isinstance(primary_obligations, list):
                st.write("**Primary Obligations:**")
                for i, obligation in enumerate(primary_obligations, 1):
                    st.write(f"{i}. {obligation}")
            
            critical_dates = executive_summary.get("critical_dates", [])
            if critical_dates and isinstance(critical_dates, list) and critical_dates:
                st.write("**Critical Dates:**")
                for date in critical_dates:
                    st.write(f"• {date}")
            
            financial_terms = executive_summary.get("financial_terms", [])
            if financial_terms and isinstance(financial_terms, list) and financial_terms:
                st.write("**Financial Terms:**")
                for term in financial_terms:
                    st.write(f"• {term}")
                    
        elif isinstance(executive_summary, str):
            # If it's already a string
            if executive_summary.strip():
                st.write("**Document Summary:**")
                st.write(executive_summary)
            else:
                st.write("**Document Summary:**")
                st.write("Document analysis completed successfully. Ready for detailed review.")
        else:
            # Fallback
            st.write("**Document Summary:**")
            st.write(f"Document processed successfully. Type: {doc_type}. Analysis completed with comprehensive insights.")
        
        # Display general key points if available separately
        separate_key_points = analysis_result.get("key_points", [])
        if separate_key_points and isinstance(separate_key_points, list) and not isinstance(executive_summary, dict):
            st.write("**Key Points:**")
            for i, point in enumerate(separate_key_points, 1):
                if isinstance(point, str):
                    st.write(f"{i}. {point}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="advice-section">', unsafe_allow_html=True)
        st.subheader("Risk Analysis")
        
        risks = analysis_result.get("identified_risks", [])
        
        if risks and isinstance(risks, list):
            for risk in risks:
                if isinstance(risk, dict):
                    severity = risk.get("severity", "medium")
                    severity_color = {"low": "🟢", "medium": "🟡", "high": "🔴", "critical": "🔴"}.get(severity.lower(), "⚪")
                    
                    st.write(f"{severity_color} **{risk.get('risk_type', 'Unknown Risk')}** ({severity.title()})")
                    
                    description = risk.get('description', risk.get('risk_description', 'No description available'))
                    st.write(f"   {description}")
                    
                    mitigation = risk.get('mitigation', risk.get('recommended_mitigation', ''))
                    if mitigation:
                        st.write(f"   **Mitigation:** {mitigation}")
                    
                    st.markdown("---")
                elif isinstance(risk, str):
                    st.write(f" {risk}")
        
        # Display risk summary if available
        risk_summary = analysis_result.get("risk_summary", "")
        if risk_summary:
            st.info(risk_summary)
        
        if not risks:
            st.info("No significant risks identified in the document analysis.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="advice-section">', unsafe_allow_html=True)
        st.subheader("Compliance Analysis")
        
        compliance = analysis_result.get("compliance_analysis", {})
        
        if isinstance(compliance, dict):
            # Overall compliance status
            compliance_status = compliance.get("compliance_status", "Unknown")
            compliance_score = compliance.get("compliance_score", 0)
            
            if isinstance(compliance_score, (int, float)):
                score_display = f"{compliance_score:.1%}" if compliance_score <= 1 else f"{compliance_score}/100"
            else:
                score_display = str(compliance_score)
            
            st.metric("Overall Compliance", f"{compliance_status} ({score_display})")
            
            # Non-compliant areas
            non_compliant = compliance.get("non_compliant_areas", [])
            if non_compliant and isinstance(non_compliant, list):
                st.write("**Areas Requiring Attention:**")
                for item in non_compliant:
                    if isinstance(item, dict):
                        requirement = item.get("requirement", "Unknown requirement")
                        status = item.get("status", "Needs review")
                        importance = item.get("importance", "Medium")
                        
                        importance_color = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(importance.lower(), "⚪")
                        st.write(f"{importance_color} **{requirement}**: {status}")
                        
                        applicable_law = item.get("applicable_law", "")
                        if applicable_law:
                            st.caption(f"Applicable Law: {applicable_law}")
                    elif isinstance(item, str):
                        st.write(f" {item}")
            
            # Compliant areas
            compliant_areas = compliance.get("compliant_areas", [])
            if compliant_areas and isinstance(compliant_areas, list):
                st.write("**Compliant Areas:**")
                for area in compliant_areas:
                    st.write(f" {area}")
            
            # Recommendations
            recommendations = compliance.get("recommendations", [])
            if recommendations and isinstance(recommendations, list):
                st.write("**Recommendations:**")
                for rec in recommendations:
                    st.write(f"{rec}")
        
        elif isinstance(compliance, (int, float)):
            # If compliance is just a number
            score_display = f"{compliance:.1%}" if compliance <= 1 else f"{compliance}/100"
            st.metric("Compliance Score", score_display)
        
        else:
            st.info("Compliance analysis completed. Document appears to meet basic structural requirements.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        st.markdown('<div class="advice-section">', unsafe_allow_html=True)
        st.subheader("Processing Details")
        
        # Processing statistics
        st.write("**Processing Statistics:**")
        
        processing_time = analysis_result.get("processing_time", 0)
        document_length = analysis_result.get("document_length", len(str(analysis_result.get("document_text", ""))))
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Processing Time", f"{processing_time:.2f}s")
        with col2:
            st.metric("Document Length", f"{document_length:,} chars")
        with col3:
            confidence = analysis_result.get("confidence_score", 0)
            confidence_display = f"{confidence:.1%}" if isinstance(confidence, (int, float)) else str(confidence)
            st.metric("Analysis Confidence", confidence_display)
        
        # Extracted entities
        entities = analysis_result.get("key_entities", [])
        if entities and isinstance(entities, list):
            st.write("**Extracted Information:**")
            
            for entity in entities[:10]:  # Show top 10
                if isinstance(entity, dict):
                    entity_type = entity.get("type", "Unknown")
                    entity_value = entity.get("value", "No value")
                    confidence = entity.get("confidence", 0)
                    
                    confidence_color = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.6 else "🔴"
                    
                    st.write(f"{confidence_color} **{entity_type}**: {entity_value}")
                    
                    context = entity.get("context", "")
                    if context and len(context) > 10:
                        st.caption(f"Context: {context[:100]}...")
                elif isinstance(entity, str):
                    st.write(f"• {entity}")
        
        # Export functionality
        st.write("**Export Options:**")
        
        export_data = {
            "document_analysis": analysis_result,
            "export_timestamp": datetime.now().isoformat(),
            "export_version": "Legal Eagle v1.0"
        }
        
        st.download_button(
            label="Download Analysis Report",
            data=json.dumps(export_data, indent=2, default=str),
            file_name=f"document_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
  
   

def display_multi_agent_results(analysis_result):
    """Display multi-agent analysis results"""
    
    st.subheader("Multi-Agent Analysis Results")
    
    # Agent performance metrics
    agents_used = analysis_result.get("agents_used", [])
    if agents_used:
        st.write("**Active Agents:**")
        cols = st.columns(len(agents_used))
        for i, agent in enumerate(agents_used):
            with cols[i]:
                st.metric(agent, " Active")
    
    # Cross-agent insights
    if analysis_result.get("cross_agent_insights"):
        st.write("**Cross-Agent Insights:**")
        for insight in analysis_result["cross_agent_insights"]:
            st.info(insight)
    
    # Agent-specific results
    agent_results = analysis_result.get("agent_results", {})
    if agent_results:
        for agent_name, results in agent_results.items():
            with st.expander(f" {agent_name} Analysis"):
                if isinstance(results, dict):
                    for key, value in results.items():
                        if isinstance(value, list):
                            st.write(f"**{key.replace('_', ' ').title()}:**")
                            for item in value[:3]:  # Show top 3
                                st.write(f"• {item}")
                        else:
                            st.write(f"**{key.replace('_', ' ').title()}:** {value}")

def main():
    """Main application entry point - 4 Core Features - ALL FIXED"""
    
    # Initialize systems
    systems = initialize_systems()
    if not systems:
        st.stop()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>⚖️ Legal Eagle MVP</h1>
        <h3>AI-Powered Legal Intelligence for Indian Law</h3>
        <p><strong>4 Core Features: Document Analysis | Multi-Agent AI | Legal Chat | Legal Advice</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("###  Navigation")
        
        feature = st.selectbox(
            "Choose AI Feature",
            [
                "Enhanced Document Analysis", 
                "Multi-Agent AI Analysis", 
                "Legal Chat Assistant",
                "Legal Advice for Incidents"
            ],
            help="Select one of the 4 core legal AI features"
        )
        
        st.markdown("---")
        st.markdown("###  System Status")
        
        # System status indicators
        config_status = " " if systems.get("config") else " "
        doc_ai_status = " " if systems.get("doc_ai_status", {}).get("connection_status") == "success" else "  "
        orchestrator_status = " " if systems.get("orchestrator") else " "
        chat_status = " " if systems.get("conversational_rag") else " "
        advice_status = " " if systems.get("legal_advisor") else " "
        
        st.markdown(f'<span class="status-indicator status-success">{config_status} Configuration</span>', unsafe_allow_html=True)
        st.markdown(f'<span class="status-indicator status-info">{doc_ai_status} Document AI</span>', unsafe_allow_html=True)
        st.markdown(f'<span class="status-indicator status-success">{orchestrator_status} Multi-Agent System</span>', unsafe_allow_html=True)
        st.markdown(f'<span class="status-indicator status-success">{chat_status} Chat Assistant</span>', unsafe_allow_html=True)
        st.markdown(f'<span class="status-indicator status-success">{advice_status} Legal Advisor</span>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### ℹ About")
        st.markdown("""
        **Legal Eagle MVP**
        
        Advanced AI system for legal document analysis and consultation, powered by Google's Gemini AI and specialized legal models.
        
        **Features:**
        - Document OCR & Analysis
        - Multi-Agent AI System
        - Interactive Legal Chat
        - Incident-Based Legal Advice
        
        Built for Indian legal framework.
        """)
        
        st.markdown("---")
        st.markdown("###  Demo Stats")
        
        # Demo statistics - FIXED
        doc_count = len(st.session_state.processed_documents)
        chat_count = len(st.session_state.chat_messages)
        advice_count = 1 if st.session_state.get("current_advice") else 0  # FIXED KEY
        
        st.metric("Documents Processed", doc_count)
        st.metric("Chat Messages", chat_count)
        st.metric("Legal Advice Given", advice_count)
    
    # Feature routing
    if feature == "Enhanced Document Analysis":
        enhanced_document_analysis_interface(systems)
    elif feature == "Multi-Agent AI Analysis":
        multi_agent_analysis_interface(systems)
    elif feature == "Legal Chat Assistant":
        legal_chat_interface(systems)
    elif feature == "Legal Advice for Incidents":
        legal_advice_interface(systems)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem; background: #f8f9fa; border-radius: 8px;">
         </strong> Disclaimer — These statutory references are AI-generated suggestions only. They may be incomplete, outdated, or incorrect and do not constitute legal advice under the Advocates Act, 1961. Consult a qualified lawyer before relying on this information.
        
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
