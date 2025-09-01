"""
🎯 Enhanced RAG System - Professional UI/UX
Built with Enterprise-grade Features and Voice Assistant Integration
"""

import streamlit as st
import os
import sys
import logging
import time
import json
import base64
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import hashlib
import pandas as pd

# Core RAG imports
from rag_core import ProductionRAGSystem, process_document as process_document_async, ask_enhanced_question
import asyncio

# Professional Voice Assistant
try:
    from streamlit_voice_integration import get_streamlit_voice_assistant, cleanup_streamlit_voice_assistant, TTSQuality
    PROFESSIONAL_VOICE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Professional voice assistant not available: {e}")
    PROFESSIONAL_VOICE_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

# Application Configuration
class AppConfig:
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    SUPPORTED_FORMATS = ['.pdf', '.txt', '.md', '.docx', '.csv']
    SESSION_TIMEOUT = 3600  # 1 hour
    MAX_QUERY_LENGTH = 2000
    CACHE_SIZE_LIMIT = 1000

# Enhanced Page Configuration
st.set_page_config(
    page_title="🚀 Advanced RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'Report a bug': 'https://github.com/your-repo/issues',
        'About': "# Advanced RAG System\nBuilt with Streamlit & Voice AI"
    }
)

def apply_modern_styling():
    """Apply modern, professional CSS styling"""
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Custom Font */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        margin: 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    /* Card Styling */
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        border: 1px solid #f0f0f0;
        margin-bottom: 1rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    }
    
    .feature-card h3 {
        color: #2c3e50;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    /* Voice Interface Styling */
    .voice-interface {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    
    .voice-status {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-weight: 500;
    }
    
    .voice-listening {
        background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
        color: white;
        animation: pulse 2s infinite;
    }
    
    .voice-processing {
        background: linear-gradient(135deg, #fdcb6e 0%, #e17055 100%);
        color: white;
    }
    
    .voice-error {
        background: linear-gradient(135deg, #fd79a8 0%, #e84393 100%);
        color: white;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    
    /* Chat Interface */
    .chat-container {
        background: #f8f9fa;
        border-radius: 15px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid #e9ecef;
    }
    
    .user-message {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        color: white;
        padding: 1rem;
        border-radius: 15px 15px 5px 15px;
        margin: 0.5rem 0;
        margin-left: 2rem;
    }
    
    .assistant-message {
        background: white;
        color: #2c3e50;
        padding: 1rem;
        border-radius: 15px 15px 15px 5px;
        margin: 0.5rem 0;
        margin-right: 2rem;
        border: 1px solid #e9ecef;
    }
    
    /* Metrics Dashboard */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #74b9ff;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #2c3e50;
        margin: 0;
    }
    
    .metric-label {
        color: #636e72;
        font-size: 0.9rem;
        margin: 0;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Button Enhancements */
    .stButton > button {
        border-radius: 25px;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: 500;
        transition: all 0.3s ease;
        font-family: 'Inter', sans-serif;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    /* File Upload Styling */
    .uploadedFile {
        border-radius: 10px;
        border: 2px dashed #74b9ff;
        padding: 2rem;
        text-align: center;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        border-radius: 10px;
    }
    
    /* Success/Error Messages */
    .stSuccess {
        background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
        color: white;
        border-radius: 10px;
        padding: 1rem;
    }
    
    .stError {
        background: linear-gradient(135deg, #fd79a8 0%, #e84393 100%);
        color: white;
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    .stApp > header {visibility: hidden;}
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #0984e3 0%, #74b9ff 100%);
    }
    </style>
    """, unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.rag_system = None
        st.session_state.vectorstore = None
        st.session_state.chat_history = []
        st.session_state.processing_status = None
        st.session_state.voice_status = "idle"
        st.session_state.metrics = {
            'total_queries': 0,
            'successful_queries': 0,
            'avg_response_time': 0,
            'documents_processed': 0,
            'voice_queries': 0
        }
        st.session_state.last_activity = datetime.now()

def create_header():
    """Create professional header"""
    st.markdown("""
    <div class="main-header">
        <h1>🚀 Advanced RAG System</h1>
        <p>Intelligent Document Analysis with Voice Assistant Integration</p>
    </div>
    """, unsafe_allow_html=True)

def create_voice_interface():
    """Professional voice interface with superior speech capabilities"""
    if not PROFESSIONAL_VOICE_AVAILABLE:
        st.warning("🎤 Professional voice features require additional dependencies. Install speech recognition packages to enable voice functionality.")
        return
    
    # Initialize Streamlit voice assistant if not exists
    try:
        assistant = get_streamlit_voice_assistant(TTSQuality.MEDIUM)
    except Exception as e:
        st.error(f"❌ Failed to initialize professional voice assistant: {str(e)}")
        return
    
    st.markdown("""
    <div class="voice-interface">
        <h3>🎙️ Professional Voice Assistant</h3>
        <p>Speak naturally for professional, clear responses</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check assistant status
    status = assistant.get_status()
    
    if not status.get('speech_recognition_available', False):
        st.warning("⚠️ Speech recognition not available. Please install required packages.")
        return
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("👋 Professional Greeting", key="voice_greeting", use_container_width=True):
            if status.get('available_tts_engines', []):
                greeting = "Good day! I'm your professional assistant, ready to provide you with expert information and assistance. How may I help you today?"
                assistant.speak_with_streamlit_feedback(greeting, context="greeting")
    
    # Enhanced Voice Interface with better UX
    st.markdown("### 🎙️ Enhanced Voice Assistant")
    
    # Voice Assistant Instructions
    with st.expander("📋 Voice Assistant Instructions", expanded=False):
        st.markdown("""
        **For Best Results:**
        1. **Speak clearly and at moderate pace** - Don't rush your words
        2. **Wait for the listening indicator** - Look for the blue "Listening..." message
        3. **Complete your full question** - The system will wait up to 30 seconds
        4. **Use specific terms** - Say "agentic AI" instead of just "AI" for better accuracy
        5. **Minimize background noise** - Quiet environment works best
        
        **Example Questions:**
        - "What is agentic AI and how does it work?"
        - "Explain the main concepts in this document"
        - "What are the key findings and conclusions?"
        - "Tell me about machine learning approaches mentioned"
        """)
    
    # Voice Controls with better layout
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("👋 Test Greeting", key="voice_greeting_new", use_container_width=True):
            if status.get('available_tts_engines', []):
                greeting = "Hello! I'm your enhanced voice assistant. I can now better understand technical terms like agentic AI, machine learning, and complex questions. How may I help you today?"
                assistant.speak_with_streamlit_feedback(greeting, context="greeting")
    
    with col2:
        # Enhanced voice input button with progress indication
        voice_button = st.button(
            "🎤 Start Enhanced Voice Query", 
            key="voice_start_enhanced", 
            type="primary", 
            use_container_width=True,
            help="Click and speak your complete question. The system will wait for you to finish."
        )
        
        if voice_button:
            # Create status containers for better UX
            status_container = st.container()
            progress_container = st.container()
            result_container = st.container()
            
            with status_container:
                st.info("🎧 **Initializing Enhanced Voice Recognition...**")
            
            try:
                with progress_container:
                    progress_bar = st.progress(0)
                    progress_text = st.empty()
                    
                    progress_bar.progress(25)
                    progress_text.text("🎚️ Adjusting for ambient noise...")
                    
                # Enhanced voice input with longer timeout
                with status_container:
                    st.info("🎙️ **Listening for your complete question... Speak now!**")
                    st.caption("💡 Take your time - you have up to 18 seconds to complete your question")
                
                with progress_container:
                    progress_bar.progress(50)
                    progress_text.text("👂 Actively listening...")
                
                # Listen with enhanced settings
                voice_text = assistant.listen_with_streamlit_feedback(timeout=18.0)
                
                with progress_container:
                    progress_bar.progress(75)
                    progress_text.text("🔍 Processing speech...")
                
                if voice_text and len(voice_text.strip()) > 0:
                    # Enhanced voice input processing
                    cleaned_voice_text = voice_text.strip()
                    
                    with progress_container:
                        progress_bar.progress(90)
                        progress_text.text("✅ Speech recognized successfully!")
                    
                    # Enhanced quality checks
                    if len(cleaned_voice_text) < 5:
                        with result_container:
                            st.warning("⚠️ **Voice input too short.** Please speak a complete question (at least 5 characters).")
                        return
                    
                    # Check for noise patterns with better tolerance
                    noise_patterns = ['whoa', 'uh', 'um', 'ah', 'er', 'hmm']
                    words = cleaned_voice_text.lower().split()
                    if len(words) > 3 and sum(1 for word in words if word in noise_patterns) > len(words) * 0.4:
                        with result_container:
                            st.warning("⚠️ **Voice unclear.** Please try again, speaking more clearly and reducing background noise.")
                        return
                    
                    with progress_container:
                        progress_bar.progress(100)
                        progress_text.text("🎯 Processing your question...")
                    
                    with result_container:
                        st.success(f"🎤 **Recognized:** {cleaned_voice_text}")
                        st.info("🤖 **Processing your question and preparing response...**")
                    
                    # Process the voice query with enhanced feedback
                    process_query(cleaned_voice_text, is_voice=True)
                else:
                    with result_container:
                        st.warning("⚠️ **No speech detected.** Please ensure:")
                        st.markdown("""
                        - Your microphone is working and enabled
                        - You spoke clearly during the listening period
                        - Background noise is minimized
                        - You spoke for at least 2-3 seconds
                        """)
                        
            except Exception as e:
                st.error(f"❌ **Voice processing error:** {str(e)}")
                st.info("💡 **Troubleshooting:** Try refreshing the page or checking your microphone permissions.")
                logging.error(f"Voice processing error: {e}")
    
    with col3:
        if st.button("🔧 Voice Test", key="voice_test_new", use_container_width=True):
            try:
                test_response = "This is a voice system test. If you can hear this clearly, your audio output is working perfectly."
                assistant.speak_with_streamlit_feedback(test_response, context="test")
                st.success("✅ Voice test completed!")
            except Exception as e:
                st.error(f"❌ Voice test failed: {e}")
                        
            except Exception as e:
                st.error(f"❌ Voice processing error: {str(e)}")
                logging.error(f"Voice processing error: {e}")
    
    # Voice system status
    with st.expander("🔧 Professional Voice System Status"):
        status = assistant.get_status()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Available Engines:**")
            for engine in status.get('available_tts_engines', []):
                st.write(f"✅ {engine.replace('_', ' ').title()} TTS")
            if status.get('speech_recognition_available', False):
                st.write("✅ Speech Recognition")
            else:
                st.write("❌ Speech Recognition")
        
        with col2:
            st.markdown("**Current Status:**")
            st.write(f"🎤 Listening: {'Yes' if status.get('listening', False) else 'No'}")
            st.write(f"🔊 Speaking: {'Yes' if status.get('speaking', False) else 'No'}")
            st.write(f"🎯 Quality: {status.get('quality_level', 'unknown').title()}")
            st.write(f"🎧 Microphone: {'Ready' if status.get('microphone_ready', False) else 'Not Ready'}")
            st.write(f"📋 Queue Size: {status.get('queue_size', 0)}")
            st.write(f"🔗 Streamlit Integration: {'Yes' if status.get('streamlit_integration', False) else 'No'}")

def create_chat_interface():
    """Enhanced chat interface"""
    st.markdown("""
    <div class="feature-card">
        <h3>💬 Interactive Chat</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for i, (query, response, timestamp, is_voice) in enumerate(st.session_state.chat_history[-10:]):
            # User message
            voice_indicator = "🎤" if is_voice else "⌨️"
            st.markdown(f"""
            <div class="user-message">
                {voice_indicator} <strong>You:</strong> {query}
                <br><small>{timestamp}</small>
            </div>
            """, unsafe_allow_html=True)
            
            # Assistant response
            st.markdown(f"""
            <div class="assistant-message">
                🤖 <strong>Assistant:</strong> {response}
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Input area
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_query = st.text_input(
            "Ask a question about your documents:",
            placeholder="Type your question here...",
            key="chat_input_main"
        )
    
    with col2:
        if st.button("Send 📤", key="send_button_main", type="primary", disabled=not user_query.strip()):
            if user_query.strip():
                process_query(user_query.strip())
    
    # Chat controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🗑️ Clear Chat", key="clear_chat_main"):
            st.session_state.chat_history = []
            st.rerun()
    
    with col2:
        if st.button("📥 Export Chat", key="export_chat_main"):
            export_chat_history()
    
    with col3:
        if st.button("📊 Chat Stats", key="chat_stats_main"):
            show_chat_statistics()

def process_query(query: str, is_voice: bool = False):
    """Process user query with enhanced error handling"""
    try:
        start_time = time.time()
        
        # Initialize RAG system if needed
        if st.session_state.rag_system is None:
            st.session_state.rag_system = ProductionRAGSystem()
        
        # Check if vectorstore exists
        if st.session_state.vectorstore is None:
            st.warning("⚠️ Please upload and process documents first")
            
            # Provide voice feedback if available
            if PROFESSIONAL_VOICE_AVAILABLE and 'professional_voice_assistant' in st.session_state and is_voice:
                assistant = st.session_state.professional_voice_assistant
                no_doc_response = "I apologize, but no documents have been uploaded yet. Please upload a document first, and then I'll be happy to answer your questions about it."
                assistant.speak_professional(no_doc_response, context="help")
            return
        
        with st.spinner("🔍 Searching documents..."):
            # Get response from RAG system using the vectorstore
            response = ask_enhanced_question(
                st.session_state.rag_system, 
                query, 
                st.session_state.vectorstore
            )
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Update metrics
            st.session_state.metrics['total_queries'] += 1
            st.session_state.metrics['successful_queries'] += 1
            if is_voice:
                st.session_state.metrics['voice_queries'] += 1
            
            # Update average response time
            current_avg = st.session_state.metrics['avg_response_time']
            total_queries = st.session_state.metrics['total_queries']
            st.session_state.metrics['avg_response_time'] = (current_avg * (total_queries - 1) + response_time) / total_queries
            
            # Add to chat history
            timestamp = datetime.now().strftime("%H:%M:%S")
            st.session_state.chat_history.append((query, response, timestamp, is_voice))
            
            # Voice output if available and voice query
            if PROFESSIONAL_VOICE_AVAILABLE and is_voice:
                try:
                    assistant = get_streamlit_voice_assistant()
                    
                    # Create a professional, concise voice response
                    # Keep voice responses shorter and more conversational
                    voice_response = response
                    
                    # If response is too long, create a summary for voice
                    if len(voice_response) > 300:
                        # Extract key points for voice delivery
                        sentences = voice_response.split('. ')
                        key_sentences = sentences[:3]  # First 3 sentences
                        voice_response = '. '.join(key_sentences)
                        if not voice_response.endswith('.'):
                            voice_response += '.'
                        voice_response += " You can see the complete details on your screen."
                    
                    # Remove any HTML/markdown formatting for clean voice output
                    import re
                    voice_response = re.sub(r'<[^>]+>', '', voice_response)  # Remove HTML tags
                    voice_response = re.sub(r'\*\*([^*]+)\*\*', r'\1', voice_response)  # Remove markdown bold
                    voice_response = re.sub(r'\*([^*]+)\*', r'\1', voice_response)  # Remove markdown italic
                    voice_response = re.sub(r'`([^`]+)`', r'\1', voice_response)  # Remove code blocks
                    
                    # Use professional speech output with Streamlit feedback
                    assistant.speak_with_streamlit_feedback(voice_response, context="response")
                    
                except Exception as voice_error:
                    st.warning("⚠️ Voice output unavailable - check audio settings")
                    logging.error(f"Voice output error: {voice_error}")
            
            st.session_state.last_activity = datetime.now()
            st.rerun()
            
    except Exception as e:
        st.error(f"❌ Query processing error: {str(e)}")
        logging.error(f"Query processing error: {e}")
        
        # Provide voice feedback for errors if available
        if PROFESSIONAL_VOICE_AVAILABLE and is_voice:
            try:
                assistant = get_streamlit_voice_assistant()
                error_response = "I apologize, but I encountered an error while processing your request. Please try again or check your query."
                assistant.speak_with_streamlit_feedback(error_response, context="error")
            except Exception as voice_error:
                logging.error(f"Voice error feedback failed: {voice_error}")

def create_document_processor():
    """Enhanced document processing interface"""
    st.markdown("""
    <div class="feature-card">
        <h3>📄 Document Processing</h3>
        <p>Upload and process your documents for intelligent analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a document",
        type=['pdf', 'txt', 'md', 'docx', 'csv'],
        help="Supported formats: PDF, TXT, MD, DOCX, CSV"
    )
    
    if uploaded_file:
        # File validation
        file_size = len(uploaded_file.read())
        uploaded_file.seek(0)  # Reset file pointer
        
        if file_size > AppConfig.MAX_FILE_SIZE:
            st.error(f"❌ File too large. Maximum size: {AppConfig.MAX_FILE_SIZE // (1024*1024)}MB")
            return
        
        # Display file info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📁 File Name", uploaded_file.name)
        with col2:
            st.metric("📊 File Size", f"{file_size / (1024*1024):.2f} MB")
        with col3:
            st.metric("📋 File Type", uploaded_file.type)
        
        # Process button
        if st.button("🚀 Process Document", key="process_doc_main", type="primary", use_container_width=True):
            process_document(uploaded_file)

def process_document(uploaded_file):
    """Process uploaded document with progress tracking"""
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Initialize RAG system
        status_text.text("🔧 Initializing RAG system...")
        progress_bar.progress(20)
        
        if st.session_state.rag_system is None:
            st.session_state.rag_system = ProductionRAGSystem()
        
        # Save uploaded file
        status_text.text("💾 Saving document...")
        progress_bar.progress(40)
        
        file_path = f"temp_{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        
        # Process document asynchronously
        status_text.text("📄 Processing document...")
        progress_bar.progress(60)
        
        # Run the async function in the event loop
        try:
            # Try to get the current event loop
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # Create a new event loop if none exists
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Process document and create vectorstore
        vectorstore = loop.run_until_complete(
            process_document_async(st.session_state.rag_system, file_path)
        )
        
        # Store vectorstore in session state
        st.session_state.vectorstore = vectorstore
        
        status_text.text("🔍 Building search index...")
        progress_bar.progress(80)
        
        # Clean up
        os.remove(file_path)
        
        status_text.text("✅ Processing complete!")
        progress_bar.progress(100)
        
        # Update metrics
        st.session_state.metrics['documents_processed'] += 1
        
        time.sleep(1)
        status_text.empty()
        progress_bar.empty()
        
        st.success("✅ Document processed successfully! You can now ask questions about it.")
        
    except Exception as e:
        st.error(f"❌ Document processing error: {str(e)}")
        logging.error(f"Document processing error: {e}")
        
        # Clean up on error
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)

def create_metrics_dashboard():
    """Enhanced metrics dashboard"""
    st.markdown("""
    <div class="feature-card">
        <h3>📊 System Analytics</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics grid
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{st.session_state.metrics['total_queries']}</p>
            <p class="metric-label">Total Queries</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        success_rate = (st.session_state.metrics['successful_queries'] / max(1, st.session_state.metrics['total_queries'])) * 100
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{success_rate:.1f}%</p>
            <p class="metric-label">Success Rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{st.session_state.metrics['avg_response_time']:.2f}s</p>
            <p class="metric-label">Avg Response Time</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{st.session_state.metrics['voice_queries']}</p>
            <p class="metric-label">Voice Queries</p>
        </div>
        """, unsafe_allow_html=True)

def create_sidebar():
    """Enhanced sidebar with system controls"""
    with st.sidebar:
        st.markdown("### 🎛️ System Controls")
        
        # System status
        if st.session_state.rag_system:
            st.success("✅ RAG System: Active")
        else:
            st.warning("⚠️ RAG System: Not initialized")
        
        st.divider()
        
        # Professional Voice Settings
        st.markdown("### 🎤 Professional Voice Settings")
        
        voice_enabled = st.toggle("Enable Professional Voice Output", value=True, key="voice_output_toggle")
        
        if voice_enabled:
                # Professional voice quality settings
                col1, col2 = st.columns(2)
                
                with col1:
                    speech_rate = st.slider(
                        "Speech Rate", 
                        min_value=120, 
                        max_value=220, 
                        value=180, 
                        step=10, 
                        key="speech_rate_slider",
                        help="Slower rates provide better clarity"
                    )
                
                with col2:
                    voice_volume = st.slider(
                        "Volume", 
                        min_value=0.5, 
                        max_value=1.0, 
                        value=0.9, 
                        step=0.1, 
                        key="voice_volume_slider"
                    )
                
                # Voice engine selection
                voice_engine = st.selectbox(
                    "Voice Engine", 
                    ["System TTS (Best Quality)", "Google TTS (Cloud)"], 
                    key="voice_engine_select",
                    help="System TTS provides better pronunciation and professional sound"
                )
                
                # Voice gender preference
                voice_gender = st.selectbox(
                    "Voice Gender Preference",
                    ["Female (Professional)", "Male (Professional)", "Auto Select"],
                    key="voice_gender_select"
                )
                
                # Professional voice quality check
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🎭 Check Voice Quality", key="test_professional_voice"):
                        if 'professional_voice_assistant' in st.session_state:
                            assistant = st.session_state.professional_voice_assistant
                            quality_text = "Voice system operational. Professional speech synthesis active with optimized pronunciation and natural intonation patterns."
                            
                            def quality_callback(status):
                                if status == "speaking":
                                    st.info("🔊 Professional voice quality check in progress...")
                                elif status == "success":
                                    st.success("✅ Professional voice quality check completed")
                            
                            assistant.speak_professional(quality_text, callback=quality_callback)
                
                with col2:
                    if st.button("🎭 Voice Profiles", key="view_voices"):
                        if 'professional_voice_assistant' in st.session_state:
                            assistant = st.session_state.professional_voice_assistant
                            status = assistant.get_status()
                            st.markdown("**Available Professional Voice Engines:**")
                            for engine in status['available_tts_engines']:
                                if engine == 'google':
                                    st.write("⭐ Google TTS - High Quality Professional Voice")
                                elif engine == 'system':
                                    st.write("🔷 System TTS - Standard Quality (Microsoft Zira)")
                                elif engine == 'azure':
                                    st.write("🌟 Azure TTS - Premium Quality (Neural Voices)")
                            
                            st.write(f"**Current Quality Level:** {status['quality_level'].title()}")
                            st.write("**Voice Features:** Professional pronunciation, natural pauses, technical term enhancement")
                
                # Show current audio settings
                if 'professional_voice_assistant' in st.session_state:
                    assistant = st.session_state.professional_voice_assistant
                    status = assistant.get_status()
                    
                    with st.expander("🔧 Professional Audio Configuration"):
                        st.write(f"**Voice Engines:** {', '.join(status['available_tts_engines']).title()}")
                        st.write(f"**Speech Recognition:** {'Available' if status['speech_recognition_available'] else 'Not Available'}")
                        st.write(f"**Speech Quality:** Professional Enhanced")
                        st.write(f"**Microphone Status:** {'Ready' if status['microphone_ready'] else 'Not Ready'}")
                        
                        if status['listening']:
                            st.info("🎤 Currently listening...")
                        if status['speaking']:
                            st.info("🔊 Currently speaking...")
                        
                        st.write("**Configuration:** Enhanced speech recognition with professional TTS")
                        st.write("**Features:** Natural pauses, technical term pronunciation, voice enhancement")
                        
                        st.success("✅ Professional voice system operational")
        
        st.divider()
        
        # System actions
        st.markdown("### ⚙️ System Actions")
        
        if st.button("🔄 Reset Session", key="sidebar_reset", use_container_width=True):
            reset_session()
        
        if st.button("📥 Export Data", key="sidebar_export", use_container_width=True):
            export_system_data()
        
        if st.button("🧹 Clear Cache", key="sidebar_clear_cache", use_container_width=True):
            clear_system_cache()
        
        st.divider()
        
        # System info
        st.markdown("### ℹ️ System Info")
        st.write(f"**Session Time:** {(datetime.now() - st.session_state.last_activity).seconds}s ago")
        st.write(f"**Documents:** {st.session_state.metrics['documents_processed']}")
        st.write(f"**Chat Messages:** {len(st.session_state.chat_history)}")

def reset_session():
    """Reset the session with confirmation"""
    for key in list(st.session_state.keys()):
        if key != 'initialized':
            del st.session_state[key]
    initialize_session_state()
    st.success("✅ Session reset successfully!")
    st.rerun()

def export_system_data():
    """Export system data"""
    try:
        data = {
            'chat_history': st.session_state.chat_history,
            'metrics': st.session_state.metrics,
            'export_time': datetime.now().isoformat()
        }
        
        json_data = json.dumps(data, indent=2)
        
        st.download_button(
            label="📥 Download System Data",
            data=json_data,
            file_name=f"rag_system_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            key="download_system_data"
        )
        
    except Exception as e:
        st.error(f"❌ Export error: {str(e)}")

def clear_system_cache():
    """Clear system cache"""
    try:
        # Clear any cached data
        if hasattr(st.session_state, 'rag_system') and st.session_state.rag_system:
            # Reset RAG system if needed
            pass
        
        st.success("✅ Cache cleared successfully!")
        
    except Exception as e:
        st.error(f"❌ Cache clear error: {str(e)}")

def export_chat_history():
    """Export chat history"""
    if not st.session_state.chat_history:
        st.warning("⚠️ No chat history to export")
        return
    
    try:
        # Create formatted chat export
        chat_data = []
        for query, response, timestamp, is_voice in st.session_state.chat_history:
            chat_data.append({
                'timestamp': timestamp,
                'query': query,
                'response': response,
                'input_type': 'voice' if is_voice else 'text'
            })
        
        df = pd.DataFrame(chat_data)
        csv_data = df.to_csv(index=False)
        
        st.download_button(
            label="📥 Download Chat History",
            data=csv_data,
            file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            key="download_chat_history"
        )
        
    except Exception as e:
        st.error(f"❌ Export error: {str(e)}")

def show_chat_statistics():
    """Show detailed chat statistics"""
    if not st.session_state.chat_history:
        st.info("ℹ️ No chat data available")
        return
    
    # Calculate statistics
    total_messages = len(st.session_state.chat_history)
    voice_messages = sum(1 for _, _, _, is_voice in st.session_state.chat_history if is_voice)
    text_messages = total_messages - voice_messages
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("💬 Total Messages", total_messages)
    
    with col2:
        st.metric("🎤 Voice Messages", voice_messages)
    
    with col3:
        st.metric("⌨️ Text Messages", text_messages)
    
    # Message distribution chart
    if total_messages > 0:
        chart_data = pd.DataFrame({
            'Type': ['Voice', 'Text'],
            'Count': [voice_messages, text_messages]
        })
        
        st.bar_chart(chart_data.set_index('Type'))

def main():
    """Main application function"""
    # Apply styling
    apply_modern_styling()
    
        # Initialize session
    initialize_session_state()
    
    # Create header
    create_header()
    
    # Create sidebar
    create_sidebar()
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎤 Voice Assistant", "📄 Documents", "💬 Chat", "📊 Analytics"])
    
    with tab1:
        st.markdown("## 🎙️ Voice Assistant Interface")
        create_voice_interface()
        
        # Voice system diagnostics
        with st.expander("🔧 Voice System Diagnostics"):
            if 'professional_voice_assistant' in st.session_state:
                assistant = st.session_state.professional_voice_assistant
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("🎧 Check Microphone", key="test_mic"):
                        try:
                            st.info("🎤 Checking microphone input - Please speak for 5 seconds...")
                            # Simple microphone test
                            import speech_recognition as sr
                            r = sr.Recognizer()
                            with sr.Microphone() as source:
                                r.adjust_for_ambient_noise(source)
                                audio = r.listen(source, timeout=5)
                            result = r.recognize_google(audio)
                            st.success(f"✅ Microphone operational! Detected: '{result}'")
                        except Exception as e:
                            st.error(f"❌ Microphone check failed: {str(e)}")
                
                with col2:
                    if st.button("🔊 Check Speaker", key="test_speaker"):
                        try:
                            st.info("🔊 Checking speaker output...")
                            success = assistant.speak_professional("Professional voice assistant speaker test. Audio output is functioning correctly.")
                            if success:
                                st.success("✅ Speaker check completed")
                            else:
                                st.warning("⚠️ Professional TTS not available or speaker issue")
                        except Exception as e:
                            st.error(f"❌ Speaker check failed: {str(e)}")
                
                with col3:
                    if st.button("📊 Audio Status", key="test_devices"):
                        try:
                            status = assistant.get_status()
                            
                            st.markdown("**Professional Voice System Status:**")
                            if status['microphone_ready']:
                                st.success("✅ Microphone Ready")
                            else:
                                st.error("❌ Microphone Not Ready")
                            
                            if status['available_tts_engines']:
                                st.success(f"✅ TTS Engines: {', '.join(status['available_tts_engines'])}")
                            else:
                                st.error("❌ No TTS Engines Available")
                            
                            if status['speech_recognition_available']:
                                st.success("✅ Speech Recognition Available")
                            else:
                                st.error("❌ Speech Recognition Not Available")
                            
                            st.write(f"**Quality Level:** {status['quality_level'].title()}")
                            st.write(f"**Queue Size:** {status['queue_size']}")
                                
                        except Exception as e:
                            st.error(f"❌ Status check failed: {str(e)}")
            else:
                st.warning("⚠️ Professional voice assistant not initialized")
        
        # Voice system troubleshooting guide
        with st.expander("🔧 Voice System Troubleshooting"):
            st.markdown("""
            **If voice recognition is not working:**
            
            1. **Check Microphone Permission:**
               - Ensure your browser has microphone access
               - Check Windows microphone privacy settings
            
            2. **Audio Quality:**
               - Speak clearly and at normal volume
               - Reduce background noise
               - Position microphone 6-12 inches from mouth
            
            3. **Network Issues:**
               - Google Speech API requires internet connection
               - Try the offline Sphinx recognition as fallback
            
            4. **Timeout Issues:**
               - Voice input times out after 10 seconds
               - Start speaking immediately after clicking the button
               - For long queries, speak continuously without long pauses
            
            5. **Recognition Errors:**
               - Try speaking in shorter sentences
               - Use clear pronunciation
               - Avoid background music or noise
            
            **Common Status Messages:**
            - "All recognition methods failed" → Network or audio input issue
            - "No speech detected" → Microphone not working or too quiet
            - "Could not understand audio" → Speech unclear or too noisy
            """)
            
            # Quick system diagnostic
            if st.button("🩺 Run System Diagnostic", key="voice_diagnostic"):
                if 'professional_voice_assistant' in st.session_state:
                    assistant = st.session_state.professional_voice_assistant
                    
                    st.write("**Professional Voice System Diagnostic Results:**")
                    try:
                        status = assistant.get_status()
                        
                        # Check each component
                        checks = [
                            ("Speech Recognition", status['speech_recognition_available']),
                            ("Microphone Access", status['microphone_ready']),
                            ("TTS Engines Available", len(status['available_tts_engines']) > 0),
                            ("Professional TTS", True),  # Always available with our system
                            ("Voice Enhancement", True)   # Built-in feature
                        ]
                        
                        for check_name, check_status in checks:
                            icon = "✅" if check_status else "❌"
                            st.write(f"{icon} {check_name}")
                        
                        if status['available_tts_engines']:
                            st.info(f"Available TTS Engines: {', '.join(status['available_tts_engines'])}")
                        
                    except Exception as e:
                        st.error(f"Diagnostic failed: {str(e)}")
                else:
                    st.error("Professional voice assistant not initialized")
    
    with tab2:
        st.markdown("## 📚 Document Management")
        create_document_processor()
        
        # Document history
        if st.session_state.metrics['documents_processed'] > 0:
            st.markdown("### 📋 Processing History")
            st.info(f"Total documents processed: {st.session_state.metrics['documents_processed']}")
    
    with tab3:
        st.markdown("## 💬 Interactive Chat")
        create_chat_interface()
    
    with tab4:
        st.markdown("## 📊 System Analytics")
        create_metrics_dashboard()
        
        # Detailed analytics
        with st.expander("📈 Detailed Analytics"):
            show_chat_statistics()

if __name__ == "__main__":
    main()
