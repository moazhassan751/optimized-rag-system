"""
Streamlit-specific Voice Integration
Handles voice assistant integration with proper Streamlit context
"""

import streamlit as st
import logging
import threading
import time
from typing import Optional, Callable, Dict, Any

try:
    from professional_voice_assistant import create_professional_voice_assistant, TTSQuality
    PROFESSIONAL_VOICE_AVAILABLE = True
except ImportError:
    PROFESSIONAL_VOICE_AVAILABLE = False

class StreamlitVoiceAssistant:
    """Voice assistant optimized for Streamlit integration"""
    
    def __init__(self, quality: TTSQuality = TTSQuality.MEDIUM):
        self.logger = logging.getLogger(__name__)
        self.quality = quality
        self._assistant = None
        self._initialized = False
        
        # Streamlit-specific state
        self._streamlit_context = True
        
        # Thread safety for Streamlit
        self._lock = threading.Lock()
        
        # Initialize assistant
        self._initialize_assistant()
    
    def _initialize_assistant(self):
        """Initialize voice assistant with Streamlit compatibility"""
        if not PROFESSIONAL_VOICE_AVAILABLE:
            self.logger.error("Professional voice assistant not available")
            return False
        
        try:
            with self._lock:
                if self._assistant is None:
                    self._assistant = create_professional_voice_assistant(self.quality)
                    self._initialized = True
                    self.logger.info("[SUCCESS] Streamlit Voice Assistant initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Streamlit voice assistant: {e}")
            return False
    
    def speak_with_streamlit_feedback(self, text: str, context: str = "response") -> bool:
        """Speak text with Streamlit UI feedback"""
        if not self._initialized or not self._assistant:
            return False
        
        try:
            # Create status placeholder
            status_placeholder = st.empty()
            
            def streamlit_callback(status):
                try:
                    if hasattr(st, '_get_script_run_ctx') and st._get_script_run_ctx():
                        # We have Streamlit context
                        if status == "speaking":
                            status_placeholder.info("[SPEAKER] Speaking response...")
                        elif status == "success":
                            status_placeholder.empty()
                        elif status == "error":
                            status_placeholder.error("[ERROR] Speech synthesis error")
                except Exception as e:
                    # Ignore Streamlit context errors
                    self.logger.debug(f"Streamlit callback warning: {e}")
            
            # Use professional voice assistant
            success = self._assistant.speak_professional(text, callback=streamlit_callback, context=context)
            
            # Brief delay to show status
            if success:
                time.sleep(0.5)
                status_placeholder.empty()
            
            return success
            
        except Exception as e:
            self.logger.error(f"Streamlit TTS error: {e}")
            return False
    
    def listen_with_streamlit_feedback(self, timeout: float = 12.0) -> Optional[str]:
        """Listen for speech with Streamlit UI feedback"""
        if not self._initialized or not self._assistant:
            return None
        
        try:
            # Create status placeholder
            status_placeholder = st.empty()
            result_placeholder = st.empty()
            
            def streamlit_callback(status):
                try:
                    if hasattr(st, '_get_script_run_ctx') and st._get_script_run_ctx():
                        if status == "listening":
                            status_placeholder.markdown("""
                            <div style="background: linear-gradient(135deg, #00b894 0%, #00a085 100%); 
                                        color: white; padding: 1rem; border-radius: 10px; text-align: center;
                                        animation: pulse 2s infinite;">
                                [LISTEN] Listening... Speak clearly now
                            </div>
                            """, unsafe_allow_html=True)
                        elif status == "recognizing":
                            status_placeholder.markdown("""
                            <div style="background: linear-gradient(135deg, #fdcb6e 0%, #e17055 100%); 
                                        color: white; padding: 1rem; border-radius: 10px; text-align: center;">
                                🔄 Processing your voice input...
                            </div>
                            """, unsafe_allow_html=True)
                        elif status == "success":
                            status_placeholder.empty()
                        elif status == "error":
                            status_placeholder.error("[ERROR] Could not understand speech. Please try again.")
                        elif status == "timeout":
                            status_placeholder.warning("⏰ No speech detected. Please try again.")
                except Exception as e:
                    self.logger.debug(f"Streamlit callback warning: {e}")
            
            # Listen for speech
            recognized_text = self._assistant.listen_for_speech(timeout=timeout, callback=streamlit_callback)
            
            # Show result
            if recognized_text:
                result_placeholder.success(f"[MIC] Recognized: {recognized_text}")
                time.sleep(1)
                result_placeholder.empty()
            
            # Clean up status
            status_placeholder.empty()
            
            return recognized_text
            
        except Exception as e:
            self.logger.error(f"Streamlit speech recognition error: {e}")
            return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get voice assistant status"""
        if not self._initialized or not self._assistant:
            return {
                'initialized': False,
                'error': 'Assistant not initialized'
            }
        
        status = self._assistant.get_status()
        status['streamlit_integration'] = True
        return status
    
    def shutdown(self):
        """Shutdown voice assistant"""
        if self._assistant:
            self._assistant.shutdown()
            self._assistant = None
            self._initialized = False

# Streamlit session state management
def get_streamlit_voice_assistant(quality: TTSQuality = TTSQuality.MEDIUM) -> StreamlitVoiceAssistant:
    """Get or create Streamlit voice assistant from session state"""
    if 'streamlit_voice_assistant' not in st.session_state:
        st.session_state.streamlit_voice_assistant = StreamlitVoiceAssistant(quality)
    return st.session_state.streamlit_voice_assistant

def cleanup_streamlit_voice_assistant():
    """Cleanup voice assistant from session state"""
    if 'streamlit_voice_assistant' in st.session_state:
        st.session_state.streamlit_voice_assistant.shutdown()
        del st.session_state.streamlit_voice_assistant

# Test function for standalone testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("[TARGET] Testing Streamlit Voice Integration...")
    
    # Test without Streamlit context (should handle gracefully)
    assistant = StreamlitVoiceAssistant(TTSQuality.MEDIUM)
    
    print("Status:", assistant.get_status())
    
    # Test TTS (will work without Streamlit UI feedback)
    print("Testing TTS...")
    success = assistant._assistant.speak_professional(
        "This is a test of the Streamlit integrated voice assistant.", 
        context="response"
    )
    print(f"TTS Success: {success}")
    
    time.sleep(3)
    
    assistant.shutdown()
    print("[SUCCESS] Streamlit Voice Integration test complete!")
