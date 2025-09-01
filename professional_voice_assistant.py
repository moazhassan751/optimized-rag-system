"""
[TARGET] Professional Voice Assistant - Production Ready
Enhanced with superior speech recognition, natural TTS, and professional responses
"""

import logging
import time
import threading
import tempfile
import os
import queue
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum
import re

# Speech recognition imports
try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False

# TTS imports
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

try:
    from gtts import gTTS
    import pygame
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

try:
    import azure.cognitiveservices.speech as speechsdk
    AZURE_TTS_AVAILABLE = True
except ImportError:
    AZURE_TTS_AVAILABLE = False

class TTSQuality(Enum):
    """TTS Quality levels"""
    HIGH = "high"        # Azure TTS (best quality)
    MEDIUM = "medium"    # Google TTS (good quality)
    BASIC = "basic"      # System TTS (basic quality)

class VoicePersonality:
    """Professional voice personality configuration"""
    def __init__(self):
        self.name = "Professional Assistant"
        self.speaking_style = "professional"
        self.rate = 175  # Words per minute (optimal for clarity)
        self.volume = 0.9
        self.pitch = 0  # Neutral pitch
        self.emphasis = True
        self.natural_pauses = True
        self.pronunciation_enhancer = True

class ProfessionalVoiceAssistant:
    """Professional Voice Assistant with superior speech capabilities"""
    
    def __init__(self, quality: TTSQuality = TTSQuality.MEDIUM):
        self.logger = logging.getLogger(__name__)
        self.quality = quality
        self.personality = VoicePersonality()
        
        # Initialize components
        self.recognizer = None
        self.microphone = None
        self.tts_engines = {}
        self.is_listening = False
        self.is_speaking = False
        
        # Speech processing queue
        self.speech_queue = queue.Queue()
        self.worker_thread = None
        self.stop_event = threading.Event()
        
        # Professional responses
        self._init_professional_responses()
        
        # Initialize engines
        self._initialize_speech_recognition()
        self._initialize_tts_engines()
        self._start_speech_worker()
        
        self.logger.info("[TARGET] Professional Voice Assistant initialized")
    
    def _init_professional_responses(self):
        """Initialize professional response templates"""
        self.responses = {
            'greeting': [
                "Good day! I'm your professional assistant, ready to help you with any questions.",
                "Hello! I'm here to provide you with expert assistance and information.",
                "Welcome! I'm your dedicated assistant, prepared to address all your inquiries."
            ],
            'listening': [
                "I'm listening carefully. Please go ahead with your question.",
                "I'm ready to hear your question. Please speak clearly.",
                "I'm attentively listening. What would you like to know?"
            ],
            'processing': [
                "I'm processing your request and searching for the most relevant information.",
                "Please allow me a moment to analyze your question and provide a comprehensive response.",
                "I'm carefully reviewing the available information to give you the best answer."
            ],
            'error': [
                "I apologize, but I didn't catch that clearly. Could you please repeat your question?",
                "I'm sorry, but there seems to have been an audio issue. Please try speaking again.",
                "Pardon me, but I need you to repeat that. Please speak clearly and at a moderate pace."
            ],
            'no_input': [
                "I didn't detect any speech. Please ensure your microphone is working and speak clearly.",
                "No audio input was detected. Please check your microphone and try again.",
                "I'm not receiving any audio. Please verify your microphone settings and speak up."
            ]
        }
    
    def _fix_common_misrecognitions(self, text: str) -> str:
        """Fix common speech recognition errors"""
        if not text:
            return text
        
        # Common misrecognitions mapping
        corrections = {
            # Technical terms
            'genetic AI': 'agentic AI',
            'genetic artificial intelligence': 'agentic artificial intelligence',
            'agent tick': 'agentic',
            'agent tic': 'agentic',
            
            # AI/ML terms that are often misheard
            'machine learning': 'machine learning',
            'deep learning': 'deep learning',
            'neural network': 'neural network',
            'artificial intelligence': 'artificial intelligence',
            'natural language processing': 'natural language processing',
            
            # Common word confusions in context
            'algorithm': 'algorithm',
            'algorithms': 'algorithms',
            'data science': 'data science',
            'analytics': 'analytics',
            'automation': 'automation',
            'optimization': 'optimization'
        }
        
        corrected_text = text
        text_lower = text.lower()
        
        # Apply corrections with context awareness
        for wrong, correct in corrections.items():
            if wrong.lower() in text_lower:
                # Use case-insensitive replacement but preserve original case style
                corrected_text = self._case_aware_replace(corrected_text, wrong, correct)
        
        return corrected_text
    
    def _case_aware_replace(self, text: str, old: str, new: str) -> str:
        """Replace text while preserving case style"""
        import re
        
        def replace_func(match):
            matched_text = match.group(0)
            if matched_text.isupper():
                return new.upper()
            elif matched_text.istitle():
                return new.title()
            elif matched_text.islower():
                return new.lower()
            else:
                return new
        
        pattern = re.compile(re.escape(old), re.IGNORECASE)
        return pattern.sub(replace_func, text)
    
    def _initialize_speech_recognition(self):
        """Initialize advanced speech recognition"""
        if not SPEECH_RECOGNITION_AVAILABLE:
            self.logger.error("Speech recognition not available")
            return False
        
        try:
            self.recognizer = sr.Recognizer()
            
            # Enhanced recognition settings
            self.recognizer.energy_threshold = 400  # Higher threshold for better noise rejection
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.pause_threshold = 1.0  # Longer pause for complete thoughts
            self.recognizer.phrase_threshold = 0.3  # Minimum audio length
            self.recognizer.non_speaking_duration = 0.8  # Duration to wait for silence
            
            # Initialize microphone with optimal settings
            self.microphone = sr.Microphone(sample_rate=16000, chunk_size=1024)
            
            # Calibrate for ambient noise
            with self.microphone as source:
                self.logger.info("[LISTEN] Calibrating microphone for optimal performance...")
                self.recognizer.adjust_for_ambient_noise(source, duration=2.5)
                self.logger.info(f"[SUCCESS] Microphone calibrated. Energy threshold: {self.recognizer.energy_threshold}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize speech recognition: {e}")
            return False
    
    def _initialize_tts_engines(self):
        """Initialize multiple TTS engines for quality fallback"""
        
        # 1. Try Azure TTS (highest quality)
        if AZURE_TTS_AVAILABLE and self.quality == TTSQuality.HIGH:
            try:
                # Note: Requires Azure API key
                self.tts_engines['azure'] = self._init_azure_tts()
                self.logger.info("[SUCCESS] Azure TTS initialized (Premium Quality)")
            except Exception as e:
                self.logger.warning(f"Azure TTS not available: {e}")
        
        # 2. Google TTS (good quality)
        if GTTS_AVAILABLE:
            try:
                if not pygame.mixer.get_init():
                    pygame.mixer.quit()
                    pygame.mixer.pre_init(frequency=22050, size=-16, channels=2, buffer=1024)
                    pygame.mixer.init()
                self.tts_engines['google'] = True
                self.logger.info("[SUCCESS] Google TTS initialized (High Quality)")
            except Exception as e:
                self.logger.warning(f"Google TTS initialization failed: {e}")
        
        # 3. System TTS (fallback)
        if PYTTSX3_AVAILABLE:
            try:
                engine = pyttsx3.init()
                
                # Configure for professional speech
                engine.setProperty('rate', self.personality.rate)
                engine.setProperty('volume', self.personality.volume)
                
                # Select best voice
                voices = engine.getProperty('voices')
                if voices:
                    # Prefer female professional voices
                    professional_voices = [v for v in voices if any(term in v.name.lower() 
                                         for term in ['zira', 'hazel', 'susan', 'samantha'])]
                    if professional_voices:
                        engine.setProperty('voice', professional_voices[0].id)
                        self.logger.info(f"Selected voice: {professional_voices[0].name}")
                    else:
                        # Fallback to first available voice
                        engine.setProperty('voice', voices[0].id)
                
                self.tts_engines['system'] = engine
                self.logger.info("[SUCCESS] System TTS initialized (Standard Quality)")
                
            except Exception as e:
                self.logger.warning(f"System TTS initialization failed: {e}")
    
    def _start_speech_worker(self):
        """Start background speech processing worker"""
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.worker_thread = threading.Thread(target=self._speech_worker, daemon=True)
            self.worker_thread.start()
    
    def _speech_worker(self):
        """Background worker for speech processing"""
        while not self.stop_event.is_set():
            try:
                task = self.speech_queue.get(timeout=1.0)
                if task is None:
                    break
                
                task_type, data, callback = task
                
                if task_type == "speak":
                    self._execute_speech(data, callback)
                elif task_type == "listen":
                    self._execute_listening(data, callback)
                
                self.speech_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Speech worker error: {e}")
    
    def enhance_text_for_speech(self, text: str, context: str = "response") -> str:
        """Enhance text for professional speech delivery"""
        if not text or not text.strip():
            return ""
        
        # Clean text
        enhanced = re.sub(r'\s+', ' ', text.strip())
        
        # Remove problematic characters
        enhanced = enhanced.replace('*', '').replace('#', 'number ')
        enhanced = enhanced.replace('@', ' at ').replace('&', ' and ')
        enhanced = enhanced.replace('>', ' greater than ').replace('<', ' less than ')
        
        # Fix markdown
        enhanced = re.sub(r'\*\*(.*?)\*\*', r'\1', enhanced)
        enhanced = re.sub(r'\*(.*?)\*', r'\1', enhanced)
        
        # Improve pronunciation of technical terms
        pronunciation_fixes = {
            'AI': 'A I',
            'API': 'A P I', 
            'URL': 'U R L',
            'PDF': 'P D F',
            'FAQ': 'F A Q',
            'UI': 'U I',
            'SQL': 'S Q L',
            'JSON': 'J SON',
            'HTTP': 'H T T P',
            'vs': 'versus',
            'etc': 'etcetera',
            '&': 'and'
        }
        
        for original, replacement in pronunciation_fixes.items():
            enhanced = re.sub(r'\b' + re.escape(original) + r'\b', replacement, enhanced, flags=re.IGNORECASE)
        
        # Add natural pauses
        if self.personality.natural_pauses:
            # Use simple punctuation for natural pauses instead of SSML
            enhanced = enhanced.replace('. ', '. ')  # Keep normal sentence breaks
            enhanced = enhanced.replace(', ', ', ')  # Keep normal comma pauses  
            enhanced = enhanced.replace(': ', ': ')  # Keep normal colon pauses
            enhanced = enhanced.replace('; ', '; ')  # Keep normal semicolon pauses
        
        # Add professional context
        if context == "response" and not any(enhanced.lower().startswith(starter) 
                                           for starter in ['certainly', 'of course', 'let me', 'here is', 'based on']):
            starters = [
                "Based on the information available, ",
                "According to the document, ",
                "Here's what I found: ",
                "Let me provide you with the details: "
            ]
            enhanced = starters[hash(enhanced) % len(starters)] + enhanced
        
        return enhanced
    
    def speak_professional(self, text: str, callback: Optional[Callable] = None, context: str = "response") -> bool:
        """Speak text with professional quality and delivery"""
        if not text:
            return False
        
        enhanced_text = self.enhance_text_for_speech(text, context)
        
        try:
            self.speech_queue.put(("speak", enhanced_text, callback))
            return True
        except Exception as e:
            self.logger.error(f"Failed to queue speech: {e}")
            if callback:
                callback("error")
            return False
    
    def _execute_speech(self, text: str, callback: Optional[Callable]):
        """Execute speech with best available engine"""
        try:
            self.is_speaking = True
            if callback:
                callback("speaking")
            
            success = False
            
            # Try Google TTS first (best quality available)
            if 'google' in self.tts_engines:
                success = self._speak_google_tts(text)
            
            # Fallback to system TTS
            if not success and 'system' in self.tts_engines:
                success = self._speak_system_tts(text)
            
            if callback:
                callback("success" if success else "error")
                
        except Exception as e:
            self.logger.error(f"Speech execution failed: {e}")
            if callback:
                callback("error")
        finally:
            self.is_speaking = False
    
    def _speak_google_tts(self, text: str) -> bool:
        """High-quality Google TTS"""
        try:
            # Create TTS with optimal settings
            tts = gTTS(
                text=text,
                lang='en',
                slow=False,
                tld='com'  # Use .com for American English
            )
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
                temp_path = temp_file.name
                tts.save(temp_path)
            
            try:
                # Play with pygame
                pygame.mixer.music.load(temp_path)
                pygame.mixer.music.play()
                
                # Wait for completion
                while pygame.mixer.music.get_busy():
                    pygame.time.wait(100)
                    if self.stop_event.is_set():
                        pygame.mixer.music.stop()
                        break
                
                return True
                
            finally:
                try:
                    pygame.mixer.music.stop()
                    pygame.mixer.music.unload()
                    os.unlink(temp_path)
                except:
                    pass
            
        except Exception as e:
            self.logger.error(f"Google TTS failed: {e}")
            return False
    
    def _speak_system_tts(self, text: str) -> bool:
        """System TTS with professional settings"""
        try:
            engine = self.tts_engines['system']
            engine.stop()  # Clear any pending speech
            engine.say(text)
            
            # Run in separate thread to avoid blocking
            def speak_thread():
                try:
                    engine.runAndWait()
                except Exception as e:
                    self.logger.warning(f"TTS runtime warning: {e}")
            
            thread = threading.Thread(target=speak_thread, daemon=True)
            thread.start()
            thread.join(timeout=30)  # Maximum 30 seconds for speech
            
            return True
            
        except Exception as e:
            self.logger.error(f"System TTS failed: {e}")
            return False
    
    def listen_for_speech(self, timeout: float = 18.0, callback: Optional[Callable] = None) -> Optional[str]:
        """Listen for speech with enhanced recognition and better error handling"""
        if not self.recognizer or not self.microphone:
            self.logger.error("Speech recognition not initialized")
            return None
        
        try:
            self.is_listening = True
            if callback:
                callback("listening")
            
            self.logger.info(f"[LISTEN] Listening for voice input (timeout: {timeout}s)...")
            
            with self.microphone as source:
                # Enhanced ambient noise adjustment with longer duration
                self.logger.info("[ADJUST] Adjusting for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=2.0)
                
                # Optimized energy threshold for better speech detection
                if self.recognizer.energy_threshold < 200:
                    self.recognizer.energy_threshold = 300
                elif self.recognizer.energy_threshold > 3000:
                    self.recognizer.energy_threshold = 800
                
                self.logger.info(f"[TARGET] Energy threshold adjusted to: {self.recognizer.energy_threshold}")
                
                # Listen for speech with enhanced settings for complete sentences
                audio = self.recognizer.listen(
                    source, 
                    timeout=timeout,
                    phrase_time_limit=30.0,  # Much longer for complete thoughts
                    snowboy_configuration=None
                )
            
            if callback:
                callback("recognizing")
            
            # Try multiple recognition engines with better error handling
            recognized_text = None
            recognition_attempts = []
            
            # 1. Try Google Web Speech (most accurate) with enhanced settings
            try:
                self.logger.info("[SEARCH] Trying Google Web Speech recognition...")
                recognized_text = self.recognizer.recognize_google(
                    audio, 
                    language='en-US',
                    show_all=False,
                    key=None  # Use default free service
                )
                if recognized_text and len(recognized_text.strip()) > 2:
                    # Post-process common misrecognitions
                    recognized_text = self._fix_common_misrecognitions(recognized_text)
                    self.logger.info(f"[SUCCESS] Successfully recognized with Google Web Speech: {recognized_text}")
                    recognition_attempts.append(('Google', recognized_text))
                else:
                    recognized_text = None
                    
            except sr.UnknownValueError:
                self.logger.info("[SEARCH] Google Web Speech: Audio unclear, trying alternative...")
            except sr.RequestError as e:
                self.logger.warning(f"Google Web Speech service error: {e}")
            except Exception as e:
                self.logger.warning(f"Google Web Speech unexpected error: {e}")
            
            # 2. Fallback to Sphinx (offline) only if Google failed
            if not recognized_text:
                try:
                    self.logger.info("[SEARCH] Trying PocketSphinx recognition...")
                    sphinx_result = self.recognizer.recognize_sphinx(audio)
                    if sphinx_result and len(sphinx_result.strip()) > 2:
                        # Validate Sphinx result quality
                        words = sphinx_result.split()
                        if len(words) >= 2:  # At least 2 words
                            recognized_text = sphinx_result
                            self.logger.info(f"[SUCCESS] Successfully recognized with Sphinx: {recognized_text}")
                            recognition_attempts.append(('Sphinx', recognized_text))
                        
                except sr.UnknownValueError:
                    self.logger.info("[SEARCH] Sphinx: Audio unclear")
                except sr.RequestError as e:
                    self.logger.warning(f"Sphinx error: {e}")
                except Exception as e:
                    self.logger.warning(f"Sphinx unexpected error: {e}")
            
            # 3. Last resort: Whisper if available (would need installation)
            if not recognized_text:
                try:
                    # This would require whisper installation
                    # recognized_text = self.recognizer.recognize_whisper(audio)
                    pass
                except:
                    pass
            
            # Final validation and cleanup
            if recognized_text:
                # Clean up the recognized text
                recognized_text = recognized_text.strip()
                
                # Filter out very short or obviously incorrect results
                if len(recognized_text) < 2:
                    self.logger.warning("Recognition result too short, discarding")
                    recognized_text = None
                elif recognized_text.lower() in ['uh', 'um', 'ah', 'er', 'the', 'a', 'an']:
                    self.logger.warning("Recognition result appears to be noise, discarding")
                    recognized_text = None
            
            if callback:
                callback("success" if recognized_text else "error")
            
            # Log recognition attempts for debugging
            if recognition_attempts:
                self.logger.info(f"Recognition attempts: {recognition_attempts}")
            
            return recognized_text
            
        except sr.WaitTimeoutError:
            self.logger.info("⏰ Listening timeout - no speech detected within time limit")
            if callback:
                callback("timeout")
            return None
            
        except Exception as e:
            self.logger.error(f"Speech recognition error: {e}")
            if callback:
                callback("error")
            return None
            
        finally:
            self.is_listening = False
    
    def get_status(self) -> Dict[str, Any]:
        """Get voice assistant status"""
        return {
            'initialized': bool(self.recognizer and self.tts_engines),
            'listening': self.is_listening,
            'speaking': self.is_speaking,
            'available_tts_engines': list(self.tts_engines.keys()),
            'speech_recognition_available': SPEECH_RECOGNITION_AVAILABLE,
            'quality_level': self.quality.value,
            'microphone_ready': self.microphone is not None,
            'queue_size': self.speech_queue.qsize()
        }
    
    def shutdown(self):
        """Graceful shutdown of voice assistant"""
        self.logger.info("🔄 Shutting down Professional Voice Assistant...")
        
        # Stop worker thread
        self.stop_event.set()
        self.speech_queue.put(None)
        
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=3.0)
        
        # Clean up TTS engines
        try:
            if 'system' in self.tts_engines:
                self.tts_engines['system'].stop()
            
            if pygame.mixer.get_init():
                pygame.mixer.quit()
                
        except Exception as e:
            self.logger.debug(f"Cleanup warning: {e}")
        
        self.logger.info("[SUCCESS] Voice Assistant shutdown complete")

# Factory function for easy initialization
def create_professional_voice_assistant(quality: TTSQuality = TTSQuality.MEDIUM) -> ProfessionalVoiceAssistant:
    """Create a professional voice assistant instance"""
    return ProfessionalVoiceAssistant(quality=quality)

# Test the voice assistant
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("[TARGET] Testing Professional Voice Assistant...")
    
    assistant = create_professional_voice_assistant(TTSQuality.MEDIUM)
    
    print("Status:", assistant.get_status())
    
    # Test TTS
    def test_callback(status):
        print(f"[SPEAKER] TTS Status: {status}")
    
    print("\n[MIC] Testing professional speech...")
    assistant.speak_professional(
        "Hello! I am your professional voice assistant. I'm ready to help you with clear, professional responses.",
        callback=test_callback
    )
    
    # Wait for speech to complete
    time.sleep(3)
    
    # Test listening
    print("\n[LISTEN] Say something to test speech recognition (or press Ctrl+C to skip)...")
    try:
        def listen_callback(status):
            print(f"[LISTEN] Listening Status: {status}")
        
        result = assistant.listen_for_speech(timeout=8.0, callback=listen_callback)
        if result:
            print(f"[SUCCESS] You said: '{result}'")
            assistant.speak_professional(f"I heard you say: {result}", callback=test_callback)
        else:
            print("[ERROR] No speech detected or recognition failed")
            
    except KeyboardInterrupt:
        print("\n⏭️ Skipping speech recognition test")
    
    # Wait before shutdown
    time.sleep(2)
    assistant.shutdown()
    print("[SUCCESS] Test complete!")
