from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import time
import re
import uuid
from datetime import datetime
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
import queue
import json
from typing import Dict, Optional, Any

# Initialize Flask app first
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ========== PERFORMANCE CONFIGURATION ==========
PERFORMANCE_CONFIG = {
    'MAX_WORKER_THREADS': 8,
    'TTS_TIMEOUT': 8.0,
    'AGENT_TIMEOUT': 12.0,
    'CONCURRENT_TTS': True,
    'FAST_RESPONSE_MODE': True,
    'MAX_SYNTHESIS_QUEUE': 3,
}

# ========== AZURE CONFIGURATION ==========
# Replace these with your actual Azure credentials
AZURE_CONNECTION_STRING = "your-connection-string"
AGENT_ID = "your_agent_ID"
AZURE_SPEECH_KEY = "your_azure_speeech_key"
AZURE_SPEECH_REGION = "your_azure_speech_key_region"

# ========== THREAD POOLS ==========
tts_executor = ThreadPoolExecutor(max_workers=PERFORMANCE_CONFIG['MAX_WORKER_THREADS'], thread_name_prefix="TTS")
agent_executor = ThreadPoolExecutor(max_workers=PERFORMANCE_CONFIG['MAX_WORKER_THREADS'], thread_name_prefix="Agent")

# ========== AZURE CLIENT INITIALIZATION ==========
project_client = None
agent = None
speech_configs = {}

def initialize_azure_clients():
    """Initialize Azure clients with error handling"""
    global project_client, agent
    
    try:
        # Try to import and initialize Azure AI client
        from azure.ai.projects import AIProjectClient
        from azure.identity import DefaultAzureCredential
        
        project_client = AIProjectClient.from_connection_string(
            credential=DefaultAzureCredential(),
            conn_str=AZURE_CONNECTION_STRING
        )
        agent = project_client.agents.get_agent(AGENT_ID)
        logger.info("✅ Azure AI client initialized successfully")
        return True
        
    except ImportError as e:
        logger.warning(f"⚠️ Azure AI modules not available: {e}")
        logger.info("🔧 Install with: pip install azure-ai-projects azure-identity")
        return False
    except Exception as e:
        logger.error(f"❌ Failed to initialize Azure AI client: {e}")
        return False

def initialize_speech_service():
    """Initialize Azure Speech Service with error handling"""
    global speech_configs
    
    try:
        import azure.cognitiveservices.speech as speechsdk
        
        common_voices = [
            "en-US-JennyNeural", "en-US-GuyNeural", "en-US-AriaNeural", 
            "en-US-DavisNeural", "en-GB-SoniaNeural", "en-GB-RyanNeural"
        ]
        
        for voice in common_voices:
            config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
            config.speech_synthesis_voice_name = voice
            config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3)
            speech_configs[voice] = config
            
        logger.info(f"✅ Pre-initialized {len(speech_configs)} speech configs")
        return True
        
    except ImportError as e:
        logger.warning(f"⚠️ Azure Speech SDK not available: {e}")
        logger.info("🔧 Install with: pip install azure-cognitiveservices-speech")
        return False
    except Exception as e:
        logger.error(f"❌ Failed to initialize speech service: {e}")
        return False

# Initialize Azure services
azure_ai_available = initialize_azure_clients()
speech_service_available = initialize_speech_service()

# ========== PERFORMANCE MONITORING ==========
performance_metrics = {
    'tts_times': [],
    'agent_times': [],
    'total_requests': 0,
    'concurrent_sessions': 0,
    'errors': 0
}

def add_performance_metric(metric_type: str, value: float):
    """Add performance metric with rolling window"""
    if metric_type in performance_metrics:
        metrics = performance_metrics[metric_type]
        metrics.append(value)
        if len(metrics) > 100:
            metrics.pop(0)

# ========== SESSION MANAGEMENT ==========
active_sessions = {}
active_synthesis_tasks = {}
synthesis_lock = threading.Lock()
session_lock = threading.Lock()

class OptimizedVoiceSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.thread_id = None
        self.thread = None
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        self.message_count = 0
        self.voice_name = "en-US-JennyNeural"
        self.is_processing = False
        self.current_synthesis_ids = set()
        self.performance_stats = {
            'avg_response_time': 0,
            'total_messages': 0,
            'tts_errors': 0,
            'agent_errors': 0
        }
        
    def create_thread(self) -> bool:
        """Create Azure AI thread with error handling"""
        if not azure_ai_available or not project_client:
            logger.warning("⚠️ Azure AI not available, creating mock thread")
            self.thread_id = f"mock_thread_{uuid.uuid4().hex[:8]}"
            return True
            
        try:
            start_time = time.time()
            self.thread = project_client.agents.create_thread()
            self.thread_id = self.thread.id
            creation_time = (time.time() - start_time) * 1000
            logger.info(f"🆕 Created thread {self.thread_id} in {creation_time:.0f}ms")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to create thread: {e}")
            # Create mock thread as fallback
            self.thread_id = f"mock_thread_{uuid.uuid4().hex[:8]}"
            return True
    
    def update_activity(self):
        self.last_activity = datetime.now()
    
    def set_voice(self, voice_name: str):
        if voice_name in speech_configs or not speech_service_available:
            self.voice_name = voice_name
        else:
            self.voice_name = "en-US-JennyNeural"
    
    def set_processing(self, is_processing: bool):
        self.is_processing = is_processing
        
    def add_synthesis_id(self, synthesis_id: str) -> bool:
        if len(self.current_synthesis_ids) >= PERFORMANCE_CONFIG['MAX_SYNTHESIS_QUEUE']:
            return False
        self.current_synthesis_ids.add(synthesis_id)
        return True
        
    def remove_synthesis_id(self, synthesis_id: str):
        self.current_synthesis_ids.discard(synthesis_id)
        
    def cancel_all_synthesis(self):
        with synthesis_lock:
            for synth_id in list(self.current_synthesis_ids):
                if synth_id in active_synthesis_tasks:
                    active_synthesis_tasks[synth_id]['cancelled'] = True
            self.current_synthesis_ids.clear()

# ========== TTS FUNCTIONS ==========
def generate_speech_audio_optimized(text: str, voice_name: str, synthesis_id: str) -> Optional[bytes]:
    """Generate speech with fallbacks"""
    start_time = time.time()
    
    try:
        # Check cancellation
        with synthesis_lock:
            if synthesis_id in active_synthesis_tasks and active_synthesis_tasks[synthesis_id].get('cancelled', False):
                return None
        
        if not speech_service_available:
            logger.warning("⚠️ Speech service not available, using mock audio")
            time.sleep(1)  # Simulate processing time
            return b"mock_audio_data"  # Return mock data
        
        import azure.cognitiveservices.speech as speechsdk
        
        # Use pre-configured speech config
        speech_config = speech_configs.get(voice_name)
        if not speech_config:
            speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
            speech_config.speech_synthesis_voice_name = voice_name
            speech_config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3)
        
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
        
        # Check cancellation again
        with synthesis_lock:
            if synthesis_id in active_synthesis_tasks and active_synthesis_tasks[synthesis_id].get('cancelled', False):
                return None
        
        result = synthesizer.speak_text_async(text).get()
        
        synthesis_time = (time.time() - start_time) * 1000
        add_performance_metric('tts_times', synthesis_time)
        
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            logger.info(f"✅ Generated audio in {synthesis_time:.0f}ms")
            return result.audio_data
        else:
            raise Exception(f"Speech synthesis failed: {result.reason}")
            
    except Exception as e:
        synthesis_time = (time.time() - start_time) * 1000
        logger.error(f"❌ TTS error after {synthesis_time:.0f}ms: {e}")
        performance_metrics['errors'] += 1
        raise

# ========== AGENT COMMUNICATION ==========
def chat_with_agent_optimized(user_input: str, session: OptimizedVoiceSession) -> Dict[str, Any]:
    """Chat with agent with fallbacks"""
    start_time = time.time()
    
    try:
        session.cancel_all_synthesis()
        session.set_processing(True)
        
        if not azure_ai_available or not project_client:
            # Mock response when Azure AI is not available
            time.sleep(1)  # Simulate processing time
            mock_responses = [
                "I understand your message. This is a test response since Azure AI is not currently available.",
                "Thank you for your input. I'm running in demo mode right now.",
                "I hear you! This is a simulated response for testing purposes.",
                "That's interesting! I'm currently operating in offline mode.",
                "I appreciate your message. This is a placeholder response."
            ]
            import random
            response = random.choice(mock_responses)
            
            session.message_count += 1
            session.update_activity()
            session.set_processing(False)
            
            processing_time = (time.time() - start_time) * 1000
            
            return {
                "success": True,
                "response": response,
                "thread_id": session.thread_id,
                "message_count": session.message_count,
                "response_time_ms": processing_time,
                "mode": "mock"
            }
        
        # Real Azure AI processing
        if not session.thread:
            return {
                "success": False,
                "error": "No active thread",
                "response": "Session error. Please start a new session."
            }
        
        # Send message to Azure AI
        project_client.agents.create_message(
            thread_id=session.thread.id,
            role="user",
            content=user_input
        )
        
        run = project_client.agents.create_and_process_run(
            thread_id=session.thread.id,
            agent_id=agent.id
        )
        
        # Wait for completion
        max_wait_time = PERFORMANCE_CONFIG['AGENT_TIMEOUT']
        wait_time = 0
        check_interval = 0.2
        
        while run.status not in ["completed", "failed", "cancelled"] and wait_time < max_wait_time:
            time.sleep(check_interval)
            wait_time += check_interval
            
            if not session.is_processing:
                return {
                    "success": False,
                    "error": "Processing interrupted",
                    "response": "Processing interrupted.",
                    "interrupted": True
                }
            
            run = project_client.agents.get_run(thread_id=session.thread.id, run_id=run.id)
        
        processing_time = (time.time() - start_time) * 1000
        add_performance_metric('agent_times', processing_time)
        
        if run.status == "failed":
            session.set_processing(False)
            return {
                "success": False,
                "error": "Agent run failed",
                "response": "I encountered an error processing your request."
            }
        
        if wait_time >= max_wait_time:
            session.set_processing(False)
            return {
                "success": False,
                "error": "Agent timeout",
                "response": "I'm taking longer than expected. Please try again."
            }
        
        # Extract response
        messages = project_client.agents.list_messages(thread_id=session.thread.id)
        latest_response = ""
        
        if hasattr(messages, 'text_messages'):
            text_messages = [msg.as_dict() for msg in messages.text_messages]
            if text_messages:
                latest_response = text_messages[0].get('text', {}).get('value', '')
        
        latest_response = re.sub(r'【.*?†source】', '', latest_response).strip()
        
        if not latest_response:
            latest_response = "I received your message but couldn't generate a proper response."
        
        session.message_count += 1
        session.update_activity()
        session.set_processing(False)
        
        return {
            "success": True,
            "response": latest_response,
            "thread_id": session.thread_id,
            "message_count": session.message_count,
            "response_time_ms": processing_time
        }
        
    except Exception as e:
        processing_time = (time.time() - start_time) * 1000
        logger.error(f"❌ Agent error: {e}")
        session.set_processing(False)
        performance_metrics['errors'] += 1
        
        return {
            "success": False,
            "error": str(e),
            "response": f"I encountered a technical issue: {str(e)}"
        }

# ========== API ENDPOINTS ==========

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check with service status"""
    avg_tts_time = sum(performance_metrics['tts_times'][-10:]) / len(performance_metrics['tts_times'][-10:]) if performance_metrics['tts_times'] else 0
    avg_agent_time = sum(performance_metrics['agent_times'][-10:]) / len(performance_metrics['agent_times'][-10:]) if performance_metrics['agent_times'] else 0
    
    return jsonify({
        "status": "healthy",
        "services": {
            "azure_ai": azure_ai_available,
            "speech_service": speech_service_available,
            "flask": True
        },
        "active_sessions": len(active_sessions),
        "active_synthesis": len(active_synthesis_tasks),
        "performance": {
            "avg_tts_time_ms": round(avg_tts_time, 2),
            "avg_agent_time_ms": round(avg_agent_time, 2),
            "total_requests": performance_metrics['total_requests'],
            "error_rate": performance_metrics['errors'] / max(performance_metrics['total_requests'], 1)
        },
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/start_session', methods=['POST'])
def start_session():
    """Start a new session"""
    try:
        start_time = time.time()
        session_id = str(uuid.uuid4())
        
        session = OptimizedVoiceSession(session_id)
        
        if not session.create_thread():
            return jsonify({
                "success": False,
                "error": "Failed to create thread"
            }), 500
        
        with session_lock:
            active_sessions[session_id] = session
            performance_metrics['concurrent_sessions'] = len(active_sessions)
        
        creation_time = (time.time() - start_time) * 1000
        performance_metrics['total_requests'] += 1
        
        logger.info(f"🚀 Session {session_id} created in {creation_time:.0f}ms")
        
        return jsonify({
            "success": True,
            "session_id": session_id,
            "thread_id": session.thread_id,
            "message": "Session started successfully",
            "created_at": session.created_at.isoformat(),
            "creation_time_ms": creation_time,
            "azure_ai_available": azure_ai_available,
            "speech_service_available": speech_service_available
        })
        
    except Exception as e:
        logger.error(f"❌ Error starting session: {e}")
        performance_metrics['errors'] += 1
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/send_message', methods=['POST'])
def send_message():
    """Send message to agent"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "success": False,
                "error": "No data provided"
            }), 400
        
        session_id = data.get('session_id')
        message = data.get('message', '').strip()
        
        if not session_id or not message:
            return jsonify({
                "success": False,
                "error": "Session ID and message are required"
            }), 400
        
        with session_lock:
            session = active_sessions.get(session_id)
        
        if not session:
            return jsonify({
                "success": False,
                "error": "Invalid session ID"
            }), 404
        
        logger.info(f"👤 Processing message: {message[:50]}...")
        
        session.cancel_all_synthesis()
        
        if session.is_processing:
            session.set_processing(False)
        
        # Check for exit commands
        exit_words = ["exit", "quit", "stop", "goodbye", "bye"]
        if any(word in message.lower() for word in exit_words):
            return jsonify({
                "success": True,
                "response": "Goodbye! Thank you for using the Voice AI Assistant!",
                "should_end_session": True,
                "thread_id": session.thread_id,
                "message_count": session.message_count
            })
        
        # Process with agent
        future = agent_executor.submit(chat_with_agent_optimized, message, session)
        
        try:
            result = future.result(timeout=PERFORMANCE_CONFIG['AGENT_TIMEOUT'] + 1)
            performance_metrics['total_requests'] += 1
            return jsonify(result)
            
        except TimeoutError:
            logger.error(f"❌ Message processing timeout")
            session.set_processing(False)
            performance_metrics['errors'] += 1
            
            return jsonify({
                "success": False,
                "error": "Processing timeout",
                "response": "I'm taking too long to respond. Please try again."
            }), 408
        
    except Exception as e:
        logger.error(f"❌ Error in send_message: {e}")
        performance_metrics['errors'] += 1
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/synthesize_speech', methods=['POST'])
def synthesize_speech():
    """Generate speech audio"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "success": False,
                "error": "No data provided"
            }), 400
        
        text = data.get('text', '').strip()
        voice_name = data.get('voice_name', 'en-US-JennyNeural')
        session_id = data.get('session_id')
        
        if not text:
            return jsonify({
                "success": False,
                "error": "Text is required"
            }), 400
        
        synthesis_id = str(uuid.uuid4())
        
        # Get session
        session = None
        if session_id:
            with session_lock:
                session = active_sessions.get(session_id)
            
            if session:
                if not session.add_synthesis_id(synthesis_id):
                    return jsonify({
                        "success": False,
                        "error": "Too many concurrent requests"
                    }), 429
        
        # Register synthesis task
        with synthesis_lock:
            active_synthesis_tasks[synthesis_id] = {
                'session_id': session_id,
                'text': text[:50] + '...' if len(text) > 50 else text,
                'voice_name': voice_name,
                'cancelled': False,
                'started_at': datetime.now()
            }
        
        if session:
            session.set_voice(voice_name)
        
        logger.info(f"🎤 Starting TTS {synthesis_id}")
        
        # Submit to thread pool
        future = tts_executor.submit(generate_speech_audio_optimized, text, voice_name, synthesis_id)
        
        try:
            audio_data = future.result(timeout=PERFORMANCE_CONFIG['TTS_TIMEOUT'])
            
            # Check if cancelled
            with synthesis_lock:
                task_info = active_synthesis_tasks.get(synthesis_id, {})
                if task_info.get('cancelled', False):
                    if session:
                        session.remove_synthesis_id(synthesis_id)
                    active_synthesis_tasks.pop(synthesis_id, None)
                    return jsonify({
                        "success": False,
                        "error": "Synthesis cancelled",
                        "cancelled": True
                    }), 499
                
                active_synthesis_tasks.pop(synthesis_id, None)
            
            if session:
                session.remove_synthesis_id(synthesis_id)
            
            if audio_data is None:
                return jsonify({
                    "success": False,
                    "error": "Synthesis was cancelled",
                    "cancelled": True
                }), 499
            
            if not speech_service_available:
                # Return success for mock audio
                return jsonify({
                    "success": True,
                    "message": "Mock audio generated (Speech service not available)",
                    "synthesis_id": synthesis_id
                })
            
            logger.info(f"✅ TTS completed for {synthesis_id}")
            
            return Response(
                audio_data,
                mimetype='audio/mpeg',
                headers={
                    'Content-Disposition': 'inline; filename="speech.mp3"',
                    'Cache-Control': 'no-cache',
                    'Content-Length': str(len(audio_data)),
                    'X-Synthesis-ID': synthesis_id
                }
            )
            
        except TimeoutError:
            logger.error(f"❌ TTS timeout for {synthesis_id}")
            
            with synthesis_lock:
                if synthesis_id in active_synthesis_tasks:
                    active_synthesis_tasks[synthesis_id]['cancelled'] = True
            
            if session:
                session.remove_synthesis_id(synthesis_id)
            
            performance_metrics['errors'] += 1
            
            return jsonify({
                "success": False,
                "error": "Speech synthesis timeout"
            }), 408
        
    except Exception as e:
        logger.error(f"❌ TTS error: {e}")
        
        if 'synthesis_id' in locals():
            with synthesis_lock:
                active_synthesis_tasks.pop(synthesis_id, None)
            if 'session' in locals() and session:
                session.remove_synthesis_id(synthesis_id)
        
        performance_metrics['errors'] += 1
        
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/end_session', methods=['POST'])
def end_session():
    """End a session"""
    try:
        data = request.get_json()
        session_id = data.get('session_id') if data else None
        
        if not session_id:
            return jsonify({
                "success": False,
                "error": "Session ID is required"
            }), 400
        
        with session_lock:
            session = active_sessions.pop(session_id, None)
            performance_metrics['concurrent_sessions'] = len(active_sessions)
        
        if not session:
            return jsonify({
                "success": False,
                "error": "Session not found"
            }), 404
        
        session.cancel_all_synthesis()
        
        with synthesis_lock:
            to_remove = [
                synth_id for synth_id, task_info in active_synthesis_tasks.items()
                if task_info.get('session_id') == session_id
            ]
            for synth_id in to_remove:
                active_synthesis_tasks.pop(synth_id, None)
        
        duration = (datetime.now() - session.created_at).total_seconds()
        
        logger.info(f"🛑 Session {session_id} ended (duration: {duration:.1f}s)")
        
        return jsonify({
            "success": True,
            "message": "Session ended successfully",
            "session_duration": duration,
            "message_count": session.message_count,
            "thread_id": session.thread_id
        })
        
    except Exception as e:
        logger.error(f"❌ Error ending session: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/set_voice', methods=['POST'])
def set_voice():
    """Set voice for session"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "success": False,
                "error": "No data provided"
            }), 400
        
        session_id = data.get('session_id')
        voice_name = data.get('voice_name')
        
        if not session_id or not voice_name:
            return jsonify({
                "success": False,
                "error": "Session ID and voice name are required"
            }), 400
        
        with session_lock:
            session = active_sessions.get(session_id)
        
        if not session:
            return jsonify({
                "success": False,
                "error": "Invalid session ID"
            }), 404
        
        session.set_voice(voice_name)
        session.update_activity()
        
        return jsonify({
            "success": True,
            "voice_name": voice_name,
            "session_id": session_id
        })
        
    except Exception as e:
        logger.error(f"❌ Error setting voice: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# ========== ERROR HANDLERS ==========

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "success": False,
        "error": "Endpoint not found"
    }), 404

@app.errorhandler(500)
def internal_error(error):
    performance_metrics['errors'] += 1
    return jsonify({
        "success": False,
        "error": "Internal server error"
    }), 500

# ========== MAIN EXECUTION ==========

def print_startup_info():
    """Print startup information"""
    print("\n" + "="*60)
    print("🎧 AZURE VOICE AI ASSISTANT - BACKEND")
    print("="*60)
    print(f"🔗 Azure AI Available: {'✅ Yes' if azure_ai_available else '❌ No (Mock mode)'}")
    print(f"🎤 Speech Service Available: {'✅ Yes' if speech_service_available else '❌ No (Mock mode)'}")
    print(f"🌐 Flask Server: ✅ Ready")
    print("="*60)
    print("📡 API ENDPOINTS:")
    print("  GET  /api/health            - Health check")
    print("  POST /api/start_session     - Start new session")
    print("  POST /api/send_message      - Send message to agent")
    print("  POST /api/end_session       - End session")
    print("  POST /api/synthesize_speech - Generate speech audio")
    print("  POST /api/set_voice         - Set voice for session")
    print("="*60)
    print("🚀 Starting server on http://localhost:5000")
    
    if not azure_ai_available:
        print("⚠️  Azure AI not available - using mock responses")
        print("💡 Install: pip install azure-ai-projects azure-identity")
    
    if not speech_service_available:
        print("⚠️  Speech service not available - using mock audio")
        print("💡 Install: pip install azure-cognitiveservices-speech")
    
    print("="*60)

if __name__ == '__main__':
    try:
        print_startup_info()
        
        app.run(
            debug=True,
            host='0.0.0.0', 
            port=5000,
            threaded=True
        )
        
    except KeyboardInterrupt:
        logger.info("🛑 Server shutdown requested")
        tts_executor.shutdown(wait=True)
        agent_executor.shutdown(wait=True)
        print("👋 Server stopped")
        
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        print(f"❌ Failed to start server: {e}")
