# 🎯 Professional Voice & Chat RAG System

A production-grade Retrieval-Augmented Generation (RAG) system with both advanced voice assistant and intelligent chat capabilities, built with Streamlit and enhanced AI technologies.

## 🚀 Features

### 🎤 Professional Voice Assistant
- **Enhanced Speech Recognition**: Google Web Speech API with 18-second timeout
- **Professional TTS**: Google Text-to-Speech with Microsoft Zira fallback
- **Misrecognition Correction**: Intelligent fixing of common speech-to-text errors
- **Natural Voice Interface**: Fluent pronunciation with enhanced audio quality
- **Real-time Feedback**: Interactive UI with progress indicators

### 💬 Intelligent Chat Interface
- **Text-based Conversation**: Traditional typing interface for detailed queries
- **Context-Aware Responses**: Maintains conversation history and context
- **Formatted Output**: Clean, readable responses with proper formatting
- **Export Capabilities**: Save and export chat history and conversations
- **Multi-turn Dialogue**: Support for extended conversations and follow-ups

### 📚 Advanced RAG System
- **Multi-format Document Processing**: PDF, TXT support with intelligent chunking
- **Semantic Search**: Sentence-transformers embeddings with Pinecone vector storage
- **AI-Powered Responses**: Google Gemini 2.5 Flash Lite for fast, accurate answers
- **Context-Aware**: Maintains conversation history and document context

### 🎨 Professional UI/UX
- **Modern Interface**: Clean, responsive Streamlit design
- **Voice Diagnostics**: Built-in audio system testing and troubleshooting
- **Real-time Metrics**: Performance tracking and analytics dashboard
- **Multi-tab Layout**: Organized workflow for voice, documents, and chat

## 📋 Prerequisites

- Python 3.8+
- Google API Key (for Gemini AI and TTS)
- Pinecone API Key (for vector storage)
- Microphone access for voice features

## 🛠️ Installation

1. **Clone or download the repository**
   ```bash
   cd optimized-rag-system
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` with your API keys:
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   PINECONE_API_KEY=your_pinecone_api_key_here
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

## 🗂️ Project Structure

```
optimized-rag-system/
├── app.py                              # Main Streamlit application
├── rag_core.py                         # Core RAG system implementation
├── professional_voice_assistant.py     # Professional voice system
├── streamlit_voice_integration.py      # Voice UI integration
├── chunking.py                         # Document processing and chunking
├── config.py                           # Configuration management
├── utils.py                           # Utility functions
├── requirements.txt                    # Python dependencies
├── .env.example                       # Environment variables template
├── Shahroz-1-1.pdf                   # Sample document for testing
└── models--sentence-transformers--all-mpnet-base-v2/  # Embedding model cache
```

## 🎯 Usage

### 🎤 Voice Interaction
1. **Start the Application**: Run `streamlit run app.py`
2. **Voice Tab**: Navigate to the voice interface
3. **Configure Voice**: Adjust speech rate, volume, and voice settings
4. **Start Conversation**: Click "🎤 Start Voice Chat" and speak your question
5. **Listen to Response**: The system will respond with professional voice output

### � Text Chat Interface
1. **Chat Tab**: Use traditional text-based interaction for detailed queries
2. **Type Questions**: Enter queries in the text input field
3. **View Responses**: Receive formatted, detailed responses instantly
4. **Chat History**: Access previous conversations and context
5. **Export Options**: Save conversations for future reference

### � Document Processing
1. **Document Tab**: Upload PDF or text files
2. **Process Documents**: Click to extract and index content
3. **Query Documents**: Ask questions about uploaded content via voice OR text chat

## ⚙️ Configuration

### Voice Settings
- **Speech Rate**: Adjust speaking speed (0.5x - 2.0x)
- **Voice Volume**: Control audio output level
- **Voice Gender**: Choose between male and female voices
- **TTS Engine**: Select between Google TTS and System TTS

### RAG Configuration
- **Index Name**: `universal-rag` (configurable in config.py)
- **Embedding Model**: `sentence-transformers/all-mpnet-base-v2`
- **AI Model**: `gemini-2.5-flash-lite`
- **Chunk Size**: Optimized for document processing

## 🔧 Voice System Diagnostics

The system includes built-in diagnostics for troubleshooting:

- **🎧 Microphone Test**: Verify speech recognition functionality
- **🔊 Speaker Test**: Check text-to-speech output
- **📊 Audio Status**: View system component status
- **🩺 System Diagnostic**: Comprehensive health check

## 🚨 Troubleshooting

### Common Voice Issues
- **"Could not understand audio"**: Speak more clearly or check microphone permissions
- **"No speech detected"**: Increase volume or reduce background noise
- **TTS not working**: Verify internet connection for Google TTS

### RAG System Issues
- **Document processing fails**: Check file format and size
- **No search results**: Verify Pinecone connection and index status
- **Slow responses**: Check internet connection and API quotas

## 🔐 Security & Privacy

- **API Keys**: Store securely in `.env` file (never commit to version control)
- **Voice Data**: Processed locally with Google Speech API (temporary processing)
- **Documents**: Stored in vector database with encryption
- **No Data Retention**: Voice input not permanently stored

## 📊 Performance Metrics

The system tracks:
- Query response times
- Voice recognition accuracy
- Document processing statistics
- System resource usage

## 🛡️ Production Deployment

For production use:
1. Set up proper SSL certificates
2. Configure rate limiting
3. Set up monitoring and logging
4. Use environment-specific API keys
5. Configure backup systems

## 🔄 Updates & Maintenance

- **Dependencies**: Regularly update `requirements.txt`
- **Models**: Embedding models are cached locally for performance
- **Logs**: Monitor application logs for issues
- **Vector Store**: Maintain Pinecone index health

## 🤝 Contributing

This is a production-grade system. For modifications:
1. Test thoroughly before deployment
2. Maintain backward compatibility
3. Update documentation accordingly
4. Follow security best practices

## 📝 License

This project is configured for internal/production use. Ensure compliance with all third-party service terms (Google AI, Pinecone, etc.).

## 🆘 Support

For technical issues:
1. Check the troubleshooting section
2. Review system diagnostics
3. Verify API configurations
4. Check application logs

---

**Status**: ✅ Production Ready | **Version**: 2.0 | **Last Updated**: September 2025
