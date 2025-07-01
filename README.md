# Enhanced RAG & Agent Question-Answering System

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

A sophisticated educational assistant leveraging Retrieval-Augmented Generation (RAG) and advanced AI agent techniques with multimodal capabilities, including voice input and YouTube video context integration.

## 🌟 Features

- **Dual Operation Modes**:
  - **RAG Mode**: Standard retrieval-augmented generation for factual responses
  - **Agent Mode**: Advanced reasoning with deeper analysis capabilities

- **Advanced Prompting Techniques**:
  - Auto Reasoning
  - Standard Reasoning
  - Chain of Thought (CoT)
  - Tree of Thought (ToT) 
  - Graph of Thought (GoT)

- **Multimodal Input/Output**:
  - Text-based questioning
  - Voice input with automatic transcription
  - YouTube video context integration

- **Dynamic Knowledge Expansion**:
  - Automatic detection of knowledge gaps
  - YouTube video search for relevant content
  - Contextual knowledge extraction from videos
  - Embedding storage in vector database

- **Sophisticated Processing Pipeline**:
  - Natural Language Processing with spaCy and NLTK
  - Semantic chunking and text preprocessing
  - Named Entity Recognition and POS tagging
  - Embedding generation with sentence-transformers

## 🛠️ Architecture

The system consists of several integrated components:

### Core Components

1. **Base System**: Handles common functionality for embedding generation, video processing, and NLP tasks
2. **Enhanced RAG System**: Implements retrieval-augmented generation with standard prompting
3. **Agent System**: Extends base functionality with advanced reasoning capabilities
4. **Video Embedding Manager**: Processes YouTube videos and manages embeddings

### Technical Flow

1. User submits a query with a video URL (text or voice)
2. System generates embeddings for the query
3. Relevant information is retrieved from Pinecone vector database
4. If knowledge in the YouTube video is insufficient, relevance is low (avg < 0.55):
   - knowledge is expanded, relevant chunks are re-retrieved to include the new information
5. If user submits a query without a video URL (text or voice)
   - System searches for relevant YouTube videos
   - Video transcripts are processed and embedded
   - Knowledge base is expanded
5. Response is generated using specified prompting technique
6. Sources and context information are provided to user

## 📋 Requirements

```
gradio>=4.0.0
pinecone-client>=3.0.0
sentence-transformers>=2.2.2
groq>=0.3.0
torch>=2.0.0
nltk>=3.8.1
spacy>=3.7.0
youtube-transcript-api>=0.6.1
google-api-python-client>=2.0.0
python-dotenv>=1.0.0
numpy>=1.24.0
pyaudio>=0.2.13
requests>=2.28.0
```

## 🚀 Installation

1. Clone this repository (not required if downloaded files already)
   ```bash
   git clone https://github.com/pranavacchu/enhanced-rag-system-for-youtube.git
   cd enhanced-rag-system-for-youtube
   ```

2. Install required packages
   ```bash
   pip install -r requirements.txt
   ```

3. Install language models
   ```bash
   python -m spacy download en_core_web_sm
   python -m nltk.downloader stopwords wordnet
   ```

4. Set up environment variables in a `.env` file
   ```
   PINECONE_API_KEY=your_pinecone_api_key
   GROQ_API_KEY=your_groq_api_key
   PINECONE_INDEX_NAME=your_index_name
   YOUTUBE_API_KEY=your_youtube_api_key
   ```

## 💻 Usage

1. Start the application
   ```bash
   python app_gradio.py
   ```

2. Access the web interface at `http://localhost:7860` (by default)

3. Choose a mode (RAG or Agent) and prompting technique

4. Enter your question or use voice input

5. Optionally provide a YouTube video URL for additional context

6. Submit your query and review the answer and sources

## 📊 System Breakdown

### `app_gradio.py`

The main application file that:
- Sets up the Gradio web interface
- Initializes system components
- Coordinates workflow between components
- Handles voice input and transcription
- Manages user interactions and query processing
- Handles YouTube video context integration

### `video_embeddings.py` 

Dedicated module for:
- Processing YouTube videos
- Extracting and preprocessing transcripts
- Generating and managing embeddings
- Integrating new knowledge into the vector database
- Performing NLP tasks on video content

## 🔧 Configuration

The system can be customized through environment variables:

- `PINECONE_API_KEY`: API key for Pinecone vector database
- `GROQ_API_KEY`: API key for Groq LLM API
- `PINECONE_INDEX_NAME`: Name of your Pinecone index
- `YOUTUBE_API_KEY`: API key for YouTube Data API

## 🧠 Advanced Prompting Techniques

The system implements four distinct prompting strategies:

1. **Standard**: Basic question-answer format for straightforward queries
2. **Chain of Thought (CoT)**: Step-by-step reasoning process for complex problems
3. **Tree of Thought (ToT)**: Explores multiple solution paths simultaneously
4. **Graph of Thought (GoT)**: Maps interconnected concepts and relationships
5. **Auto**: Automatically selects appropriate prompting technique based on keywords present in user query. 

## 🌐 Voice Input Functionality

The system includes voice recording capabilities:
- Real-time audio recording using PyAudio
- Audio transcription via AssemblyAI's API
- Seamless integration with the query processing pipeline

## 📊 Knowledge Expansion Logic

The system intelligently expands its knowledge base when:
- Existing knowledge relevance is below threshold (0.55)
- New video content is provided by the user
- YouTube search yields relevant educational content

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Groq](https://groq.com/) for LLM API
- [Pinecone](https://www.pinecone.io/) for vector database
- [AssemblyAI](https://www.assemblyai.com/) for speech transcription
- [YouTube Data API](https://developers.google.com/youtube/v3) for video search