import gradio as gr
import os
from dotenv import load_dotenv
import numpy as np
import pinecone
from sentence_transformers import SentenceTransformer
from groq import Groq
import torch
from typing import List, Dict, Any, Optional
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
from youtube_transcript_api import YouTubeTranscriptApi
import re

# Load environment variables
load_dotenv()

# Custom CSS for dynamic background only
custom_css = """
@keyframes gradient {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.gradio-container {
    background: linear-gradient(-45deg, #000000, #156A70, #000000, #074044);
    background-size: 400% 400%;
    animation: gradient 15s ease infinite;
}
"""

class EnhancedRAGSystem:
    def __init__(self, pinecone_api_key: str, groq_api_key: str, index_name: str,
                 model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        """
        Initialize Enhanced RAG system with Pinecone, Groq, and models.
        """
        # Initialize Pinecone client
        self.pc = pinecone.Pinecone(api_key=pinecone_api_key)
        self.index = self.pc.Index(index_name)

        # Initialize Groq client
        self.groq_client = Groq(api_key=groq_api_key)

        # Check for GPU availability
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        # Initialize embedding model
        self.embedding_model = SentenceTransformer(model_name).to(self.device)

        # Set parameters
        self.top_k = 1  # Retrieve only the top source

        # Initialize NLP tools
        try:
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
        except Exception as e:
            print(f"Warning: NLTK resource download issue. Error: {e}")
            self.stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'to', 'of', 'in', 'for'}
            self.lemmatizer = None

        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            print("Warning: spaCy model not found. Using simple pipeline.")
            self.nlp = spacy.blank("en")

    def extract_video_id(self, url: str) -> Optional[str]:
        """Extract YouTube video ID from URL."""
        patterns = [
            r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([^&\n?]+)',
            r'youtube\.com\/embed\/([^&\n?]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None

    def get_video_transcript(self, video_id: str) -> str:
        """Get transcript from YouTube video."""
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            return ' '.join([entry['text'] for entry in transcript])
        except Exception as e:
            print(f"Error getting transcript: {e}")
            return ""

    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for the query."""
        return self.embedding_model.encode(query, convert_to_tensor=True).cpu().tolist()

    def retrieve_relevant_chunks(self, query_embedding: List[float]) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks from Pinecone."""
        query_response = self.index.query(
            vector=query_embedding,
            top_k=self.top_k,
            include_metadata=True
        )
        
        return [{
            'score': match.score,
            'text': match.metadata.get('text_sample', 'No text available')
        } for match in query_response['matches']]

    def generate_answer(self, query: str, relevant_chunks: List[Dict[str, Any]], video_context: str = "") -> str:
        """Generate a comprehensive answer using Groq."""
        if not relevant_chunks and not video_context:
            return "I couldn't find sufficient information to answer your question."

        # Combine context from both sources
        context = ""
        if relevant_chunks:
            context += f"Knowledge Base Context:\n{relevant_chunks[0]['text']}\n\n"
        if video_context:
            context += f"Video Context:\n{video_context}\n\n"

        # Prepare prompt
        prompt = f"""
        You are an intelligent assistant specialized in educational content. Answer the question using the provided context.

        USER QUESTION:
        {query}

        CONTEXT:
        {context}

        Please provide a clear, concise answer based on the context. If the context doesn't contain enough information, say so.

        ANSWER:
        """

        # Generate response using Groq
        chat_completion = self.groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an intelligent assistant specialized in educational content."},
                {"role": "user", "content": prompt}
            ],
            model="llama3-70b-8192",
            temperature=0.7,
            top_p=0.95,
            max_tokens=512
        )

        return chat_completion.choices[0].message.content

    def query(self, question: str, video_url: str = "") -> Dict[str, Any]:
        """Process a user query and return an answer with supporting evidence."""
        # Generate embedding
        query_embedding = self.embed_query(question)
        
        # Get relevant chunks from knowledge base
        relevant_chunks = self.retrieve_relevant_chunks(query_embedding)
        
        # Process video transcript if URL is provided
        video_context = ""
        if video_url:
            video_id = self.extract_video_id(video_url)
            if video_id:
                video_context = self.get_video_transcript(video_id)
        
        # Generate answer
        answer = self.generate_answer(question, relevant_chunks, video_context)
        
        return {
            "answer": answer,
            "sources": relevant_chunks,
            "video_context": video_context
        }

# Initialize RAG system
rag_system = EnhancedRAGSystem(
    pinecone_api_key=os.getenv('PINECONE_API_KEY'),
    groq_api_key=os.getenv('GROQ_API_KEY'),
    index_name=os.getenv('PINECONE_INDEX_NAME', 'embeddings')
)

def process_query(question: str, video_url: str = "") -> tuple[str, str]:
    """Process the query and return answer and sources."""
    if not question.strip():
        return "Please enter a question.", "No sources available."
    
    try:
        result = rag_system.query(question, video_url)
        
        # Format sources
        sources_text = ""
        if result["sources"]:
            sources_text += "Knowledge Base Sources:\n"
            sources_text += "\n\n".join([
                f"Source (Relevance: {source['score']:.2f}):\n{source['text']}"
                for source in result["sources"]
            ])
        
        if result["video_context"]:
            if sources_text:
                sources_text += "\n\n"
            sources_text += "Video Context:\n" + result["video_context"]
        
        return result["answer"], sources_text
    except Exception as e:
        return f"Error: {str(e)}", "Error retrieving sources."

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown("""
    # 🤖 Enhanced RAG Q&A System
    Ask questions and get answers based on the knowledge base and YouTube videos.
    """)
    
    with gr.Row():
        with gr.Column(scale=4):
            question_input = gr.Textbox(
                label="Your Question",
                placeholder="Type your question here...",
                lines=2
            )
            video_url_input = gr.Textbox(
                label="YouTube Video URL (Optional)",
                placeholder="Paste a YouTube video URL here...",
                lines=1
            )
            submit_btn = gr.Button("🔍 Get Answer", variant="primary")
        
    with gr.Row():
        with gr.Column(scale=2):
            answer_output = gr.Textbox(
                label="Answer",
                lines=5,
                show_copy_button=True
            )
        with gr.Column(scale=2):
            sources_output = gr.Textbox(
                label="Sources",
                lines=5,
                show_copy_button=True
            )
    
    # Handle submission
    submit_btn.click(
        fn=process_query,
        inputs=[question_input, video_url_input],
        outputs=[answer_output, sources_output]
    )
    
    gr.Markdown("""
    ### Tips:
    - Be specific in your questions
    - Questions should be related to the content in the knowledge base or video
    - You can optionally provide a YouTube video URL for additional context
    - The system will provide relevant sources along with the answer
    """)

# Launch the app
if __name__ == "__main__":
    demo.launch(share=True)  # share=True creates a public URL