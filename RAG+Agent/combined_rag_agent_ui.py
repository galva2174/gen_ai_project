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

# Custom CSS for dynamic background
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

# Original RAG Implementation
class OriginalRAGSystem:
    def __init__(self, pinecone_api_key: str, groq_api_key: str, index_name: str,
                 model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        """
        Initialize Original RAG system with Pinecone, Groq, and models.
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

    def generate_answer(self, query: str, relevant_chunks: List[Dict[str, Any]], video_context: str = "", 
                       prompting_technique: str = "standard") -> str:
        """Generate a comprehensive answer using Groq with different prompting techniques."""
        if not relevant_chunks and not video_context:
            return "I couldn't find sufficient information to answer your question."

        # Combine context from both sources
        context = ""
        if relevant_chunks:
            context += f"Knowledge Base Context:\n{relevant_chunks[0]['text']}\n\n"
        if video_context:
            context += f"Video Context:\n{video_context}\n\n"

        # Select prompting technique
        if prompting_technique.lower() == "cot":
            prompt = self._generate_cot_prompt(query, context)
        elif prompting_technique.lower() == "tot":
            prompt = self._generate_tot_prompt(query, context)
        elif prompting_technique.lower() == "got":
            prompt = self._generate_got_prompt(query, context)
        else:  # standard
            prompt = self._generate_standard_prompt(query, context)

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

    def _generate_standard_prompt(self, query: str, context: str) -> str:
        """Generate standard RAG prompt."""
        return f"""
        You are an intelligent assistant specialized in educational content. Your task is to create a comprehensive, well-structured answer to the user's question using the provided context.

        USER QUESTION:
        {query}

        RELEVANT CONTEXT:
        {context}

        Please provide a clear, concise answer based on the context. If the context doesn't contain enough information, say so.
        """

    def _generate_cot_prompt(self, query: str, context: str) -> str:
        """Generate Chain of Thought prompt for RAG."""
        return f"""
        You are an intelligent assistant using Chain of Thought reasoning. Break down the problem and think step by step.

        USER QUESTION:
        {query}

        RELEVANT CONTEXT:
        {context}

        Let's think step by step:

        1. First, let's understand what the question is asking:
        [Your understanding of the question]

        2. Next, let's analyze the relevant context:
        [Your analysis of the context]

        3. Now, let's break down the reasoning process:
        [Step-by-step reasoning]

        4. Let's consider potential implications and connections:
        [Analysis of implications]

        5. Finally, let's formulate a comprehensive answer:
        [Your final answer]

        Remember to:
        - Show your work and reasoning at each step
        - Explain how you arrived at each conclusion
        - Consider alternative perspectives
        - Identify any assumptions or limitations
        """

    def _generate_tot_prompt(self, query: str, context: str) -> str:
        """Generate Tree of Thought prompt for RAG."""
        return f"""
        You are an intelligent assistant using Tree of Thought reasoning. Explore different branches of reasoning and evaluate them.

        USER QUESTION:
        {query}

        RELEVANT CONTEXT:
        {context}

        Let's explore different branches of reasoning:

        Branch 1: [First approach/interpretation]
        - Reasoning:
        - Evidence:
        - Strengths:
        - Weaknesses:

        Branch 2: [Alternative approach/interpretation]
        - Reasoning:
        - Evidence:
        - Strengths:
        - Weaknesses:

        Branch 3: [Another perspective]
        - Reasoning:
        - Evidence:
        - Strengths:
        - Weaknesses:

        Now, let's evaluate these branches:
        1. Compare the different approaches
        2. Identify the most promising path
        3. Combine insights from multiple branches
        4. Consider potential synthesis

        Final Answer:
        [Your comprehensive answer that synthesizes the best elements from different branches]
        """

    def _generate_got_prompt(self, query: str, context: str) -> str:
        """Generate Graph of Thought prompt for RAG."""
        return f"""
        You are an intelligent assistant using Graph of Thought reasoning. Create a network of interconnected ideas and explore their relationships.

        USER QUESTION:
        {query}

        RELEVANT CONTEXT:
        {context}

        Let's create a knowledge graph of interconnected concepts:

        Core Concepts:
        1. [Main concept 1]
           - Related ideas:
           - Connections:
           - Implications:

        2. [Main concept 2]
           - Related ideas:
           - Connections:
           - Implications:

        3. [Main concept 3]
           - Related ideas:
           - Connections:
           - Implications:

        Now, let's analyze the relationships:
        1. Direct connections between concepts
        2. Indirect relationships and implications
        3. Feedback loops and dependencies
        4. Emergent properties from the network

        Let's synthesize the insights:
        1. Key patterns in the relationships
        2. Most significant connections
        3. Potential new insights from the network
        4. Implications for the original question

        Final Answer:
        [Your comprehensive answer that leverages the interconnected nature of the concepts]
        """

    def query(self, question: str, video_url: str = "", prompting_technique: str = "standard") -> Dict[str, Any]:
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
        
        # Generate answer with selected prompting technique
        answer = self.generate_answer(question, relevant_chunks, video_context, prompting_technique)
        
        return {
            "answer": answer,
            "sources": relevant_chunks,
            "video_context": video_context
        }

# Original Agent Implementation
class OriginalAgentSystem:
    def __init__(self, pinecone_api_key: str, groq_api_key: str, index_name: str,
                 model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        """
        Initialize Original Agent system with Pinecone, Groq, and models.
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

    def generate_answer(self, query: str, relevant_chunks: List[Dict[str, Any]], video_context: str = "", 
                       prompting_technique: str = "standard") -> str:
        """Generate a comprehensive answer using Groq with different prompting techniques."""
        if not relevant_chunks and not video_context:
            return "I couldn't find sufficient information to answer your question."

        # Combine context from both sources
        context = ""
        if relevant_chunks:
            context += f"Knowledge Base Context:\n{relevant_chunks[0]['text']}\n\n"
        if video_context:
            context += f"Video Context:\n{video_context}\n\n"

        # Select prompting technique
        if prompting_technique.lower() == "cot":
            prompt = self._generate_cot_prompt(query, context)
        elif prompting_technique.lower() == "tot":
            prompt = self._generate_tot_prompt(query, context)
        elif prompting_technique.lower() == "got":
            prompt = self._generate_got_prompt(query, context)
        else:  # standard
            prompt = self._generate_standard_prompt(query, context)

        # Generate response using Groq
        chat_completion = self.groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an advanced AI agent capable of complex reasoning and problem-solving."},
                {"role": "user", "content": prompt}
            ],
            model="llama3-70b-8192",
            temperature=0.3,
            top_p=0.9,
            max_tokens=1024
        )

        return chat_completion.choices[0].message.content

    def _generate_standard_prompt(self, query: str, context: str) -> str:
        """Generate standard agent prompt."""
        return f"""
        You are an advanced AI agent capable of complex reasoning and problem-solving. Your task is to analyze the question and context carefully to provide a detailed, well-reasoned answer.

        USER QUESTION:
        {query}

        RELEVANT CONTEXT:
        {context}

        Please provide a detailed, well-reasoned answer that:
        1. Analyzes the question thoroughly
        2. Uses the context effectively
        3. Provides additional insights where relevant
        4. Considers multiple perspectives
        5. Explains the reasoning process
        6. Identifies any gaps in the available information
        7. Suggests potential follow-up questions or areas for further exploration

        Your answer should be structured, clear, and demonstrate deep understanding of both the question and the context.
        """

    def _generate_cot_prompt(self, query: str, context: str) -> str:
        """Generate Chain of Thought prompt."""
        return f"""
        You are an advanced AI agent using Chain of Thought reasoning. Break down the problem and think step by step.

        USER QUESTION:
        {query}

        RELEVANT CONTEXT:
        {context}

        Let's think step by step:

        1. First, let's understand what the question is asking:
        [Your understanding of the question]

        2. Next, let's analyze the relevant context:
        [Your analysis of the context]

        3. Now, let's break down the reasoning process:
        [Step-by-step reasoning]

        4. Let's consider potential implications and connections:
        [Analysis of implications]

        5. Finally, let's formulate a comprehensive answer:
        [Your final answer]

        Remember to:
        - Show your work and reasoning at each step
        - Explain how you arrived at each conclusion
        - Consider alternative perspectives
        - Identify any assumptions or limitations
        """

    def _generate_tot_prompt(self, query: str, context: str) -> str:
        """Generate Tree of Thought prompt."""
        return f"""
        You are an advanced AI agent using Tree of Thought reasoning. Explore different branches of reasoning and evaluate them.

        USER QUESTION:
        {query}

        RELEVANT CONTEXT:
        {context}

        Let's explore different branches of reasoning:

        Branch 1: [First approach/interpretation]
        - Reasoning:
        - Evidence:
        - Strengths:
        - Weaknesses:

        Branch 2: [Alternative approach/interpretation]
        - Reasoning:
        - Evidence:
        - Strengths:
        - Weaknesses:

        Branch 3: [Another perspective]
        - Reasoning:
        - Evidence:
        - Strengths:
        - Weaknesses:

        Now, let's evaluate these branches:
        1. Compare the different approaches
        2. Identify the most promising path
        3. Combine insights from multiple branches
        4. Consider potential synthesis

        Final Answer:
        [Your comprehensive answer that synthesizes the best elements from different branches]
        """

    def _generate_got_prompt(self, query: str, context: str) -> str:
        """Generate Graph of Thought prompt."""
        return f"""
        You are an advanced AI agent using Graph of Thought reasoning. Create a network of interconnected ideas and explore their relationships.

        USER QUESTION:
        {query}

        RELEVANT CONTEXT:
        {context}

        Let's create a knowledge graph of interconnected concepts:

        Core Concepts:
        1. [Main concept 1]
           - Related ideas:
           - Connections:
           - Implications:

        2. [Main concept 2]
           - Related ideas:
           - Connections:
           - Implications:

        3. [Main concept 3]
           - Related ideas:
           - Connections:
           - Implications:

        Now, let's analyze the relationships:
        1. Direct connections between concepts
        2. Indirect relationships and implications
        3. Feedback loops and dependencies
        4. Emergent properties from the network

        Let's synthesize the insights:
        1. Key patterns in the relationships
        2. Most significant connections
        3. Potential new insights from the network
        4. Implications for the original question

        Final Answer:
        [Your comprehensive answer that leverages the interconnected nature of the concepts]
        """

    def query(self, question: str, video_url: str = "", prompting_technique: str = "standard") -> Dict[str, Any]:
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
        
        # Generate answer with selected prompting technique
        answer = self.generate_answer(question, relevant_chunks, video_context, prompting_technique)
        
        return {
            "answer": answer,
            "sources": relevant_chunks,
            "video_context": video_context
        }

# Initialize systems
def create_system(mode: str):
    if mode.lower() == "rag":
        return OriginalRAGSystem(
            pinecone_api_key="pcsk_7EKroD_MaZi2zjikyZTdpaDPCkit4qEAE6cjKuJ7C2ot9htS7EE6uurWQLrfznykMd7bW3",
            groq_api_key="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",
            index_name="embeddings"
        )
    else:
        return OriginalAgentSystem(
            pinecone_api_key="pcsk_7EKroD_MaZi2zjikyZTdpaDPCkit4qEAE6cjKuJ7C2ot9htS7EE6uurWQLrfznykMd7bW3",
            groq_api_key="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",
            index_name="embeddings"
        )

def process_query(question: str, video_url: str, mode: str, prompting_technique: str) -> tuple[str, str]:
    """Process the query and return answer and sources."""
    if not question.strip():
        return "Please enter a question.", "No sources available."
    
    try:
        system = create_system(mode)
        result = system.query(question, video_url, prompting_technique)
        
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
    # 🤖 Enhanced RAG & Agent Q&A System
    Choose between RAG or Agent mode and get answers based on the knowledge base and YouTube videos.
    """)
    
    with gr.Row():
        with gr.Column(scale=4):
            mode_radio = gr.Radio(
                choices=["RAG", "Agent"],
                value="RAG",
                label="Select Mode",
                info="RAG: Standard retrieval-augmented generation. Agent: Advanced reasoning with deeper analysis."
            )
            prompting_technique = gr.Radio(
                choices=["Standard", "CoT", "ToT", "GoT"],
                value="Standard",
                label="Prompting Technique",
                info="Standard: General-purpose reasoning. CoT: Chain of Thought. ToT: Tree of Thought. GoT: Graph of Thought"
            )
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
        inputs=[question_input, video_url_input, mode_radio, prompting_technique],
        outputs=[answer_output, sources_output]
    )
    
    gr.Markdown("""
    ### Tips:
    - RAG Mode: Best for straightforward questions with clear answers
    - Agent Mode: Best for complex questions requiring deeper analysis
    - Prompting Techniques:
      - Standard: General-purpose reasoning
      - CoT (Chain of Thought): Step-by-step reasoning
      - ToT (Tree of Thought): Multiple reasoning branches
      - GoT (Graph of Thought): Networked reasoning
    - Be specific in your questions
    - Questions should be related to the content in the knowledge base or video
    - You can optionally provide a YouTube video URL for additional context
    - The system will provide relevant sources along with the answer
    """)

# Launch the app
if __name__ == "__main__":
    demo.launch(share=True)  # share=True creates a public URL 