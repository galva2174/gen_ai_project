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
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from video_embeddings import VideoEmbeddingManager

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

class BaseSystem:
    def __init__(self, pinecone_api_key: str, groq_api_key: str, index_name: str,
                 model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        """
        Initialize base system with Pinecone, Groq, and models.
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

        # Initialize VideoEmbeddingManager
        self.video_embedding_manager = VideoEmbeddingManager(
            pinecone_api_key=pinecone_api_key,
            index_name=index_name,
            model_name=model_name
        )
        
        # Initialize YouTube API client
        youtube_api_key = os.getenv('YOUTUBE_API_KEY')
        if youtube_api_key:
            try:
                self.youtube = build('youtube', 'v3', developerKey=youtube_api_key)
                self.youtube_available = True
                print("YouTube API initialized successfully")
            except Exception as e:
                print(f"Warning: Failed to initialize YouTube API client. Error: {e}")
                self.youtube_available = False
        else:
            print("Warning: YOUTUBE_API_KEY not found in environment variables")
            self.youtube_available = False

        # Initialize NLP tools
        try:
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
        except Exception as e:
            print(f"Warning: NLTK resource download issue. Error: {e}")
            self.stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are'}
            self.lemmatizer = None

        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            print("Warning: spaCy model not found. Using simple pipeline.")
            self.nlp = spacy.blank("en")

        self.system_prompt = """You are an expert educational assistant specialized in providing clear, accurate, and engaging learning experiences. Your primary goals are:

1. Educational Excellence:
   - Provide accurate, well-researched information
   - Break down complex concepts into understandable parts
   - Use clear explanations and examples
   - Encourage critical thinking and deeper understanding

2. Learning Support:
   - Adapt explanations to the user's level of understanding
   - Provide relevant examples and analogies
   - Suggest additional learning resources when appropriate
   - Encourage active learning and engagement

3. Content Quality:
   - Ensure information is up-to-date and reliable
   - Present information in a structured, logical manner
   - Use appropriate educational methodologies
   - Maintain academic integrity and proper citations

4. Engagement:
   - Be encouraging and supportive
   - Ask questions to check understanding
   - Provide constructive feedback
   - Create a positive learning environment

Remember to:
- Always prioritize educational value and learning outcomes
- Be patient and supportive
- Encourage questions and deeper exploration
- Provide clear, structured explanations
- Use appropriate educational methodologies
- Maintain academic integrity
"""

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

    def search_relevant_video(self, query: str) -> Optional[str]:
        """Search for a relevant YouTube video based on the query."""
        if not hasattr(self, 'youtube_available') or not self.youtube_available:
            print("YouTube API is not available. Skipping video search.")
            return None
            
        try:
            # Extract keywords from the query
            doc = self.nlp(query.lower())
            keywords = [token.text for token in doc if not token.is_stop and 
                       (token.pos_ in ['NOUN', 'VERB', 'ADJ'] or len(token.text) > 3)]
            
            if not keywords:
                keywords = [query]
            
            search_query = ' '.join(keywords[:3])
            print(f"Searching YouTube for: {search_query}")
            
            search_response = self.youtube.search().list(
                q=search_query,
                part='id,snippet',
                maxResults=5,
                type='video',
                videoEmbeddable='true',
                videoDuration='medium',
                order='relevance'
            ).execute()

            if not search_response.get('items'):
                print("No videos found for the query.")
                return None
                
            video_ids = [item['id']['videoId'] for item in search_response['items']]
            
            videos_stats = self.youtube.videos().list(
                part='statistics,snippet',
                id=','.join(video_ids)
            ).execute()
            
            if not videos_stats.get('items'):
                print("Failed to retrieve video statistics.")
                videos_info = []
                for item in search_response['items']:
                    video_id = item['id']['videoId']
                    title = item['snippet']['title']
                    description = item['snippet']['description']
                    videos_info.append({
                        'id': video_id,
                        'title': title,
                        'description': description
                    })
            else:
                videos_info = []
                for item in videos_stats['items']:
                    video_id = item['id']
                    title = item['snippet']['title']
                    description = item['snippet']['description']
                    view_count = int(item['statistics'].get('viewCount', 0))
                    videos_info.append({
                        'id': video_id,
                        'title': title,
                        'description': description,
                        'view_count': view_count
                    })
                
                videos_info.sort(key=lambda x: x['view_count'], reverse=True)

            if not videos_info:
                print("No valid videos found.")
                return None

            formatted_videos = []
            for v in videos_info:
                view_info = f"\nViews: {v.get('view_count', 'unknown')}" if 'view_count' in v else ""
                formatted_videos.append(f"Title: {v['title']}\nDescription: {v['description']}{view_info}\nID: {v['id']}\n")
                
            prompt = f"""
            Analyze these YouTube videos and select the most relevant EDUCATIONAL video for the query: "{query}"

            Videos:
            {formatted_videos}

            Consider:
            1. Educational value and learning potential (most important)
            2. Content structure and teaching methodology
            3. Accuracy and reliability of information
            4. Engagement and clarity of explanation
            5. Relevance to the query
            6. Number of views (popularity) if available

            Return ONLY the video ID of the most relevant EDUCATIONAL video, nothing else.
            """

            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are an educational video selection assistant specialized in identifying high-quality educational content. Return only the video ID."},
                    {"role": "user", "content": prompt}
                ],
                model="llama3-70b-8192",
                temperature=0.3,
                top_p=0.95,
                max_tokens=20
            )

            selected_video_id = chat_completion.choices[0].message.content.strip()
            
            try:
                YouTubeTranscriptApi.get_transcript(selected_video_id)
                return f"https://www.youtube.com/watch?v={selected_video_id}"
            except Exception as transcript_error:
                print(f"Selected video doesn't have transcript: {transcript_error}")
                for video in videos_info:
                    try:
                        YouTubeTranscriptApi.get_transcript(video['id'])
                        return f"https://www.youtube.com/watch?v={video['id']}"
                    except Exception as e:
                        continue
                
                return f"https://www.youtube.com/watch?v={selected_video_id}"
        
        except HttpError as e:
            print(f"YouTube API HTTP error: {e.resp.status} {e.content}")
            return None
        except Exception as e:
            print(f"An error occurred while searching for videos: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

class EnhancedRAGSystem(BaseSystem):
    def generate_answer(self, query: str, relevant_chunks: List[Dict[str, Any]], video_context: str = "", prompting_technique: str = "standard") -> str:
        """Generate a comprehensive answer using Groq."""
        if not relevant_chunks and not video_context:
            return "I couldn't find sufficient information to answer your question."

        context = ""
        if relevant_chunks:
            context += f"Knowledge Base Context:\n{relevant_chunks[0]['text']}\n\n"
        if video_context:
            context += f"Video Context:\n{video_context}\n\n"

        if prompting_technique.lower() == "cot":
            prompt = self._generate_cot_prompt(query, context)
        elif prompting_technique.lower() == "tot":
            prompt = self._generate_tot_prompt(query, context)
        elif prompting_technique.lower() == "got":
            prompt = self._generate_got_prompt(query, context)
        else:
            prompt = self._generate_standard_prompt(query, context)

        chat_completion = self.groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            model="llama3-70b-8192",
            temperature=0.7,
            top_p=0.95,
            max_tokens=512
        )

        return chat_completion.choices[0].message.content

    def _generate_standard_prompt(self, query: str, context: str) -> str:
        return f"""
        You are an intelligent assistant specialized in educational content. Answer the question using the provided context.

        USER QUESTION:
        {query}

        CONTEXT:
        {context}

        Please provide a clear, concise answer based on the context. If the context doesn't contain enough information, say so.
        """

    def _generate_cot_prompt(self, query: str, context: str) -> str:
        return f"""
        You are an intelligent assistant using Chain of Thought reasoning. Let's solve this step by step.

        USER QUESTION:
        {query}

        CONTEXT:
        {context}

        Let's think through this:
        1) First, let's understand what we know from the context
        2) Then, let's break down the question
        3) Finally, let's combine this information to form a complete answer

        ANSWER:
        """

    def _generate_tot_prompt(self, query: str, context: str) -> str:
        return f"""
        You are an intelligent assistant using Tree of Thought reasoning. Let's explore different approaches.

        USER QUESTION:
        {query}

        CONTEXT:
        {context}

        Let's consider multiple perspectives:
        Branch 1: Direct approach
        Branch 2: Alternative interpretation
        Branch 3: Combined approach

        After evaluating all branches, here's the most comprehensive answer:
        """

    def _generate_got_prompt(self, query: str, context: str) -> str:
        return f"""
        You are an intelligent assistant using Graph of Thought reasoning. Let's analyze through interconnected concepts.

        USER QUESTION:
        {query}

        CONTEXT:
        {context}

        Let's map out the relationships:
        1) Core concepts from the context
        2) Related ideas and connections
        3) Synthesis of all connected information

        Based on this interconnected analysis, here's the answer:
        """

class AgentSystem(BaseSystem):
    def generate_answer(self, query: str, relevant_chunks: List[Dict[str, Any]], video_context: str = "", prompting_technique: str = "standard") -> str:
        """Generate a comprehensive answer using Groq with agent capabilities."""
        if not relevant_chunks and not video_context:
            return "I couldn't find sufficient information to answer your question."

        context = ""
        if relevant_chunks:
            context += f"Knowledge Base Context:\n{relevant_chunks[0]['text']}\n\n"
        if video_context:
            context += f"Video Context:\n{video_context}\n\n"

        if prompting_technique.lower() == "cot":
            prompt = self._generate_cot_prompt(query, context)
        elif prompting_technique.lower() == "tot":
            prompt = self._generate_tot_prompt(query, context)
        elif prompting_technique.lower() == "got":
            prompt = self._generate_got_prompt(query, context)
        else:
            prompt = self._generate_standard_prompt(query, context)

        chat_completion = self.groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            model="llama3-70b-8192",
            temperature=0.3,
            top_p=0.9,
            max_tokens=1024
        )

        return chat_completion.choices[0].message.content

    def _generate_standard_prompt(self, query: str, context: str) -> str:
        return f"""
        You are an advanced AI agent capable of complex reasoning and problem-solving. Analyze the question and context carefully.

        USER QUESTION:
        {query}

        CONTEXT:
        {context}

        Please provide a detailed, well-reasoned answer that:
        1. Analyzes the question thoroughly
        2. Uses the context effectively
        3. Provides additional insights
        4. Considers multiple perspectives
        5. Explains the reasoning process
        6. Identifies information gaps
        7. Suggests follow-up areas

        Your answer should be structured, clear, and demonstrate deep understanding.
        """

    def _generate_cot_prompt(self, query: str, context: str) -> str:
        return f"""
        You are an advanced AI agent using Chain of Thought reasoning. Break down the problem and think step by step.

        USER QUESTION:
        {query}

        CONTEXT:
        {context}

        Let's analyze systematically:
        1. Question Analysis:
           - Core components
           - Key requirements
           - Hidden assumptions

        2. Context Evaluation:
           - Relevant information
           - Missing elements
           - Potential implications

        3. Solution Development:
           - Initial approach
           - Alternative perspectives
           - Synthesis and integration

        4. Verification and Refinement:
           - Logic check
           - Completeness assessment
           - Clarity enhancement

        Based on this analysis, here's the comprehensive answer:
        """

    def _generate_tot_prompt(self, query: str, context: str) -> str:
        return f"""
        You are an advanced AI agent using Tree of Thought reasoning. Explore multiple solution paths.

        USER QUESTION:
        {query}

        CONTEXT:
        {context}

        Let's explore different solution paths:

        Path 1: Analytical Approach
        - Initial assumptions
        - Logical steps
        - Conclusions
        - Limitations

        Path 2: Contextual Approach
        - Background considerations
        - Contextual implications
        - Practical applications
        - Trade-offs

        Path 3: Integrative Approach
        - Cross-domain connections
        - Synthesis opportunities
        - Novel insights
        - Future implications

        After evaluating all paths, here's the optimal solution:
        """

    def _generate_got_prompt(self, query: str, context: str) -> str:
        return f"""
        You are an advanced AI agent using Graph of Thought reasoning. Map and connect concepts.

        USER QUESTION:
        {query}

        CONTEXT:
        {context}

        Let's create a conceptual network:

        1. Core Concepts:
           - Primary elements
           - Key relationships
           - Critical dependencies

        2. Contextual Framework:
           - Environmental factors
           - System dynamics
           - Emerging patterns

        3. Integration Analysis:
           - Cross-concept impacts
           - Feedback loops
           - Emergent properties

        4. Solution Synthesis:
           - Pattern recognition
           - Insight generation
           - Practical implications

        Based on this network analysis, here's the comprehensive solution:
        """

def create_system(mode: str):
    """Create and return the appropriate system based on mode."""
    if mode.lower() == "rag":
        return EnhancedRAGSystem(
            pinecone_api_key=os.getenv('PINECONE_API_KEY'),
            groq_api_key=os.getenv('GROQ_API_KEY'),
            index_name=os.getenv('PINECONE_INDEX_NAME', 'embeddings')
        )
    else:
        return AgentSystem(
            pinecone_api_key=os.getenv('PINECONE_API_KEY'),
            groq_api_key=os.getenv('GROQ_API_KEY'),
            index_name=os.getenv('PINECONE_INDEX_NAME', 'embeddings')
        )

def process_query(question: str, video_url: str = "", mode: str = "RAG", prompting_technique: str = "standard") -> tuple[str, str, str]:
    """Process the query and return answer, sources, and fetched URL."""
    if not question.strip():
        return "Please enter a question.", "No sources available.", ""
    
    try:
        system = create_system(mode)
        
        # Generate embedding
        query_embedding = system.embed_query(question)
        
        # Get relevant chunks from knowledge base
        relevant_chunks = system.retrieve_relevant_chunks(query_embedding)
        
        # Check if knowledge should be expanded with the video
        relevance_scores = [chunk['score'] for chunk in relevant_chunks]
        should_expand = False
        
        # Calculate average relevance score
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
        low_relevance = avg_relevance < 0.55
        
        # Process video transcript
        video_context = ""
        if video_url:
            video_id = system.extract_video_id(video_url)
            if video_id:
                try:
                    video_context = system.get_video_transcript(video_id)
                    if not video_context:
                        print(f"No transcript available for video: {video_id}")
                    else:
                        should_expand = system.video_embedding_manager.should_expand_knowledge(relevance_scores)
                        if should_expand:
                            print(f"Expanding knowledge with video as relevance score of existing knowledge base is low: {video_id}")
                            system.video_embedding_manager.process_video(video_url)
                            relevant_chunks = system.retrieve_relevant_chunks(query_embedding)
                except Exception as e:
                    print(f"Error getting transcript: {str(e)}")
        elif low_relevance and hasattr(system, 'youtube_available') and system.youtube_available:
            print(f"Knowledge base relevance is low ({avg_relevance}). Searching for video...")
            relevant_video_url = system.search_relevant_video(question)
            if relevant_video_url:
                print(f"Found relevant video: {relevant_video_url}")
                video_id = system.extract_video_id(relevant_video_url)
                if video_id:
                    try:
                        video_context = system.get_video_transcript(video_id)
                        if not video_context:
                            print(f"No transcript available for video: {video_id}")
                    except Exception as e:
                        print(f"Error getting transcript: {str(e)}")
                    video_url = relevant_video_url
            else:
                print("No relevant video found.")
        
        # Generate answer
        answer = system.generate_answer(question, relevant_chunks, video_context, prompting_technique)
        
        # Format sources
        sources_text = ""
        if relevant_chunks:
            sources_text += "Knowledge Base Sources:\n"
            sources_text += "\n\n".join([
                f"Source (Relevance: {source['score']:.2f}):\n{source['text']}"
                for source in relevant_chunks
            ])
        
        if video_context:
            if sources_text:
                sources_text += "\n\n"
            sources_text += "Video Context:\n" + video_context
            
            if should_expand:
                sources_text += "\n\n[Knowledge base was expanded with this video as relevance score of existing knowledge base is low]"
        
        return answer, sources_text, video_url
    except Exception as e:
        return f"Error: {str(e)}", "Error retrieving sources.", ""

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
                choices=["standard", "cot", "tot", "got"],
                value="standard",
                label="Prompting Technique",
                info="standard: Basic reasoning, cot: Chain of Thought, tot: Tree of Thought, got: Graph of Thought"
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
            fetched_url_output = gr.Textbox(
                label="Automatically Fetched Video URL",
                placeholder="No video URL fetched yet...",
                lines=1,
                interactive=False
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
        outputs=[answer_output, sources_output, fetched_url_output]
    )
    
    gr.Markdown("""
    ### Tips:
    - RAG Mode: Best for straightforward questions with clear answers
    - Agent Mode: Best for complex questions requiring deeper analysis
    - Prompting Techniques:
      - standard: Basic reasoning for straightforward questions
      - cot (Chain of Thought): Step-by-step reasoning
      - tot (Tree of Thought): Multiple reasoning branches
      - got (Graph of Thought): Networked concept analysis
    - Be specific in your questions
    - Questions should be related to the content in the knowledge base or video
    - You can optionally provide a YouTube video URL for additional context
    - The system will provide relevant sources along with the answer
    """)

# Launch the app
if __name__ == "__main__":
    demo.launch(share=True)  # share=True creates a public URL