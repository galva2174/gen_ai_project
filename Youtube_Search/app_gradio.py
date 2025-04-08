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

        # Initialize VideoEmbeddingManager for expanding the knowledge base
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
            print("Warning: YOUTUBE_API_KEY not found in environment variables. YouTube search feature will be disabled.")
            self.youtube_available = False

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

    def search_relevant_video(self, query: str) -> Optional[str]:
        """Search for a relevant YouTube video based on the query using LLM for selection."""
        if not hasattr(self, 'youtube_available') or not self.youtube_available:
            print("YouTube API is not available. Skipping video search.")
            return None
            
        try:
            # Extract keywords from the query
            doc = self.nlp(query.lower())
            # Get nouns, verbs, and important words, excluding stop words
            keywords = [token.text for token in doc if not token.is_stop and 
                       (token.pos_ in ['NOUN', 'VERB', 'ADJ'] or len(token.text) > 3)]
            
            # If no keywords found, use the original query
            if not keywords:
                keywords = [query]
            
            # Create a search query with the most relevant keywords
            search_query = ' '.join(keywords[:3])  # Use top 3 keywords
            print(f"Searching YouTube for: {search_query}")
            
            # Search for videos with more specific parameters
            search_response = self.youtube.search().list(
                q=search_query,
                part='id,snippet',
                maxResults=5,  # Get more results to find the best match
                type='video',
                videoEmbeddable='true',  # Must be string 'true', not boolean True
                videoDuration='medium',
                order='relevance'  # Ensure most relevant results first
            ).execute()

            # Check if we found any videos
            if not search_response.get('items'):
                print("No videos found for the query.")
                return None
                
            # Get video statistics (including view count) for all results
            video_ids = [item['id']['videoId'] for item in search_response['items']]
            
            # Get video statistics including view counts
            videos_stats = self.youtube.videos().list(
                part='statistics,snippet',
                id=','.join(video_ids)
            ).execute()
            
            if not videos_stats.get('items'):
                print("Failed to retrieve video statistics.")
                # Fall back to using search results without statistics
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
                # Prepare video information for LLM analysis with view counts
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
                
                # Sort videos by view count (in descending order)
                videos_info.sort(key=lambda x: x['view_count'], reverse=True)
                print(f"Found {len(videos_info)} videos with statistics.")

            if not videos_info:
                print("No valid videos found.")
                return None

            # Use LLM to analyze and select the most relevant video
            formatted_videos = []
            for v in videos_info:
                view_info = f"\nViews: {v.get('view_count', 'unknown')}" if 'view_count' in v else ""
                formatted_videos.append(f"Title: {v['title']}\nDescription: {v['description']}{view_info}\nID: {v['id']}\n")
                
            prompt = f"""
            Analyze these YouTube videos and select the most relevant one for the query: "{query}"

            Videos:
            {formatted_videos}

            Consider:
            1. Relevance to the query (most important)
            2. Number of views (popularity) if available
            3. Quality of content based on title and description
            4. Educational value

            Return ONLY the video ID of the most relevant video, nothing else.
            """

            # Get LLM's selection
            print("Asking LLM to select the most relevant video...")
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a video selection assistant. Return only the video ID."},
                    {"role": "user", "content": prompt}
                ],
                model="llama3-70b-8192",
                temperature=0.3,  # Lower temperature for more focused selection
                top_p=0.95,
                max_tokens=20
            )

            selected_video_id = chat_completion.choices[0].message.content.strip()
            print(f"LLM selected video ID: {selected_video_id}")
            
            # Verify the selected video has a transcript
            try:
                print(f"Checking if selected video has transcript...")
                YouTubeTranscriptApi.get_transcript(selected_video_id)
                return f"https://www.youtube.com/watch?v={selected_video_id}"
            except Exception as transcript_error:
                print(f"Selected video doesn't have transcript: {transcript_error}")
                # If selected video has no transcript, try other videos
                for video in videos_info:
                    try:
                        print(f"Trying alternative video ID: {video['id']}")
                        YouTubeTranscriptApi.get_transcript(video['id'])
                        return f"https://www.youtube.com/watch?v={video['id']}"
                    except Exception as e:
                        print(f"No transcript for {video['id']}: {e}")
                        continue
                
                # If no video has transcript, return the LLM-selected video anyway
                print("No videos with transcripts found. Returning selected video anyway.")
                return f"https://www.youtube.com/watch?v={selected_video_id}"
        
        except HttpError as e:
            print(f"YouTube API HTTP error: {e.resp.status} {e.content}")
            return None
        except Exception as e:
            print(f"An error occurred while searching for videos: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def generate_answer(self, query: str, relevant_chunks: List[Dict[str, Any]], video_context: str = "", prompting_technique: str = "standard") -> str:
        """Generate a comprehensive answer using Groq."""
        if not relevant_chunks and not video_context:
            return "I couldn't find sufficient information to answer your question."

        # Combine context from both sources
        context = ""
        if relevant_chunks:
            context += f"Knowledge Base Context:\n{relevant_chunks[0]['text']}\n\n"
        if video_context:
            context += f"Video Context:\n{video_context}\n\n"

        # Prepare prompt based on selected technique
        if prompting_technique == "chain_of_thought":
            prompt = f"""
            You are an intelligent assistant specialized in educational content. Let's solve this step by step.

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
        elif prompting_technique == "tree_of_thought":
            prompt = f"""
            You are an intelligent assistant specialized in educational content. Let's explore different approaches to answer this question.

            USER QUESTION:
            {query}

            CONTEXT:
            {context}

            Let's consider multiple perspectives:
            Branch 1: Direct approach
            Branch 2: Alternative interpretation
            Branch 3: Combined approach

            After evaluating all branches, here's the most comprehensive answer:

            ANSWER:
            """
        elif prompting_technique == "graph_of_thought":
            prompt = f"""
            You are an intelligent assistant specialized in educational content. Let's analyze this question through interconnected concepts.

            USER QUESTION:
            {query}

            CONTEXT:
            {context}

            Let's map out the relationships:
            1) Core concepts from the context
            2) Related ideas and connections
            3) Synthesis of all connected information

            Based on this interconnected analysis, here's the answer:

            ANSWER:
            """
        else:  # standard prompting
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

    def query(self, question: str, video_url: str = "", prompting_technique: str = "standard") -> Dict[str, Any]:
        """Process a user query and return an answer with supporting evidence."""
        # Generate embedding
        query_embedding = self.embed_query(question)
        
        # Get relevant chunks from knowledge base
        relevant_chunks = self.retrieve_relevant_chunks(query_embedding)
        
        # Check if knowledge should be expanded with the video
        relevance_scores = [chunk['score'] for chunk in relevant_chunks]
        should_expand = False
        
        # Calculate average relevance score
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
        low_relevance = avg_relevance < 0.55  # Threshold between 0.5 and 0.6
        
        # Process video transcript
        video_context = ""
        if video_url:
            video_id = self.extract_video_id(video_url)
            if video_id:
                try:
                    video_context = self.get_video_transcript(video_id)
                    if not video_context:
                        print(f"No transcript available for user-provided video: {video_id}")
                    else:
                        # Check if knowledge base should be expanded with this video
                        should_expand = self.video_embedding_manager.should_expand_knowledge(relevance_scores)
                        if should_expand:
                            print(f"Knowledge base relevance is low ({avg_relevance}). Expanding knowledge with video: {video_id}")
                            # Process the video and add it to Pinecone
                            self.video_embedding_manager.process_video(video_url)
                            # Re-query Pinecone after adding the new video
                            relevant_chunks = self.retrieve_relevant_chunks(query_embedding)
                except Exception as e:
                    print(f"Error getting transcript for user-provided video: {str(e)}")
        elif low_relevance and hasattr(self, 'youtube_available') and self.youtube_available:
            # Only search for videos if relevance is low and no video URL is provided
            print(f"Knowledge base relevance is low ({avg_relevance}). Searching for relevant YouTube video...")
            relevant_video_url = self.search_relevant_video(question)
            if relevant_video_url:
                print(f"Found relevant video: {relevant_video_url}")
                video_id = self.extract_video_id(relevant_video_url)
                if video_id:
                    try:
                        video_context = self.get_video_transcript(video_id)
                        if not video_context:
                            print(f"No transcript available for found video: {video_id}")
                    except Exception as e:
                        print(f"Error getting transcript for found video: {str(e)}")
                    # Update the video_url parameter to include the found URL
                    video_url = relevant_video_url
            else:
                print("No relevant video found or no transcript available.")
        else:
            if not low_relevance:
                print(f"Knowledge base relevance is good ({avg_relevance}). Using only knowledge base for answering.")
            else:
                print("YouTube search is disabled. Using only knowledge base for answering.")
        
        # Generate answer
        answer = self.generate_answer(question, relevant_chunks, video_context, prompting_technique)
        
        return {
            "answer": answer,
            "sources": relevant_chunks,
            "video_context": video_context,
            "video_url": video_url,  # Include the video URL in the response
            "knowledge_expanded": should_expand  # Indicate if knowledge was expanded
        }

# Initialize RAG system
rag_system = EnhancedRAGSystem(
    pinecone_api_key=os.getenv('PINECONE_API_KEY','pcsk_7EKroD_MaZi2zjikyZTdpaDPCkit4qEAE6cjKuJ7C2ot9htS7EE6uurWQLrfznykMd7bW3'),
    groq_api_key=os.getenv('GROQ_API_KEY','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'),
    index_name=os.getenv('PINECONE_INDEX_NAME', 'embeddings')
)

def process_query(question: str, video_url: str = "", prompting_technique: str = "standard") -> tuple[str, str, str]:
    """Process the query and return answer, sources, and fetched URL."""
    if not question.strip():
        return "Please enter a question.", "No sources available.", ""
    
    try:
        result = rag_system.query(question, video_url, prompting_technique)
        
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
            
            # Add information about knowledge expansion if applicable
            if result.get("knowledge_expanded"):
                sources_text += "\n\n[Knowledge base was expanded with this video's transcript due to low relevance scores]"
        
        return result["answer"], sources_text, result["video_url"]
    except Exception as e:
        return f"Error: {str(e)}", "Error retrieving sources.", ""

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
            fetched_url_output = gr.Textbox(
                label="Automatically Fetched Video URL",
                placeholder="No video URL fetched yet...",
                lines=1,
                interactive=False
            )
            prompting_technique = gr.Dropdown(
                choices=["standard", "chain_of_thought", "tree_of_thought", "graph_of_thought"],
                value="standard",
                label="Prompting Technique",
                info="Choose how you want the AI to think through the answer"
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
        inputs=[question_input, video_url_input, prompting_technique],
        outputs=[answer_output, sources_output, fetched_url_output]
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