import streamlit as st
import numpy as np
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

# Initialize Pinecone
pc = Pinecone(api_key="pcsk_3rWW1w_Eua9C9tD1rbQybpChVD9nDijUycon7auXNs3afy7T2Z2zK2YnSHEFeLmKJsx4pp", region="us-east-1")

# Connect to or create your Pinecone index
index_name = "video-embeddings"
index = pc.Index(index_name)

# Load Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_transcript(video_id):
    """Extracts transcript from a YouTube video."""
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        formatter = TextFormatter()
        return formatter.format_transcript(transcript)
    except Exception as e:
        st.error(f"Error fetching transcript: {e}")
        return None

def get_embedding(text):
    """Generates an embedding using Sentence Transformers."""
    return model.encode(text).tolist()

def generate_embeddings(video_id):
    """Generates embeddings from a YouTube video transcript."""
    transcript = extract_transcript(video_id)
    if not transcript:
        return None, None

    lines = transcript.split("\n")
    chunk_size = 5  # Adjust chunk size
    text_chunks = [" ".join(lines[i:i+chunk_size]) for i in range(0, len(lines), chunk_size)]

    embeddings = np.array([get_embedding(chunk) for chunk in text_chunks], dtype=np.float32)
    return embeddings, text_chunks  # Return both embeddings and text chunks

def store_embeddings_in_pinecone(embeddings, text_chunks, video_id):
    """Stores embeddings in Pinecone index."""
    if embeddings is not None:
        ids = [f"{video_id}_{i}" for i in range(len(embeddings))]

        # Prepare data for Pinecone (storing text as metadata)
        data = [
            (ids[i], embeddings[i].tolist(), {"text": text_chunks[i]})
            for i in range(len(embeddings))
        ]

        # Upsert the embeddings into Pinecone
        index.upsert(vectors=data)
        st.success(f"Successfully upserted {len(data)} embeddings into Pinecone.")

def search_transcript(user_query, top_k=5):
    """Searches Pinecone for relevant transcript sections based on a user query."""
    
    # Step 1: Convert query to embedding
    query_embedding = get_embedding(user_query)

    # Step 2: Search Pinecone
    result = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)

    # Step 3: Extract relevant text sections
    retrieved_texts = [match.metadata["text"] for match in result["matches"] if match.metadata and "text" in match.metadata]

    return retrieved_texts

# Streamlit UI
def main():
    st.title("YouTube Video Transcript Search")

    # Input for YouTube video ID
    yt_video_id = st.text_input("Enter YouTube Video ID", "")

    # Input for user query
    user_query = st.text_input("Enter your question", "")

    if yt_video_id and user_query:
        st.write(f"Searching for: `{user_query}` in video `{yt_video_id}`...")

        # Generate embeddings and text chunks
        embeddings, text_chunks = generate_embeddings(yt_video_id)

        if embeddings is not None and len(embeddings) > 0 and text_chunks:

            # Store embeddings in Pinecone
            store_embeddings_in_pinecone(embeddings, text_chunks, yt_video_id)

            # Retrieve most relevant transcript sections
            relevant_sections = search_transcript(user_query)

            st.subheader("🔹 Most Relevant Transcript Sections:")
            if relevant_sections:
                for idx, section in enumerate(relevant_sections, 1):
                    st.write(f"{idx}. {section}\n")
            else:
                st.write("No matching transcript sections found.")
        else:
            st.write("Failed to generate embeddings. Please check the YouTube video ID.")
    else:
        st.write("Please enter a YouTube video ID and a query to start the search.")

if __name__ == "__main__":
    main()
