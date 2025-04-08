import numpy as np
import torch
from youtube_transcript_api import YouTubeTranscriptApi
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
import os
from dotenv import load_dotenv
from typing import List, Dict, Any, Tuple, Optional

# Load environment variables
load_dotenv()

class VideoEmbeddingManager:
    def __init__(self, pinecone_api_key: str = None, index_name: str = None, 
                 model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        """Initialize the Video Embedding Manager with Pinecone and embedding model."""
        # Check for GPU availability
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"VideoEmbeddingManager using device: {self.device}")

        # Initialize Pinecone
        pinecone_api_key = pinecone_api_key or os.getenv('PINECONE_API_KEY')
        index_name = index_name or os.getenv('PINECONE_INDEX_NAME', 'embeddings')
        
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index = self.pc.Index(index_name)
        
        # Load embedding model
        self.model = SentenceTransformer(model_name).to(self.device)
        
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
        except Exception as e:
            print(f"Warning: spaCy model not found. Using simple pipeline. Error: {e}")
            self.nlp = spacy.blank("en")

    def extract_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from YouTube URL."""
        patterns = [
            r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([^&\n?]+)',
            r'youtube\.com\/embed\/([^&\n?]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None

    def extract_transcript(self, video_id: str) -> Optional[str]:
        """Extract transcript from a YouTube video."""
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            full_transcript = ' '.join([segment['text'] for segment in transcript_list])
            return full_transcript
        except Exception as e:
            print(f"Error fetching transcript: {e}")
            return None

    def simple_tokenize(self, text: str) -> List[str]:
        """Simple tokenizer that avoids NLTK's punkt."""
        text = re.sub(r'[^\w\s]', ' ', text)
        return [token for token in text.lower().split() if token]

    def preprocess_text(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """NLP preprocessing: stopword removal, lemmatization, NER, POS tagging."""
        text = re.sub(r'[^\w\s]', ' ', text).lower()
        tokens = self.simple_tokenize(text)
        filtered_tokens = [word for word in tokens if word not in self.stop_words]
        
        lemmatized_tokens = []
        if self.lemmatizer:
            lemmatized_tokens = [self.lemmatizer.lemmatize(word) for word in filtered_tokens]
        else:
            lemmatized_tokens = filtered_tokens

        doc = self.nlp(' '.join(lemmatized_tokens))
        named_entities = [ent.text for ent in doc.ents]
        pos_counts = {}
        for token in doc:
            pos_counts[token.pos_] = pos_counts.get(token.pos_, 0) + 1

        processed_text = ' '.join(lemmatized_tokens)

        metadata = {
            "named_entities": named_entities[:10] if named_entities else [],
            "top_pos_tags": [f"{pos}:{count}" for pos, count in sorted(pos_counts.items(), key=lambda x: x[1], reverse=True)[:5]]
        }
        return processed_text, metadata

    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding using Sentence Transformers."""
        return self.model.encode(text, convert_to_tensor=True).cpu().tolist()

    def chunk_text(self, text: str, max_chunk_size: int = 150, min_chunk_size: int = 50, 
                  max_chunks: Optional[int] = None) -> List[str]:
        """Split text into chunks based on semantic boundaries while respecting size constraints."""
        # Normalize whitespace and split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())

        chunks = []
        current_chunk = []
        current_chunk_size = 0

        for sentence in sentences:
            # Count words in the sentence
            sentence_words = sentence.split()
            sentence_word_count = len(sentence_words)

            # If adding this sentence would exceed max chunk size, finalize current chunk
            if current_chunk_size + sentence_word_count > max_chunk_size:
                # Join and add current chunk if it's not empty
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_chunk_size = 0

            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_chunk_size += sentence_word_count

            # If chunk is getting too large, force a split
            if current_chunk_size >= max_chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_chunk_size = 0

            # Stop if we've reached max chunks
            if max_chunks and len(chunks) >= max_chunks:
                break

        # Add any remaining chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        # Handle edge cases
        if not chunks:
            chunks = [text]

        return chunks

    def generate_embeddings(self, video_id: str) -> Tuple[Optional[np.ndarray], Optional[List[str]], Optional[List[Dict]]]:
        """Generate embeddings from YouTube video transcript with NLP preprocessing."""
        transcript = self.extract_transcript(video_id)
        if not transcript:
            return None, None, None

        text_chunks = self.chunk_text(transcript)
        print(f"Created {len(text_chunks)} text chunks from transcript")

        embeddings, original_chunks, nlp_metadata_list = [], [], []
        for chunk in text_chunks:
            if len(chunk.strip()) > 10:
                processed_chunk, nlp_metadata = self.preprocess_text(chunk)
                embedding = self.get_embedding(processed_chunk)
                embeddings.append(embedding)
                original_chunks.append(chunk)
                nlp_metadata_list.append(nlp_metadata)

        print(f"Generated {len(embeddings)} embeddings with NLP preprocessing")
        if embeddings:
            return np.array(embeddings), original_chunks, nlp_metadata_list
        return None, None, None

    def store_embeddings_in_pinecone(self, embeddings: np.ndarray, video_id: str, 
                                    original_chunks: List[str] = None, 
                                    nlp_metadata_list: List[Dict] = None) -> bool:
        """Store embeddings in Pinecone index with valid metadata."""
        if embeddings is not None and len(embeddings) > 0:
            ids = [f"{video_id}_{i}" for i in range(len(embeddings))]
            vectors = []

            for i, emb in enumerate(embeddings):
                metadata = {"video_id": video_id, "chunk_id": i, "source_type": "youtube_video"}
                if original_chunks and i < len(original_chunks):
                    metadata["text_sample"] = original_chunks[i][:500] + "..." if len(original_chunks[i]) > 500 else original_chunks[i]

                if nlp_metadata_list and i < len(nlp_metadata_list):
                    nlp_meta = nlp_metadata_list[i]
                    metadata["named_entities"] = nlp_meta["named_entities"]
                    metadata["top_pos_tags"] = nlp_meta["top_pos_tags"] if nlp_meta["top_pos_tags"] else []

                vectors.append({"id": ids[i], "values": emb.tolist(), "metadata": metadata})

            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)
                print(f"Upserted batch {i//batch_size + 1}/{(len(vectors)-1)//batch_size + 1} to Pinecone")

            print(f"Successfully upserted {len(vectors)} embeddings into Pinecone.")
            return True
        return False

    def process_video(self, video_url: str) -> bool:
        """Process a YouTube video and store its transcript embeddings in Pinecone."""
        video_id = self.extract_video_id(video_url)
        if not video_id:
            print(f"Invalid YouTube URL: {video_url}")
            return False

        print(f"Processing YouTube video: {video_id}")
        embeddings, original_chunks, nlp_metadata_list = self.generate_embeddings(video_id)

        if embeddings is not None and len(embeddings) > 0:
            print(f"Generated {len(embeddings)} embedding chunks")
            return self.store_embeddings_in_pinecone(embeddings, video_id, original_chunks, nlp_metadata_list)
        else:
            print("Failed to generate embeddings")
            return False

    def should_expand_knowledge(self, relevance_scores: List[float], threshold: float = 0.4) -> bool:
        """Determine if knowledge base should be expanded based on relevance scores."""
        if not relevance_scores:
            return True
        
        # Calculate the average score of top matches
        avg_score = sum(relevance_scores) / len(relevance_scores)
        return avg_score < threshold


# Example usage
if __name__ == "__main__":
    video_manager = VideoEmbeddingManager()
    video_url = "https://www.youtube.com/watch?v=NUy_wOxOM8E"
    video_manager.process_video(video_url) 