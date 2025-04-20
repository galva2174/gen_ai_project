import os
import json
import random
from pinecone import Pinecone
from tqdm import tqdm
from groq import Groq  # or use OpenAI, HuggingFace, etc.

# --- Setup ---

# Pinecone setup
PINECONE_API_KEY = "pcsk_7EKroD_MaZi2zjikyZTdpaDPCkit4qEAE6cjKuJ7C2ot9htS7EE6uurWQLrfznykMd7bW3"
INDEX_NAME = "embeddings"

# Initialize Pinecone
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    # Connect to your index
    index = pc.Index(INDEX_NAME)
    print(f"Successfully connected to index: {INDEX_NAME}")
    
    # Get index stats
    stats = index.describe_index_stats()
    total_vectors = stats.total_vector_count
    vector_dimension = stats.dimension
    print(f"Found {total_vectors} vectors with dimension {vector_dimension}")
    
    USE_SAMPLE_DATA = False
except Exception as e:
    print(f"Error connecting to Pinecone: {e}")
    print("Falling back to sample data...")
    USE_SAMPLE_DATA = True

# Sample texts as fallback
sample_texts = [
    "The mitochondria is the powerhouse of the cell. It generates most of the cell's supply of adenosine triphosphate (ATP), used as a source of chemical energy.",
    "Newton's Third Law states that for every action, there is an equal and opposite reaction. This fundamental principle helps explain how rockets work and why we feel forces in our daily lives.",
    "Climate change is causing global temperatures to rise, leading to melting ice caps, rising sea levels, and more frequent extreme weather events.",
    "The water cycle describes how water evaporates from the surface of the earth, rises into the atmosphere, cools and condenses into rain or snow, and falls again to the surface.",
    "Photosynthesis is the process used by plants, algae and certain bacteria to harness energy from sunlight and turn it into chemical energy."
]

# Groq API setup
GROQ_API_KEY = "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"
groq_client = Groq(api_key=GROQ_API_KEY)
output_path = "pinecone_qa_dataset.jsonl"
  # Limit the number of records to process

# --- Helper to Generate Question ---
def generate_question(text_sample: str) -> str:
    prompt = f"""
Given the following educational text, create a question that a student might ask to elicit this information.

Text:
{text_sample}

Return only the question.
"""
    try:
        chat = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            temperature=0.5,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant generating educational questions."},
                {"role": "user", "content": prompt}
            ]
        )
        return chat.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating question: {e}")
        return None

# --- Main Processing Function ---
def process_text_sample(text_sample, fout):
    """Process a single text sample to generate a question and write to output file"""
    if not text_sample or len(text_sample.strip()) < 50:  # Skip very short texts
        return False
        
    question = generate_question(text_sample)
    if not question:
        return False
        
    chatml = {
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": text_sample}
        ]
    }
    fout.write(json.dumps(chatml) + "\n")
    print(f"Generated Q: {question}")
    print(f"Answer: {text_sample[:50]}...\n")
    return True

# --- Generate + Save ---
processed_count = 0
with open(output_path, "w") as fout:
    if USE_SAMPLE_DATA:
        print("Using sample data...")
        for text_sample in tqdm(sample_texts):
            if process_text_sample(text_sample, fout):
                processed_count += 1
    else:
        print("Using Pinecone data...")
        try:
            # Generate a random vector for querying
            random_vector = [random.uniform(-1, 1) for _ in range(vector_dimension)]
            
            # Query the index
            print(f"Querying index with random vector...")
            query_response = index.query(
                vector=random_vector,
                top_k=min(100, total_vectors),
                include_metadata=True
            )
            
            if hasattr(query_response, 'matches') and query_response.matches:
                print(f"Found {len(query_response.matches)} matches")
                
                # Process each match
                for match in tqdm(query_response.matches):
                    
                        
                    # Extract text from metadata
                    if hasattr(match, 'metadata') and match.metadata:
                        metadata = match.metadata
                        
                        # Look for text_sample in the metadata
                        text_sample = None
                        if isinstance(metadata, dict) and 'text_sample' in metadata:
                            text_sample = metadata['text_sample']
                            
                        if text_sample:
                            if process_text_sample(text_sample, fout):
                                processed_count += 1
                
            else:
                print("No matches found in query response")
                
        except Exception as e:
            print(f"Error querying Pinecone: {e}")
            print("Falling back to sample data...")
            
            # Fallback to sample data
            for text_sample in tqdm(sample_texts):
                if process_text_sample(text_sample, fout):
                    processed_count += 1

print(f"✅ Finished! Processed {processed_count} records. Dataset saved to: {output_path}") 