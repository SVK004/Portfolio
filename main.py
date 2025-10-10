import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables
load_dotenv()

# --- FINAL, ROBUST FILE LOADING LOGIC ---
def load_rag_data(file_name="rag_data.txt"):
    """Reads content from a data file located in the project root."""
    # Get the directory of the current script (e.g., /.../project/api)
    script_dir = os.path.dirname(__file__)
    # Go up one level to the project root (e.g., /.../project)
    # project_root = os.path.abspath(os.path.join(script_dir, '..'))
    # Construct the full path to the data file
    # file_path = os.path.join(project_root, file_name)

    try:
        with open(file_name, "r", encoding="utf-8") as f:
            print(f"Successfully loaded data from: {file_name}")
            return f.read()
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Could not find the data file at {file_name}. Make sure 'rag_data.txt' is in your project's root directory.")
        return ""

RAG_DATA_SOURCE = load_rag_data()
# --- END OF FILE LOADING LOGIC ---

# Prepare the RAG data
knowledge_chunks = [chunk.strip() for chunk in RAG_DATA_SOURCE.strip().split('\n') if chunk.strip()] if RAG_DATA_SOURCE else []

# Configure the Gemini API client
model = None
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        print("Successfully configured Gemini API.")
    except Exception as e:
        print(f"Error configuring Gemini API: {e}")
else:
    print("Warning: GEMINI_API_KEY not found in environment variables.")

# Pydantic model for the request body
class Query(BaseModel):
    userQuery: str

# API endpoint for the chat
@app.post("/api/chat")
async def chat_endpoint(query: Query):
    user_query = query.userQuery
    if not model:
        raise HTTPException(status_code=500, detail="Gemini API is not configured on the server.")
    if not user_query:
        raise HTTPException(status_code=400, detail="Query is required.")
    if not knowledge_chunks:
        raise HTTPException(status_code=500, detail="Knowledge base is empty. Check if rag_data.txt is present and not empty.")
    try:
        query_words = set(re.findall(r'\w+', user_query.lower()))

        # Score each chunk based on the number of matching words
        scored_chunks = []
        for i, chunk in enumerate(knowledge_chunks):
            chunk_words = set(re.findall(r'\w+', chunk.lower()))
            score = len(query_words.intersection(chunk_words))
            if score > 0:
                scored_chunks.append((score, i))
        
        # Sort chunks by score in descending order and get the top 3
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        top_n_indices = [index for score, index in scored_chunks[:3]]
        
        if not top_n_indices:
             # Fallback if no words match at all
            context = RAG_DATA_SOURCE
        else:
            context = "\n".join([knowledge_chunks[i] for i in top_n_indices])
        prompt = f"""Based ONLY on the following information, answer the user's question concisely...\n\nCONTEXT:\n{context}\n\nQUESTION:\n{user_query}"""
        response = await model.generate_content_async(prompt)
        return {"botResponse": response.text}
    except Exception as e:
        print(f"RAG Backend Error: {e}")
        raise HTTPException(status_code=500, detail="Failed to process the request.")