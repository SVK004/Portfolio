import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Robust File Loading for Vercel ---
def load_rag_data(file_name="rag_data.txt"):
    """Reads content from a data file, ensuring it's found in the Vercel environment."""
    # The current script's directory (e.g., /var/task/api)
    script_dir = os.path.dirname(__file__)
    # The project root (e.g., /var/task)
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    # Full path to the data file
    file_path = os.path.join(project_root, file_name)

    # Vercel copies files included via "includeFiles" into the root of the lambda
    # so we check there as a primary location.
    if os.path.exists(file_name):
        file_path = file_name
    elif not os.path.exists(file_path):
         print(f"CRITICAL ERROR: Data file not found at {file_path} or in the root. Make sure 'rag_data.txt' is in your project's root and included in vercel.json.")
         return ""

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            print(f"Successfully loaded data from: {file_path}")
            return f.read()
    except Exception as e:
        print(f"Error reading the data file: {e}")
        return ""

RAG_DATA_SOURCE = load_rag_data()
# --- End of File Loading ---

# Prepare the RAG data into chunks
knowledge_chunks = [chunk.strip() for chunk in RAG_DATA_SOURCE.strip().split('\n') if chunk.strip()] if RAG_DATA_SOURCE else []

# Configure the Gemini API client from environment variables
model = None
api_key = os.getenv("GEMINI_API_KEY")

if api_key:
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        print("Successfully configured Gemini API client.")
    except Exception as e:
        print(f"Error configuring Gemini API: {e}")
else:
    print("CRITICAL WARNING: GEMINI_API_KEY environment variable not found.")

# Pydantic model for the incoming request body
class Query(BaseModel):
    userQuery: str

# API endpoint for the chat
@app.post("/api/chat")
async def chat_endpoint(query: Query):
    user_query = query.userQuery

    # Pre-flight checks for a healthy server environment
    if not model:
        raise HTTPException(status_code=503, detail="Server Error: The Gemini API client is not configured. Check API key.")
    if not knowledge_chunks:
        raise HTTPException(status_code=503, detail="Server Error: The knowledge base is empty. Check rag_data.txt.")
    if not user_query:
        raise HTTPException(status_code=400, detail="User query cannot be empty.")

    try:
        # 1. Create a corpus with the user query and knowledge chunks
        corpus = [user_query] + knowledge_chunks
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(corpus)

        # 2. Find the most relevant chunks using cosine similarity
        cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        
        # Get the top 3 most similar document indices
        top_n_indices = cosine_similarities.argsort()[-3:][::-1]
        context = "\n".join([knowledge_chunks[i] for i in top_n_indices])

        # 3. Construct the prompt for the language model
        prompt = f"""Based ONLY on the following information, answer the user's question concisely. If the information is not present, say that you don't have enough information.

CONTEXT:
{context}

QUESTION:
{user_query}"""

        # 4. Generate content using the Gemini model
        response = await model.generate_content_async(prompt)
        
        return {"botResponse": response.text}

    except Exception as e:
        print(f"RAG Backend Error: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred while processing the request.")
