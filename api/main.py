import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
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

# --- ROBUST FILE LOADING FOR VERCEL ---
def load_rag_data(file_name="rag_data.txt"):
    """Reads content from a data file, designed to work reliably in Vercel."""
    # The script runs inside the /api directory in Vercel
    script_dir = os.path.dirname(__file__) 
    # Go up one level to the project root to find the data file
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    file_path = os.path.join(project_root, file_name)

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            print(f"Successfully loaded data from: {file_path}")
            return f.read()
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Could not find the data file at {file_path}. Ensure 'rag_data.txt' is in the root and included in vercel.json.")
        return ""
# --- END OF FILE LOADING ---

RAG_DATA_SOURCE = load_rag_data()
knowledge_chunks = [line.strip() for line in RAG_DATA_SOURCE.splitlines() if line.strip()]

# --- Gemini API Setup ---
api_key = os.getenv("GEMINI_API_KEY")
model = None
if api_key:
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        print("Successfully configured Gemini API.")
    except Exception as e:
        print(f"Error configuring Gemini API: {e}")
else:
    print("Warning: GEMINI_API_KEY not found in environment variables.")


# Pydantic model for the request body
class Query(BaseModel):
    userQuery: str

# API endpoint for the chat - Made synchronous to prevent event loop issues
@app.post("/api/chat")
def chat_endpoint(query: Query):
    user_query = query.userQuery
    if not model:
        raise HTTPException(status_code=503, detail="Gemini API model is not initialized on the server.")
    if not user_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    if not knowledge_chunks:
        raise HTTPException(status_code=503, detail="Knowledge base is empty. Check if rag_data.txt was loaded correctly.")

    try:
        user_query_lower = user_query.lower()
        context = ""
        found_category = False

        # 1. Define categories and their keywords/file prefixes
        category_map = {
            'Project:': ('project', 'projects', 'work', 'built', 'develop'),
            'Skills in': ('skill', 'skills', 'knows', 'proficient', 'technologies'),
            'Education:': ('education', 'school', 'college', 'degree', 'study'),
            'Work Experience:': ('experience', 'intern', 'internship', 'job'),
            'Internship Task:': ('task', 'tasks', 'duties', 'responsibilities')
        }

        # 2. Check if the user is asking for a specific category
        for prefix, keywords in category_map.items():
            if any(keyword in user_query_lower for keyword in keywords):
                # If a category keyword is found, gather ALL matching lines
                matching_chunks = [chunk for chunk in knowledge_chunks if chunk.strip().startswith(prefix)]
                if matching_chunks:
                    context = "\n".join(matching_chunks)
                    found_category = True
                    break # Stop after finding the first relevant category

        # 3. If no specific category was found, fall back to the general keyword search
        if not found_category:
            stop_words = set(["a", "an", "the", "in", "on", "of", "what", "is", "who", "where", "when", "tell", "me", "about", "for", "this", "that", "with", "by", "has"])
            meaningful_words = [word for word in user_query_lower.split() if word not in stop_words]
            
            ranked_chunks = sorted(
                knowledge_chunks,
                key=lambda chunk: sum((word in chunk.lower() or (word.endswith('s') and word[:-1] in chunk.lower())) for word in meaningful_words),
                reverse=True
            )
            # Use a larger context window for general questions
            context = "\n".join(ranked_chunks[:5])

        prompt = f"""Based ONLY on the following information, answer the user's question concisely.
If the information required to answer the question is not in the context, say that you do not have enough information to answer.

CONTEXT:
{context}

QUESTION:
{user_query}"""

        # Switched to the synchronous SDK call
        response = model.generate_content(prompt)
        return {"botResponse": response.text}
    except Exception as e:
        print(f"RAG Backend Error: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred while processing the request.")

# No handler needed! Vercel will automatically find and serve the `app` object.

