import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load data ---
def load_rag_data(file_name="rag_data.txt"):
    script_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    file_path = os.path.join(project_root, file_name)
    if os.path.exists(file_name):
        file_path = file_name
    elif not os.path.exists(file_path):
        print(f"ERROR: Missing file at {file_path}")
        return ""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

RAG_DATA_SOURCE = load_rag_data()
knowledge_chunks = [line.strip() for line in RAG_DATA_SOURCE.splitlines() if line.strip()]

# --- Gemini Setup ---
api_key = os.getenv("GEMINI_API_KEY")
model = None
if api_key:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash-latest")

class Query(BaseModel):
    userQuery: str

@app.post("/api/chat")
async def chat_endpoint(query: Query):
    if not model:
        raise HTTPException(status_code=503, detail="Gemini model not initialized.")
    if not knowledge_chunks:
        raise HTTPException(status_code=503, detail="No RAG data loaded.")
    if not query.userQuery:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # Simple keyword-based retrieval (lightweight)
    user_query = query.userQuery.lower()
    ranked_chunks = sorted(
        knowledge_chunks,
        key=lambda c: sum(w in c.lower() for w in user_query.split()),
        reverse=True
    )
    context = "\n".join(ranked_chunks[:3])

    prompt = f"""Based ONLY on the following information, answer concisely.
If information is missing, say you don't have enough info.

CONTEXT:
{context}

QUESTION:
{user_query}"""

    response = await model.generate_content_async(prompt)
    return {"botResponse": response.text}

# ✅ Export for Vercel Lambda
# handler = Mangum(app)
