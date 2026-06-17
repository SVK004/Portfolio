import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
# import google.generativeai as genai
import requests
from fastapi.middleware.cors import CORSMiddleware
# from dotenv import load_dotenv
# load_dotenv()

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
    except Exception as e:
        print(f"CRITICAL ERROR: Could not find the data file at {file_path}. Ensure 'rag_data.txt' is in the root and included in vercel.json.")
        return ""
# --- END OF FILE LOADING ---

RAG_DATA_SOURCE = load_rag_data()
API_URL = "https://router.huggingface.co/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}",
}
def queryyy(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

knowledge_chunks = [line.strip() for line in RAG_DATA_SOURCE.split("---") if line.strip()]

# --- Gemini API Setup ---
# api_key = os.getenv("GEMINI_API_KEY")
# model = None
# if api_key:
#     try:
#         genai.configure(api_key=api_key)
#         model = genai.GenerativeModel("gemini-2.0-flash")
#         print("Successfully configured Gemini API.")
#     except Exception as e:
#         print(f"Error configuring Gemini API: {e}")
# else:
#     print("Warning: GEMINI_API_KEY not found in environment variables.")


# Pydantic model for the request body
class Query(BaseModel):
    userQuery: str

# API endpoint for the chat - Made synchronous to prevent event loop issues
@app.post("/api/chat")
def chat_endpoint(query: Query):
    print(f"HuggingFace key: {os.getenv('HUGGINGFACE_API_KEY')}")
    user_query_lower = query.userQuery.lower()
    user_query = query.userQuery
    try:
        stop_words = {
    "a", "an", "the", "in", "on", "of", "what", "is",
    "who", "where", "when", "tell", "me", "about",
    "for", "this", "that", "with", "by", "has"
}

        meaningful_words = [
            word
            for word in user_query_lower.split()
            if word not in stop_words
        ]

        ranked_chunks = sorted(
            knowledge_chunks,
            key=lambda chunk: sum(
                word in chunk.lower()
                for word in meaningful_words
            ),
            reverse=True
        )

        top_chunks = ranked_chunks[:3]

        context = "\n\n".join(top_chunks)

        print("\n========== DEBUG ==========")
        print("Query:", user_query)
        print("\nMeaningful Words:", meaningful_words)

        for i, chunk in enumerate(top_chunks, start=1):
            print(f"\n--- Chunk {i} ---")
            print(chunk[:500])

        print("===========================\n")
            

        llama_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

    You are Venkat's Portfolio Assistant.

    Your purpose is to answer questions about Venkat's education, experience, projects, skills, achievements, and career interests.

    Use the KNOWLEDGE BASE as the primary source of truth.

    Guidelines:

    1. Prefer information from the KNOWLEDGE BASE whenever available.
    2. You may rephrase and summarize information naturally.
    3. Do not invent companies, degrees, certifications, years of experience, achievements, or projects.
    4. If a question asks for information that is not available in the KNOWLEDGE BASE, politely say that the information is not available.
    5. Keep responses concise and professional.
    6. When discussing projects, mention technologies and key features when relevant.
    7. When discussing skills or experience, use only information present in the KNOWLEDGE BASE.

    KNOWLEDGE BASE:
    {context}

    USER QUESTION:
    {user_query}


    <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        
        payload = {
        "model": "meta-llama/Llama-3.1-8B-Instruct:featherless-ai", # Example model
        "messages": [{
            "role": "user",
            "content": llama_prompt
            }], # Add your RAG data here
        "temperature": 0.2,
        "max_tokens": 100
        }
            

        print("========== CONTEXT ==========")
        print(context)
        print("=============================")

        output = queryyy({
        "model": "meta-llama/Llama-3.1-8B-Instruct:featherless-ai", # Example model
        "messages": [{
            "role": "user",
            "content": llama_prompt
            }], # Add your RAG data here
        "temperature": 0.2,
        "max_tokens": 100
    })
        print(output)
            # Switched to the synchronous SDK call
        response = requests.post(API_URL, headers = headers, json=payload, timeout=20)
        print(f"Status: {response.status_code} | Response: {response.text}")
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Hugging Face API Error")
        
        result = response.json()
        print("botResponse", result)
        print("botResponse", result['choices'][0]["message"]["content"].strip())
        # 1. Check if the API actually gave us 'choices'
        if 'choices' in output and len(output['choices']) > 0:
            # 2. Extract the 'text' field (Completions API uses 'text', not 'message')
            bot_text = output['choices'][0]['message']['content']
            
            # 3. Clean up any leftover Llama 3 instruct tags if they leaked
            tags_to_clean = ["<|start_header_id|>assistant<|end_header_id|>", "<|eot_id|>", "assistant"]
            for tag in tags_to_clean:
                bot_text = bot_text.replace(tag, "").strip()

            # 4. Return the clean string to your frontend
            return {"botResponse": bot_text}
        else:
            # Log the full output for debugging if it's empty
            print(f"Unexpected API structure: {output}")
            raise HTTPException(status_code=502, detail="AI response format was invalid.")
    except Exception as e:
        print(f"RAG Backend Error: {str(e)}")
        raise HTTPException(status_code=500, detail="An internal error occurred while processing the request.")

# No handler needed! Vercel will automatically find and serve the `app` object.

