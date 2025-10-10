import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add this middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods
    allow_headers=["*"], # Allows all headers
)


# Load environment variables from .env file
load_dotenv()

# Configure the Gemini API client
# This will fail gracefully if the key is not found
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel('gemini-2.0-flash')
except Exception as e:
    print(f"Warning: Gemini API key not found or invalid. The model will not work. Error: {e}")
    model = None

# Your portfolio data, which acts as the knowledge base for the RAG model
# print("Hdkijfgb", os.getenv("RAG_DATA_SOURCE"))
RAG_DATA_SOURCE = '''
About Me: A motivated Computer Science undergraduate specializing in AI & ML. My core experience lies in architecting and developing robust backend systems with Node.js, FastAPI, and MongoDB. As a tech freak and cybersecurity enthusiast, I am constantly exploring new technologies and security principles to build secure, innovative solutions. My project portfolio demonstrates a strong capability in integrating cutting-edge technologies like Large Language Models (LLMs) to build practical, real-world tools. I am passionate about system design and eager to contribute my skills to a challenging development role.
Education at Keshav Memorial Engineering College: B.Tech in Computer Science (AI & ML), expected graduation March 2026, CGPA 8.40.
Education at Sri Chaitanya Junior Kalasala: Intermediate, completed March 2022, Percentage 94.3%.
Education at Takshasila Public School: Schooling, completed March 2020, GPA 8.2.
Project SmartTask AI: An LLM-Powered Task Planner. It was a full-stack task planner using Ollama, CrewAI, and FastAPI. It integrates with Google Calendar for auto-scheduling.
Project RedactPDF: An Agentic AI PDF Redaction Tool. It was developed with LangChain and CrewAI. It features a FastAPI backend and a React interface.
Project Emotion Recognizer: A Speech-Based ML App. It classified emotions from speech using MFCC and NLP features. It had a REST API in Flask and a React frontend.
Project Word Matching Game: Developed an accessible word game for dyslexic and autistic learners using React and Pixi.js. Deployed via Vercel.
Project Nyaay Sahaayak: A user-friendly legal advisory platform using a regex-based query strategy to interpret user input.
Skills in Programming Languages: Java, C++, C, Python, JavaScript.
Skills in Web Technologies: React, Node.js, Express.js, HTML, CSS.
Skills in Databases: MySQL, MongoDB.
Skills in Cloud & Tools: AWS basics (S3, EC2), Git, GitHub, Docker, Postman.
Skills in Concepts: Data Structures & Algorithms, Problem Solving, System Design.'''

# Prepare the RAG data by splitting it into meaningful chunks
knowledge_chunks = [chunk.strip() for chunk in RAG_DATA_SOURCE.strip().split('\n') if chunk.strip()]

# Pydantic model to define the structure of the request body
class Query(BaseModel):
    userQuery: str

# Define the API endpoint for the chat
@app.post("/api/chat")
async def chat_endpoint(query: Query):
    user_query = query.userQuery

    if not model:
        raise HTTPException(status_code=500, detail="Gemini API key is not configured on the server.")

    if not user_query:
        raise HTTPException(status_code=400, detail="Query is required.")

    try:
        # --- 1. RETRIEVAL (Your custom TF-IDF model) ---
        # Create a corpus with the user query as the first item, followed by all knowledge chunks
        corpus = [user_query] + knowledge_chunks
        
        # Initialize the TF-IDF Vectorizer and transform the corpus into a matrix of TF-IDF features
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(corpus)
        
        # Calculate the cosine similarity between the user query's vector (the first row) and all chunk vectors
        cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        
        top_n_indices = cosine_similarities.argsort()[-3:][::-1]
        
        # Combine the text from these top 3 chunks to create a richer context
        context = "\n".join([knowledge_chunks[i] for i in top_n_indices])

        # --- 2. AUGMENTATION & 3. GENERATION ---
        # Construct the final prompt for the Gemini model
        prompt = f"""Based ONLY on the following information, answer the user's question concisely. If the information isn't present, state that you don't have that information.\n\nCONTEXT:\n{context}\n\nQUESTION:\n{user_query}"""
        
        # Asynchronously generate content using the Gemini model
        response = await model.generate_content_async(prompt)

        return {"botResponse": response.text}

    except Exception as e:
        print(f"RAG Backend Error: {e}")
        raise HTTPException(status_code=500, detail="Failed to process the request.")
