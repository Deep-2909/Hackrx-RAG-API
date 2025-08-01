# main.py (Final Submission Version)
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import requests
import tempfile
import time
from fastapi import FastAPI, Request, HTTPException, status
from pydantic import BaseModel
import pinecone
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from typing import List
import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
from contextlib import asynccontextmanager
import asyncio
import openai

# --- 1. SETUP & CLIENT CONFIGURATION ---
load_dotenv()

# Define constants
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
INDEX_NAME = "hackrx-index"
GENERATION_MODEL = "mistralai/mistral-7b-instruct-v0.2"
EMBEDDING_DIMENSION = 384
EXPECTED_AUTH_TOKEN = "c88d7e70b6c77cd88271a48126bcd54761315985a275d864cd7e2b7ba342f1cf"
DB_FILE = "processed_docs.txt"

# A dictionary to hold our models and resources, loaded during lifespan
ml_models = {}

# --- 2. LIFESPAN MANAGER FOR STABLE MODEL LOADING ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # This code runs ONCE when the server starts up
    print("Server starting up...")
    
    # Configure clients with API keys
    pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    ml_models["openrouter_client"] = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    
    print("Loading embedding model...")
    ml_models["embedding_model"] = SentenceTransformer(EMBEDDING_MODEL)
    
    print("Connecting to Pinecone index...")
    if INDEX_NAME not in pc.list_indexes().names():
        print(f"Index '{INDEX_NAME}' not found. Creating a new one...")
        pc.create_index(
            name=INDEX_NAME, dimension=EMBEDDING_DIMENSION, metric='cosine',
            spec=pinecone.ServerlessSpec(cloud='aws', region='us-east-1')
        )
    ml_models["pinecone_index"] = pc.Index(INDEX_NAME)
    print("Pinecone index is ready.")
    
    print("Loading processed documents database...")
    ml_models["processed_docs"] = load_processed_documents()
    print("Startup complete.")
    
    yield
    
    # This code runs ONCE when the server shuts down
    print("Server shutting down...")
    ml_models.clear()

# --- 3. FASTAPI APP DEFINITION ---
app = FastAPI(title="Intelligent Query-Retrieval System", lifespan=lifespan)

# --- Database Simulation ---
def load_processed_documents():
    if not os.path.exists(DB_FILE): return set()
    with open(DB_FILE, "r") as f: return set(line.strip() for line in f)

def is_document_indexed(document_id: str) -> bool:
    return document_id in ml_models["processed_docs"]

def mark_document_as_indexed(document_id: str):
    with open(DB_FILE, "a") as f: f.write(document_id + "\n")
    ml_models["processed_docs"].add(document_id)

# --- 4. RAG HELPER FUNCTIONS ---
def get_embedding(text):
    text = text.replace("\n", " ")
    return ml_models["embedding_model"].encode(text).tolist()

def process_and_index_pdf(file_path: str, document_id: str) -> str:
    doc = fitz.open(file_path)
    full_text = "".join(page.get_text() for page in doc)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(full_text)
    
    last_chunk_id = ""
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        embeddings = [get_embedding(text) for text in batch_chunks]
        vectors_to_upsert = []
        for j, chunk_text in enumerate(batch_chunks):
            chunk_id = f"{document_id}-chunk-{i+j}"
            vectors_to_upsert.append({"id": chunk_id, "values": embeddings[j], "metadata": {"text": chunk_text, "document_id": document_id}})
            last_chunk_id = chunk_id
        ml_models["pinecone_index"].upsert(vectors=vectors_to_upsert)
    return last_chunk_id

def find_most_similar_chunks(query_embedding, document_id: str, top_k=5):
    results = ml_models["pinecone_index"].query(vector=query_embedding, top_k=top_k, include_metadata=True, filter={"document_id": {"$eq": document_id}})
    return [match['metadata']['text'] for match in results['matches']]

def get_llm_answer(query, context_chunks):
    context = "\n\n---\n\n".join(context_chunks)
    prompt = f"""
    You are a precise assistant for answering questions about an insurance policy. Your answer MUST be based SOLELY on the context provided below. CRITICAL INSTRUCTION: When answering, you must first look for any specific exclusions, waiting periods, or limitations related to the user's question. If an exclusion clause is present, it OVERRIDES any general definition. FORMATTING INSTRUCTION: Your response must be concise and to the point. Answer in a single, direct sentence.
    CONTEXT: {context} QUESTION: {query} ANSWER:
    """
    response = ml_models["openrouter_client"].chat.completions.create(model=GENERATION_MODEL, messages=[{"role": "user", "content": prompt}])
    return response.choices[0].message.content.strip().replace("\n", " ")

# --- 5. Main Blocking Function to run in a separate thread ---
def run_rag_pipeline(document_url, questions):
    document_id = os.path.basename(document_url.split('?')[0])
    
    if not is_document_indexed(document_id):
        try:
            print(f"Starting indexing for new document: {document_id}")
            pdf_response = requests.get(document_url)
            pdf_response.raise_for_status()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                temp_pdf.write(pdf_response.content)
                temp_pdf_path = temp_pdf.name
            
            last_id = process_and_index_pdf(temp_pdf_path, document_id)
            os.unlink(temp_pdf_path)

            print(f"Waiting for last chunk ({last_id}) to be indexed...")
            for _ in range(45):
                try:
                    fetch_response = ml_models["pinecone_index"].fetch([last_id])
                    if fetch_response.vectors.get(last_id):
                        print("Index updated successfully.")
                        break
                except Exception: pass
                time.sleep(1)
            else:
                print("Warning: Index polling timed out.")
            mark_document_as_indexed(document_id)
        except Exception as e:
            raise RuntimeError(f"Failed to process new document: {e}")

    all_answers = []
    for question in questions:
        query_embedding = get_embedding(question)
        context_chunks = find_most_similar_chunks(query_embedding, document_id)
        if not context_chunks:
            answer = "Could not find relevant information in the specified document to answer this question."
        else:
            answer = get_llm_answer(question, context_chunks)
        all_answers.append(answer)
    return {"answers": all_answers}

# --- 6. API ENDPOINT ---
class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

@app.post("/hackrx/run")
async def process_queries(request: QueryRequest, http_request: Request):
    auth_header = http_request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Authorization header")
    token = auth_header.split(" ")[1]
    if token != EXPECTED_AUTH_TOKEN:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid token")

    try:
        # Run the entire blocking pipeline in a separate thread to prevent crashes
        result = await asyncio.to_thread(run_rag_pipeline, request.documents, request.questions)
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unexpected error occurred: {e}")