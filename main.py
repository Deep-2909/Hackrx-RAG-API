# main.py (Final Version with Long Timeout)
import os
import json
import time
import uuid
from fastapi import FastAPI, Request, HTTPException, status
from pydantic import BaseModel
from typing import List

# --- 1. SETUP & CONSTANTS ---
EXPECTED_AUTH_TOKEN = "c88d7e70b6c77cd88271a48126bcd54761315985a275d864cd7e2b7ba342f1cf"
JOBS_DIR = "jobs"
RESULTS_DIR = "results"

# Ensure directories exist on startup
os.makedirs(JOBS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- 2. FASTAPI APP DEFINITION ---
app = FastAPI(title="Intelligent Query-Retrieval System")

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class AnswerResponse(BaseModel):
    answers: List[str]

@app.post("/hackrx/run", response_model=AnswerResponse)
async def process_queries(request: QueryRequest, http_request: Request):
    # --- Start the timer ---
    start_time = time.time()

    # --- A. Authentication ---
    auth_header = http_request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Authorization header")
    token = auth_header.split(" ")[1]
    if token != EXPECTED_AUTH_TOKEN:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid token")

    # --- B. Job Creation ---
    job_id = str(uuid.uuid4())
    job_path = os.path.join(JOBS_DIR, f"{job_id}.json")
    result_path = os.path.join(RESULTS_DIR, f"{job_id}.json")

    job_data = {
        "document_url": request.documents,
        "questions": request.questions
    }
    with open(job_path, "w") as f:
        json.dump(job_data, f)
    print(f"Job {job_id} created for document: {request.documents}")

    # --- C. Wait for Result (with a very long timeout for testing) ---
    print(f"Waiting for worker to process job {job_id}...")
    # This loop will now wait for up to 1 hour (3600 seconds)
    for _ in range(3600):
        if os.path.exists(result_path):
            with open(result_path, "r") as f:
                result_data = json.load(f)
            os.remove(result_path) # Clean up the result file
            
            # --- Stop the timer and print the duration ---
            end_time = time.time()
            duration = end_time - start_time
            print(f"--- Request for job {job_id} completed in {duration:.2f} seconds ---")
            
            return result_data
        time.sleep(1)

    # If the worker takes longer than an hour, it will timeout.
    raise HTTPException(status_code=status.HTTP_408_REQUEST_TIMEOUT, detail="Processing timed out after 1 hour.")