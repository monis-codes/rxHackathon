import os
import hashlib
import requests
import brainRag as br

from fastapi import FastAPI, Depends, HTTPException, status, Header
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from typing import List

from dotenv import load_dotenv
import google.generativeai as genai
from pinecone import Pinecone
from pypdf import PdfReader
from io import BytesIO

# --- Environment and API Key Setup ---

load_dotenv()

# Load API keys from .env file
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
MY_API_KEY = os.getenv("MY_API_KEY") # Your secret key for API authentication
PINECONE_INDEX_NAME = "hack-rx" # Your pinecone index name
PINECONE_HOST = os.getenv("PINECONE_HOST")
# --- FastAPI App Initialization ---

app = FastAPI(
    title="HackRx API",
    description="API for ingesting documents and answering questions using a RAG pipeline.",
    version="1.0.0"
)

# --- API Key Authentication ---

api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

async def get_api_key(auth_header: str = Depends(api_key_header)):
    """Dependency to validate the API key from the Authorization header."""
    if not auth_header:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing"
        )
    
    # Expecting "Bearer YOUR_API_KEY"
    try:
        scheme, key = auth_header.split()
        if scheme.lower() != "bearer" or key != MY_API_KEY:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials"
            )
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication format. Use 'Bearer YOUR_KEY'."
        )
    return key


# --- Client Configuration ---

# Configure Google AI Client
genai.configure(api_key=GOOGLE_API_KEY)

# Configure Pinecone Client
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host=PINECONE_HOST)

# --- Pydantic Models (API Data Shapes) ---

class HackathonRequest(BaseModel):
    documents: str = Field(..., description="A single URL to a PDF document.")
    questions: List[str] = Field(..., description="A list of questions to be answered based on the document.")

class Answer(BaseModel):
    question: str
    answer: str

class HackathonResponse(BaseModel):
    answers: List[str]


# In main.py

# ... (keep all the code from before this function) ...

@app.post("/hackrx/run", response_model=HackathonResponse, status_code=status.HTTP_200_OK)
async def run_hackathon_logic(
    request: HackathonRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Main endpoint to handle the two-stage hackathon logic.
    Stage 1 (Ingestion): If the document hasn't been processed, ingest it.
    Stage 2 (Querying): If the document is already processed, answer questions.
    """
    try:
        # Use a hash of the document URL as a unique namespace identifier
        namespace = hashlib.sha256(request.documents.encode()).hexdigest()

        # Check if the namespace exists, passing the 'index' object correctly
        doc_is_processed = br.namespace_exists(index, namespace)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to check document status in Pinecone: {e}"
        )

    if not doc_is_processed:
        # --- Stage 1: Ingestion ---
        try:
            # Download the PDF content from the URL
            response = requests.get(request.documents)
            response.raise_for_status()  # Raises an HTTPError for bad responses
            pdf_content = response.content

            # Process and ingest the document
            br.process_document(index, pdf_content, namespace)

            # Return an empty list of answers as per hackathon rules for the first run
            return HackathonResponse(answers=[])

        except requests.RequestException as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to download document: {e}")
        except Exception as e:
            # Catch errors from PDF processing, embedding, or upserting
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to process document: {e}")

    else:
        # --- Stage 2: Querying ---
        try:
            final_answers = []
            for q in request.questions:
                generated_answer = br.get_answer_for_question(index, q, namespace)
                final_answers.append(generated_answer)

            return HackathonResponse(answers=final_answers)

        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to generate answers: {e}")