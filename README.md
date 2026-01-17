# ChatDocs

ChatDocs is a lightweight RAG (Retrieval-Augmented Generation) application that allows users to upload a PDF, DOCX, or provide a URL, automatically generates a short summary of the content, and then enables users to ask questions grounded in the uploaded document.  
The system uses OpenAI embeddings + FAISS vector search to retrieve the most relevant parts of the document and generate accurate answers using an LLM.

---

## Features

### 1. Upload Support

ChatDocs supports three types of inputs:

- PDF Upload  
- DOCX Upload  
- URL Input (Webpage)

Each upload is treated as the active document, meaning the system resets the vector database every time a new document/link is uploaded.

---

### 2. Automatic Summarization

After uploading a document or URL, the app automatically generates a concise summary using OpenAI's chat model.  
This gives the user an immediate understanding of what the document contains before asking questions.

---

### 3. Question Answering (RAG-based)

Users can ask questions about the uploaded content.  
The system:

1. Embeds the question into a vector  
2. Retrieves the most relevant document chunks using FAISS similarity search  
3. Sends the retrieved chunks as context into the LLM  
4. Returns an answer grounded in the document

---

### 4. Conversational Chat Experience (History Support)

The UI behaves like a chat interface, and supports follow-up questions using basic chat memory.  
Examples:

- "Who is he?"  
- "What does that mean?"  
- "Explain more"  

The frontend sends recent chat history to the backend along with the question, and the backend includes it inside the prompt for better continuity.

---

## Tech Stack

### Backend

- FastAPI (API server)  
- FAISS (vector similarity search)  
- OpenAI API  
  - Embeddings: text-embedding-3-small  
  - Answer model: gpt-3.5-turbo  
  - Summary model: gpt-3.5-turbo  

### Frontend

- HTML + TailwindCSS  
- Vanilla JavaScript  
- Chat-style interface and unified uploader  

### Additional Tools

- pypdf (extract text from PDFs)  
- python-docx (extract text from DOCX files)  
- LangChain WebBaseLoader (load webpage text)  

---

## Project Flow (Pipeline Explanation)

This app follows a standard RAG pipeline:

---

### Step 1: Upload / Load Text

Depending on the input type:

#### PDF Upload

Handled by:

- app/api/upload.py  

Text is extracted using:

- PdfReader from pypdf  

The PDF is processed page-by-page so each chunk stores its correct page number.

---

#### DOCX Upload

Handled by:

- app/api/upload_docx.py  

Text is extracted using:

- python-docx via Document(io.BytesIO(file_bytes))  

DOCX does not have true page numbers, so page is stored as 0.

---

#### URL Upload

Handled by:

- app/api/upload_url.py  

Text is extracted using:

- WebBaseLoader from langchain_community.document_loaders  

LangChain is used here because it handles webpage loading and returns structured Document objects, making URL ingestion simpler and cleaner than manually parsing HTML.

---

### Step 2: Clean the Text

File:

- app/processing/cleaner.py  

Cleaning includes:

- removing unnecessary newlines  
- removing repeated/multiple spaces  

This improves chunk quality and embedding consistency.

---

### Step 3: Chunk the Text

File:

- app/processing/chunker.py  

Chunking splits text into smaller overlapping pieces so embedding retrieval works properly.

Default chunking logic:

- chunk_size = 900  
- overlap = 100  

Why chunking is required:

- Embedding models work best when input text is not too large  
- Smaller chunks increase accuracy when retrieving relevant information  
- Overlap ensures important sentences are not cut off between chunks  

---

### Step 4: Add Metadata to Each Chunk

Each chunk stores metadata like:

- document_id  
- chunk_id  
- page_number  
- source  
- chunk_text  

This helps:

- show sources to users  
- trace retrieved content back to a location in the document  

---

### Step 5: Create Embeddings

File:

- app/vectorstore/embeddings.py  

Chunks are embedded using:

- OpenAI embeddings: text-embedding-3-small  

Cost control:

- Only embeds up to MAX_CHUNKS = 25  

This prevents expensive embedding calls on very large documents.

---

### Step 6: Store Embeddings in FAISS Vector Database

File:

- app/vectorstore/faiss_store.py  

FAISS is used for fast similarity search using L2 distance:

faiss.IndexFlatL2(embedding_dim)

The vector store keeps:

- FAISS index (vectors)  
- metadata list (chunk details)  

---

### Step 7: Ask Questions (Retrieval + Answer Generation)

API endpoint:

- POST /ask  

Handled by:

- app/api/qa.py  

Pipeline file:

- app/qa/qa_pipeline.py  

Question answering works like this:

1. User sends question  
2. Question is converted into an embedding  
3. FAISS retrieves the most similar chunks (TOP_K = 5)  
4. Retrieved chunks are sent into the LLM as context  
5. LLM generates answer only from that context  

---

## Prompting / Answer Grounding Rules

The assistant follows these strict rules:

- Answer using only the provided context  
- If the context clearly contains the answer, it must answer  
- No outside knowledge  
- No guessing  
- Keep responses short and direct (2–5 sentences)  
- Quote document text when possible  

The context is constructed like:

(Page X): chunk_text

---

## Single Document Mode

The system is designed to work in single active document mode.  
Each upload resets the FAISS index:

vector_store.reset()

This ensures:

- old documents do not interfere with new uploads  
- results always match the currently uploaded content  

---

## API Endpoints

### 1. Upload PDF

POST /upload

Uploads a PDF file and returns:

- summary  
- embedded chunk count  

---

### 2. Upload DOCX

POST /upload/docx

Uploads a DOCX file and returns:

- summary  
- embedded chunk count  

---

### 3. Upload URL

POST /upload/url

Uploads a URL and returns:

- summary  
- embedded chunk count  

---

### 4. Ask Question

POST /ask

Request body includes:

```json
{
  "question": "your question",
  "history": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```
Response includes:

answer

sources

used_context

Running Locally

Create virtual environment  
python -m venv venv  

Activate environment  
Windows PowerShell:  
venv\Scripts\Activate.ps1  

Install dependencies  
pip install -r requirements.txt  

Add environment variables  
Create a .env file in the root:  
OPENAI_API_KEY=your_api_key_here  

Run the app  
uvicorn app.main:app --reload  

App runs at:  
http://127.0.0.1:8000  

Important Notes  

API Key Security  

.env is ignored using .gitignore, so it will not be uploaded to GitHub.  
The .gitignore file includes:  

.env  

.env.*  

venv/  

pycache/  

*.pyc  

Limitations (Current Version)  

DOCX does not support true page numbers (stored as page 0)  

Vector store is in-memory (data resets when server restarts)  

Only embeds first 25 chunks for cost control  

Future Improvements  

Potential upgrades for production-readiness:  

Persistent vector store (save FAISS index to disk)  

Multi-document support with document switching  

Better chunking strategy (semantic chunking)  

Add citations with exact chunk text preview  

Improve summarization using the whole document progressively  

Add authentication and user sessions  

Deploy on Render / Railway / AWS make it look good on github readme.md donot change the text

