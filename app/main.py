from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pathlib import Path

from app.api import upload
from app.api import qa
from app.api import upload_url
from app.api import upload_docx

app = FastAPI(
    title="ChatDocs",
    description="Chat with PDFs, DOCX files, and URLs using a Retrieval-Augmented Generation (RAG) pipeline",
    version="0.1.0"
)


app.include_router(upload.router)
app.include_router(upload_url.router)
app.include_router(upload_docx.router)
app.include_router(qa.router)


BASE_DIR = Path(__file__).resolve().parent
UI_PATH = BASE_DIR / "ui" / "index.html"


@app.get("/", response_class=HTMLResponse)
def serve_ui():
    return UI_PATH.read_text(encoding="utf-8")


@app.get("/health")
def health_check():
    return {"status": "ok"}
