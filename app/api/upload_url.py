from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.adapters.langchain_loaders import load_url_text
from app.processing.cleaner import clean_text
from app.processing.chunker import chunk_text
from app.vectorstore.embeddings import embed_chunks
from app.api.upload import vector_store
from app.summary.summary_generator import generate_summary

router = APIRouter()


class URLRequest(BaseModel):
    url: str


@router.post("/upload/url")
def upload_url(request: URLRequest):
    
    vector_store.reset()

    try:
        raw_text = load_url_text(request.url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    cleaned = clean_text(raw_text)
    if not cleaned:
        raise HTTPException(status_code=400, detail="No usable text found")

    chunks = chunk_text(cleaned)

    chunks_with_metadata = []
    for i, chunk in enumerate(chunks, start=1):
        chunks_with_metadata.append({
            "document_id": request.url,
            "chunk_id": i,
            "page_number": 0,
            "source": request.url,
            "chunk_text": chunk
        })

    embeddings = embed_chunks(chunks_with_metadata)

    vector_store.add_embeddings(
        embeddings=embeddings,
        metadatas=chunks_with_metadata[:len(embeddings)]
    )

    summary = generate_summary(chunks_with_metadata)

    return {
        "document": request.url,
        "summary": summary,
        "embedded_chunks": len(embeddings),
        "status": "Ready for questions"
    }
