from fastapi import APIRouter, UploadFile, File, HTTPException

from app.adapters.docx_loader import load_docx_text
from app.processing.cleaner import clean_text
from app.processing.chunker import chunk_text
from app.vectorstore.embeddings import embed_chunks
from app.api.upload import vector_store
from app.summary.summary_generator import generate_summary

router = APIRouter()


@router.post("/upload/docx")
async def upload_docx(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".docx"):
        raise HTTPException(
            status_code=400,
            detail="Only DOCX files are supported."
        )

    
    vector_store.reset()

    file_bytes = await file.read()

    try:
        raw_text = load_docx_text(file_bytes)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Failed to read DOCX file."
        )

    cleaned = clean_text(raw_text)
    if not cleaned:
        raise HTTPException(
            status_code=400,
            detail="No readable text found in DOCX."
        )

    chunks = chunk_text(cleaned)

    chunks_with_metadata = []
    for i, chunk in enumerate(chunks, start=1):
        chunks_with_metadata.append({
            "document_id": file.filename,
            "chunk_id": i,
            "page_number": 0,   
            "source": file.filename,
            "chunk_text": chunk
        })

    embeddings = embed_chunks(chunks_with_metadata)

    vector_store.add_embeddings(
        embeddings=embeddings,
        metadatas=chunks_with_metadata[:len(embeddings)]
    )

    summary = generate_summary(chunks_with_metadata)

    return {
        "document": file.filename,
        "summary": summary,
        "embedded_chunks": len(embeddings),
        "status": "Ready for questions"
    }
