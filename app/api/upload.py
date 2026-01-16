from fastapi import APIRouter, UploadFile, File, HTTPException
from pypdf import PdfReader
import io

from app.processing.cleaner import clean_text
from app.processing.chunker import chunk_text
from app.vectorstore.embeddings import embed_chunks
from app.vectorstore.faiss_store import FaissVectorStore
from app.summary.summary_generator import generate_summary

router = APIRouter()


VECTOR_DIM = 1536


vector_store = FaissVectorStore(embedding_dim=VECTOR_DIM)


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    
    vector_store.reset()

    document_id = file.filename

    pdf_bytes = await file.read()
    reader = PdfReader(io.BytesIO(pdf_bytes))

    chunks_with_metadata = []
    chunk_counter = 0

    for page_index, page in enumerate(reader.pages):
        raw_text = page.extract_text()
        cleaned_text = clean_text(raw_text)

        if not cleaned_text:
            continue

        chunks = chunk_text(cleaned_text)

        for chunk in chunks:
            chunk_counter += 1
            chunks_with_metadata.append({
                "document_id": document_id,
                "chunk_id": chunk_counter,
                "page_number": page_index + 1,
                "source": file.filename,
                "chunk_text": chunk
            })

    if not chunks_with_metadata:
        raise HTTPException(
            status_code=400,
            detail="No readable text found in the PDF."
        )

    embeddings = embed_chunks(chunks_with_metadata)

    vector_store.add_embeddings(
        embeddings=embeddings,
        metadatas=chunks_with_metadata[:len(embeddings)]
    )

    summary = generate_summary(chunks_with_metadata)

    return {
        "document_id": document_id,
        "summary": summary,
        "embedded_chunks": len(embeddings),
        "status": "Ready for questions"
    }
