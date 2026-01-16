from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Literal, Optional

from app.qa.qa_pipeline import (
    embed_question,
    retrieve_relevant_chunks,
    generate_answer
)
from app.api.upload import vector_store

router = APIRouter()


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class QuestionRequest(BaseModel):
    question: str
    history: Optional[List[ChatMessage]] = []


@router.post("/ask")
def ask_question(request: QuestionRequest):
    question_embedding = embed_question(request.question)

    retrieved_chunks = retrieve_relevant_chunks(
        question_embedding,
        vector_store
    )

    result = generate_answer(
        question=request.question,
        retrieved_chunks=retrieved_chunks,
        history=[m.model_dump() for m in (request.history or [])]
    )

    return result
