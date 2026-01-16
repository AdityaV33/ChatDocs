from openai import OpenAI
from typing import List, Dict

client = OpenAI()

QUESTION_EMBED_MODEL = "text-embedding-3-small"
ANSWER_MODEL = "gpt-3.5-turbo"
TOP_K = 5


def embed_question(question: str) -> list:
    response = client.embeddings.create(
        model=QUESTION_EMBED_MODEL,
        input=question
    )
    return response.data[0].embedding


def retrieve_relevant_chunks(
    question_embedding: list,
    vector_store
) -> List[Dict]:
    return vector_store.similarity_search(
        query_embedding=question_embedding,
        top_k=TOP_K
    )


def generate_answer(
    question: str,
    retrieved_chunks: List[Dict],
    history: List[Dict] = None
) -> Dict:
    """
    Generate an answer grounded in retrieved chunks + conversation history.
    """

    if history is None:
        history = []

    context = "\n\n".join(
        f"(Page {chunk['page_number']}): {chunk['chunk_text']}"
        for chunk in retrieved_chunks
    )

    system_prompt = """
You are a document question-answering assistant.

RULES (must follow):
1) Answer using ONLY the provided context.
2) If the context contains the answer OR clearly describes it, you MUST answer.
3) Do NOT use outside knowledge.
4) Do NOT guess names, numbers, dates, or facts.
5) Keep the answer short and direct (2–5 sentences).
6) If possible, quote a short phrase from the context to support the answer.
7) If the document contains something related to the question just asked you must answer accordingly.
""".strip()

   
    recent_history = history[-6:]

    history_block = "\n".join(
        f"{m.get('role', '').upper()}: {m.get('content', '')}"
        for m in recent_history
        if m.get("content")
    ).strip()

    user_prompt = f"""
Context:
{context}

Chat History:
{history_block}

Question:
{question}

Answer:
""".strip()

    response = client.chat.completions.create(
        model=ANSWER_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": [
            {
                "page_number": chunk["page_number"],
                "chunk_id": chunk["chunk_id"],
                "source": chunk["source"]
            }
            for chunk in retrieved_chunks
        ],
        "used_context": [chunk["chunk_text"] for chunk in retrieved_chunks]
    }
