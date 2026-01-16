from openai import OpenAI
from typing import List, Dict

client = OpenAI()

SUMMARY_MODEL = "gpt-3.5-turbo"


def generate_summary(chunks: List[Dict]) -> str:
    """
    Generate a concise summary from document chunks.
    """

    
    summary_context = "\n\n".join(
        chunk["chunk_text"] for chunk in chunks[:5]
    )

    system_prompt = (
        "You are a helpful assistant that summarizes documents clearly "
        "and concisely for a student."
    )

    user_prompt = f"""
Document content:
{summary_context}

Give a concise summary (5–7 sentences):
"""

    response = client.chat.completions.create(
        model=SUMMARY_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3,
        max_tokens=200
    )

    return response.choices[0].message.content
