import os
from dotenv import load_dotenv

load_dotenv()

GENERATION_PROVIDER = os.environ.get("GENERATION_PROVIDER", "openai").lower()

if GENERATION_PROVIDER not in ["openai", "groq"]:
    raise ValueError(f"Invalid GENERATION_PROVIDER: {GENERATION_PROVIDER}. Must be 'openai' or 'groq'.")

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")

if GENERATION_PROVIDER == "groq":
    GROQ_MODEL = os.environ.get("GROQ_MODEL")
    if not GROQ_MODEL:
        raise ValueError("GROQ_MODEL must be explicitly set in the environment when GENERATION_PROVIDER is 'groq'.")
else:
    GROQ_MODEL = None

# Embeddings are STRICTLY OpenAI.
EMBEDDING_MODEL = "text-embedding-3-small"
VECTOR_DIM = 1536
