import sys, os, json, time
import numpy as np
from openai import OpenAI
sys.path.insert(0, os.path.abspath('.'))
from dotenv import load_dotenv
load_dotenv()

# We use the direct OpenAI client to bypass the MAX_CHUNKS=25 limit in the app's vectorstore.
client = OpenAI()
EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 500

def get_embeddings_in_batches(texts, desc=""):
    all_embeddings = []
    total = len(texts)
    for i in range(0, total, BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        print(f"[{desc}] Requesting batch {i} to {i+len(batch)} of {total}...")
        
        # Make the real API call
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch
        )
        
        # We DO NOT normalize. We want exactly what OpenAI returns natively.
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
        
        # Respect rate limits briefly
        time.sleep(0.5)
        
    return np.array(all_embeddings, dtype=np.float32)

def main():
    print("Loading datasets...")
    with open('evaluation/dataset/document_chunks.json', 'r', encoding='utf-8') as f:
        chunks = json.load(f)
        
    with open('evaluation/dataset/frozen_queries.json', 'r', encoding='utf-8') as f:
        queries = json.load(f)
        
    # 1. Prepare Document Chunks
    # We want exactly 2668 chunks
    doc_ids = []
    doc_texts = []
    for c in chunks:
        doc_ids.append(f"{c['document_id']}_{c['chunk_id']}")
        doc_texts.append(c['chunk_text'])
        
    # 2. Prepare Queries
    # We want exactly 100 queries
    query_ids = []
    query_texts = []
    for q in queries:
        query_ids.append(q['query_id'])
        query_texts.append(q['question'])
        
    print(f"Loaded {len(doc_texts)} document chunks and {len(query_texts)} queries.")
    
    # 3. Fetch Embeddings
    print("\nFetching document embeddings from OpenAI...")
    doc_embeddings = get_embeddings_in_batches(doc_texts, desc="Documents")
    
    print("\nFetching query embeddings from OpenAI...")
    query_embeddings = get_embeddings_in_batches(query_texts, desc="Queries")
    
    # 4. Verification and Math
    print("\n--- VERIFICATION SUMMARY ---")
    print(f"Documents: {len(doc_texts)}")
    print(f"Queries: {len(query_texts)}")
    print(f"Embedding dimension: {doc_embeddings.shape[1]}")
    
    # Compute norms
    doc_norms = np.linalg.norm(doc_embeddings, axis=1)
    query_norms = np.linalg.norm(query_embeddings, axis=1)
    
    print("\nDocument norm:")
    print(f"  min / max / mean / std")
    print(f"  {doc_norms.min():.5f} / {doc_norms.max():.5f} / {doc_norms.mean():.5f} / {doc_norms.std():.5f}")
    
    print("\nQuery norm:")
    print(f"  min / max / mean / std")
    print(f"  {query_norms.min():.5f} / {query_norms.max():.5f} / {query_norms.mean():.5f} / {query_norms.std():.5f}")
    
    # Check NaN / Inf
    nan_inf_count = 0
    if not np.isfinite(doc_embeddings).all():
        nan_inf_count += np.sum(~np.isfinite(doc_embeddings))
    if not np.isfinite(query_embeddings).all():
        nan_inf_count += np.sum(~np.isfinite(query_embeddings))
        
    print(f"\nNaN/Inf vectors: {nan_inf_count}")
    
    # Assertions for safety
    assert doc_embeddings.shape == (2668, 1536), f"Doc array shape is wrong: {doc_embeddings.shape}"
    assert query_embeddings.shape == (100, 1536), f"Query array shape is wrong: {query_embeddings.shape}"
    assert len(doc_ids) == 2668, f"Doc IDs count is wrong: {len(doc_ids)}"
    assert len(query_ids) == 100, f"Query IDs count is wrong: {len(query_ids)}"
    assert nan_inf_count == 0, "Found NaN or Inf values!"
    
    # 5. Save the exact NPZ
    out_path = 'evaluation/dataset/cached_embeddings.npz'
    np.savez_compressed(
        out_path,
        doc_embeddings=doc_embeddings,
        doc_ids=np.array(doc_ids, dtype=object),
        query_embeddings=query_embeddings,
        query_ids=np.array(query_ids, dtype=object)
    )
    print(f"\nSuccessfully cached to {out_path}")

if __name__ == "__main__":
    main()
