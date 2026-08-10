import sys, os, json, time
import numpy as np
from openai import OpenAI
sys.path.insert(0, os.path.abspath('.'))
from dotenv import load_dotenv
load_dotenv()

client = OpenAI()
EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 500

def get_embeddings_in_batches(texts, desc=""):
    all_embeddings = []
    total = len(texts)
    for i in range(0, total, BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        print(f"[{desc}] Requesting batch {i} to {i+len(batch)} of {total}...")
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
        time.sleep(0.5)
    return np.array(all_embeddings, dtype=np.float32)

def main():
    print("Loading primary cache...")
    npz = np.load('evaluation/dataset/cached_embeddings.npz', allow_pickle=True)
    primary_doc_embeddings = npz['doc_embeddings']
    primary_doc_ids = npz['doc_ids']
    query_embeddings = npz['query_embeddings']
    query_ids = npz['query_ids']
    
    print("Loading secondary chunks...")
    with open('evaluation/dataset/secondary_chunks.json', 'r', encoding='utf-8') as f:
        sec_chunks = json.load(f)
        
    sec_ids = []
    sec_texts = []
    for c in sec_chunks:
        sec_ids.append(f"{c['document_id']}_{c['chunk_id']}")
        sec_texts.append(c['chunk_text'])
        
    print(f"Loaded {len(sec_texts)} secondary chunks.")
    
    print("\nFetching secondary embeddings from OpenAI...")
    sec_embeddings = get_embeddings_in_batches(sec_texts, desc="Secondary")
    
    # Verify shape
    assert sec_embeddings.shape == (7332, 1536)
    sec_ids_arr = np.array(sec_ids, dtype=object)
    
    # Construct 5K
    # 2668 + 2332 = 5000
    doc_embeddings_5k = np.vstack([primary_doc_embeddings, sec_embeddings[:2332]])
    doc_ids_5k = np.concatenate([primary_doc_ids, sec_ids_arr[:2332]])
    
    assert doc_embeddings_5k.shape[0] == 5000
    
    out_5k = 'evaluation/dataset/cached_embeddings_5k.npz'
    np.savez_compressed(
        out_5k,
        doc_embeddings=doc_embeddings_5k,
        doc_ids=doc_ids_5k,
        query_embeddings=query_embeddings,
        query_ids=query_ids
    )
    print(f"\nSaved 5K cache to {out_5k}")
    
    # Construct 10K
    # 2668 + 7332 = 10000
    doc_embeddings_10k = np.vstack([primary_doc_embeddings, sec_embeddings])
    doc_ids_10k = np.concatenate([primary_doc_ids, sec_ids_arr])
    
    assert doc_embeddings_10k.shape[0] == 10000
    
    out_10k = 'evaluation/dataset/cached_embeddings_10k.npz'
    np.savez_compressed(
        out_10k,
        doc_embeddings=doc_embeddings_10k,
        doc_ids=doc_ids_10k,
        query_embeddings=query_embeddings,
        query_ids=query_ids
    )
    print(f"Saved 10K cache to {out_10k}")

if __name__ == "__main__":
    main()
