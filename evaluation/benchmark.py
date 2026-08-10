import sys, os, json, time
import numpy as np
import faiss

sys.path.insert(0, os.path.abspath('.'))
from evaluation.retrieval.cosine_retriever import CosineRetriever

def compute_metrics(retrieved_ids, ground_truth):
    ground_truth_set = set(ground_truth)
    hits = [int(rid in ground_truth_set) for rid in retrieved_ids]
    
    # Hit@5
    hit_at_5 = 1 if sum(hits) > 0 else 0
    
    # Precision@5
    p_at_5 = sum(hits) / len(retrieved_ids) if retrieved_ids else 0.0
    
    # Recall@5
    r_at_5 = sum(hits) / len(ground_truth_set) if ground_truth_set else 0.0
    
    # F1@5
    f1_at_5 = 2 * p_at_5 * r_at_5 / (p_at_5 + r_at_5) if (p_at_5 + r_at_5) > 0 else 0.0
    
    # MRR
    mrr = 0.0
    for i, h in enumerate(hits):
        if h == 1:
            mrr = 1.0 / (i + 1)
            break
            
    return {
        "hit_at_5": hit_at_5,
        "precision_at_5": p_at_5,
        "recall_at_5": r_at_5,
        "f1_at_5": f1_at_5,
        "mrr": mrr
    }

def main():
    print("Loading cached embeddings and dataset...")
    
    # 1. Load Data
    npz = np.load('evaluation/dataset/cached_embeddings.npz', allow_pickle=True)
    doc_embeddings = npz['doc_embeddings']
    doc_ids = npz['doc_ids']
    query_embeddings = npz['query_embeddings']
    query_ids = npz['query_ids']
    
    with open('evaluation/dataset/frozen_queries.json', 'r', encoding='utf-8') as f:
        queries = json.load(f)
        
    # Map ground truth
    ground_truths = {}
    for q in queries:
        doc_id = q['document_id']
        # In finalize_queries.py it is stored as "relevant_chunks": [int, int, ...] or similar depending on the source.
        # Wait, the finalize script output 'relevant_chunks' which is a list of integers.
        # But wait! I should handle both 'relevant_chunks' and 'candidate_relevant_chunks' just in case.
        chunks_key = 'relevant_chunks' if 'relevant_chunks' in q else 'candidate_relevant_chunks'
        
        rel_chunks = []
        for cid in q[chunks_key]:
            rel_chunks.append(f"{doc_id}_{cid}")
        ground_truths[q['query_id']] = rel_chunks
        
    print(f"Loaded {len(doc_embeddings)} doc vectors, {len(query_embeddings)} query vectors.")
    
    # 2. Setup Retrievers
    print("Setting up retrievers...")
    
    # FAISS
    faiss_index = faiss.IndexFlatL2(1536)
    faiss_index.add(doc_embeddings)
    
    # Cosine
    cosine_retriever = CosineRetriever(1536)
    metadata = [{"id": str(did)} for did in doc_ids]
    cosine_retriever.add_embeddings(doc_embeddings.tolist(), metadata)
    
    # 3. Warm-up
    print("Warming up...")
    dummy_query = np.random.rand(1, 1536).astype(np.float32)
    faiss_index.search(dummy_query, 5)
    cosine_retriever.similarity_search(dummy_query[0].tolist(), top_k=5)
    
    # 4. Benchmark Loop
    print("Running benchmark...")
    
    per_query_results = []
    
    faiss_total_time = 0.0
    cosine_total_time = 0.0
    
    for i, q_id in enumerate(query_ids):
        q_emb = query_embeddings[i]
        q_emb_2d = q_emb.reshape(1, -1)
        gt = ground_truths[q_id]
        
        # --- FAISS ---
        start = time.perf_counter()
        D, I = faiss_index.search(q_emb_2d, 5)
        faiss_time = time.perf_counter() - start
        
        faiss_result_ids = [str(doc_ids[idx]) for idx in I[0]]
        faiss_total_time += faiss_time
        
        faiss_metrics = compute_metrics(faiss_result_ids, gt)
        faiss_metrics["latency_ms"] = faiss_time * 1000
        
        # --- Cosine ---
        start = time.perf_counter()
        c_res = cosine_retriever.similarity_search(q_emb.tolist(), top_k=5)
        cosine_time = time.perf_counter() - start
        
        cosine_result_ids = [r["id"] for r in c_res]
        cosine_total_time += cosine_time
        
        cosine_metrics = compute_metrics(cosine_result_ids, gt)
        cosine_metrics["latency_ms"] = cosine_time * 1000
        
        # --- Comparison ---
        ranking_agreement = (faiss_result_ids == cosine_result_ids)
        
        per_query_results.append({
            "query_id": q_id,
            "ground_truth": gt,
            "faiss": {
                "top_5": faiss_result_ids,
                "metrics": faiss_metrics
            },
            "cosine": {
                "top_5": cosine_result_ids,
                "metrics": cosine_metrics
            },
            "ranking_agreement": ranking_agreement
        })
        
    # 5. Aggregate Results
    def agg_metric(key, system):
        return np.mean([r[system]["metrics"][key] for r in per_query_results])

    faiss_agg = {
        "precision_at_5": agg_metric("precision_at_5", "faiss"),
        "recall_at_5": agg_metric("recall_at_5", "faiss"),
        "f1_at_5": agg_metric("f1_at_5", "faiss"),
        "mrr": agg_metric("mrr", "faiss"),
        "hit_at_5": agg_metric("hit_at_5", "faiss"),
        "mean_latency_ms": (faiss_total_time / len(query_ids)) * 1000
    }
    
    cosine_agg = {
        "precision_at_5": agg_metric("precision_at_5", "cosine"),
        "recall_at_5": agg_metric("recall_at_5", "cosine"),
        "f1_at_5": agg_metric("f1_at_5", "cosine"),
        "mrr": agg_metric("mrr", "cosine"),
        "hit_at_5": agg_metric("hit_at_5", "cosine"),
        "mean_latency_ms": (cosine_total_time / len(query_ids)) * 1000
    }
    
    total_agreement = sum([1 for r in per_query_results if r["ranking_agreement"]])
    agreement_percentage = (total_agreement / len(query_ids)) * 100
    
    final_report = {
        "metadata": {
            "num_queries": len(query_ids),
            "num_documents": len(doc_embeddings),
            "embedding_dim": 1536
        },
        "aggregate_results": {
            "faiss": faiss_agg,
            "cosine": cosine_agg,
            "ranking_agreement_percentage": agreement_percentage
        },
        "per_query_results": per_query_results
    }
    
    os.makedirs('evaluation/results', exist_ok=True)
    with open('evaluation/results/part_a_results.json', 'w', encoding='utf-8') as f:
        json.dump(final_report, f, indent=2)
        
    # 6. Print Report
    print("\n==================================================")
    print("               PART A BENCHMARK RESULTS")
    print("==================================================")
    print(f"Ranking Agreement (Exact Top-5 Match): {agreement_percentage:.1f}%\n")
    
    print(f"{'Metric':<20} | {'FAISS IndexFlatL2':<20} | {'Exact NumPy Cosine':<20}")
    print("-" * 65)
    metrics_to_print = [
        ("Precision@5", "precision_at_5"),
        ("Recall@5", "recall_at_5"),
        ("F1@5", "f1_at_5"),
        ("MRR", "mrr"),
        ("Hit Rate@5", "hit_at_5"),
    ]
    for label, key in metrics_to_print:
        print(f"{label:<20} | {faiss_agg[key]:<20.4f} | {cosine_agg[key]:<20.4f}")
        
    print("-" * 65)
    print(f"{'Mean Latency (ms)':<20} | {faiss_agg['mean_latency_ms']:<20.4f} | {cosine_agg['mean_latency_ms']:<20.4f}")
    
    print(f"\nDetailed per-query results saved to evaluation/results/part_a_results.json")

if __name__ == "__main__":
    main()
