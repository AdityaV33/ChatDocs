import sys, os, json, time
import numpy as np
import faiss

sys.path.insert(0, os.path.abspath('.'))
from evaluation.retrieval.cosine_retriever import CosineRetriever

def run_benchmark_for_scale(scale_name, npz_path):
    print(f"\n--- Running scale: {scale_name} ---")
    npz = np.load(npz_path, allow_pickle=True)
    doc_embeddings = npz['doc_embeddings']
    doc_ids = npz['doc_ids']
    query_embeddings = npz['query_embeddings']
    query_ids = npz['query_ids']
    
    num_queries = len(query_ids)
    
    # 1. Setup
    faiss_index = faiss.IndexFlatL2(1536)
    faiss_index.add(doc_embeddings)
    
    cosine_retriever = CosineRetriever(1536)
    metadata = [{"id": str(did)} for did in doc_ids]
    cosine_retriever.add_embeddings(doc_embeddings.tolist(), metadata)
    
    # 2. Warm-up
    dummy_query = np.random.rand(1, 1536).astype(np.float32)
    faiss_index.search(dummy_query, 5)
    cosine_retriever.similarity_search(dummy_query[0].tolist(), top_k=5)
    
    # 3. Benchmark
    faiss_latencies = []
    cosine_latencies = []
    
    # FAISS
    for q_emb in query_embeddings:
        q_emb_2d = q_emb.reshape(1, -1)
        start = time.perf_counter()
        faiss_index.search(q_emb_2d, 5)
        faiss_latencies.append((time.perf_counter() - start) * 1000)
        
    # Cosine
    for q_emb in query_embeddings:
        start = time.perf_counter()
        cosine_retriever.similarity_search(q_emb.tolist(), top_k=5)
        cosine_latencies.append((time.perf_counter() - start) * 1000)
        
    # 4. Compute Stats
    def compute_stats(latencies):
        arr = np.array(latencies)
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
            "p95": float(np.percentile(arr, 95)),
            "throughput_qps": 1000.0 / float(np.mean(arr)) if np.mean(arr) > 0 else 0
        }
        
    return {
        "scale": scale_name,
        "num_docs": len(doc_embeddings),
        "faiss": compute_stats(faiss_latencies),
        "cosine": compute_stats(cosine_latencies)
    }

def main():
    scales = [
        ("2.6K (Primary)", 'evaluation/dataset/cached_embeddings.npz'),
        ("5K (Primary+Sec)", 'evaluation/dataset/cached_embeddings_5k.npz'),
        ("10K (Primary+Sec)", 'evaluation/dataset/cached_embeddings_10k.npz')
    ]
    
    results = []
    for name, path in scales:
        res = run_benchmark_for_scale(name, path)
        results.append(res)
        
    os.makedirs('evaluation/results', exist_ok=True)
    with open('evaluation/results/part_b_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
        
    # Print clean table
    print("\n==================================================")
    print("           PART B LATENCY SCALING RESULTS")
    print("==================================================")
    
    print(f"\nFAISS IndexFlatL2")
    print(f"{'Index Size':<15} | {'Mean (ms)':<10} | {'Median (ms)':<11} | {'p95 (ms)':<10} | {'Std Dev':<10} | {'Throughput (QPS)':<15}")
    print("-" * 80)
    for r in results:
        s = r["faiss"]
        print(f"{r['num_docs']:<15} | {s['mean']:<10.4f} | {s['median']:<11.4f} | {s['p95']:<10.4f} | {s['std']:<10.4f} | {s['throughput_qps']:<15.1f}")

    print(f"\nExact NumPy Cosine")
    print(f"{'Index Size':<15} | {'Mean (ms)':<10} | {'Median (ms)':<11} | {'p95 (ms)':<10} | {'Std Dev':<10} | {'Throughput (QPS)':<15}")
    print("-" * 80)
    for r in results:
        s = r["cosine"]
        print(f"{r['num_docs']:<15} | {s['mean']:<10.4f} | {s['median']:<11.4f} | {s['p95']:<10.4f} | {s['std']:<10.4f} | {s['throughput_qps']:<15.1f}")
        
    print(f"\nDetailed stats saved to evaluation/results/part_b_results.json")

if __name__ == "__main__":
    main()
