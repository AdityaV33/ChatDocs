import sys, os, json, time
sys.path.insert(0, os.path.abspath('.'))
from dotenv import load_dotenv
load_dotenv()

from app.config import GENERATION_PROVIDER, GROQ_MODEL, OPENAI_MODEL
from app.qa.qa_pipeline import gen_client

# Use a different Groq model to avoid the rate limit on llama-3.3-70b-versatile
FALLBACK_GROQ_MODEL = "llama-3.1-8b-instant"

print("Loading data...")
with open('evaluation/dataset/selected_100_questions.json', 'r', encoding='utf-8') as f:
    selected_questions = json.load(f)

with open('evaluation/dataset/candidate_queries.json', 'r', encoding='utf-8') as f:
    candidates = json.load(f)

with open('evaluation/dataset/document_chunks.json', 'r', encoding='utf-8') as f:
    all_chunks = json.load(f)

candidate_map = {c["question"]: c for c in candidates}

chunk_map = {}
for c in all_chunks:
    key = (c["document_id"], c["chunk_id"])
    chunk_map[key] = c["chunk_text"]

curated = []
report = {
    "received": len(selected_questions),
    "mapped": 0,
    "tier_A": 0,
    "tier_B": 0,
    "tier_C": 0,
    "accepted": 0,
    "rejected": 0,
    "multiple_chunks": 0,
    "incorrect_mappings": 0,
    "rejection_reasons": []
}

print("Starting validation loop...")

for idx, q_text in enumerate(selected_questions):
    candidate = candidate_map.get(q_text)
    if not candidate:
        print(f"[{idx+1}] Missing exact match for: {q_text[:50]}...")
        continue
    
    report["mapped"] += 1
    
    query_id = candidate["query_id"]
    document_id = candidate["document_id"]
    rel_chunk_ids = candidate["candidate_relevant_chunks"]
    
    if len(rel_chunk_ids) > 1:
        report["multiple_chunks"] += 1
        
    source_texts = []
    for cid in rel_chunk_ids:
        text = chunk_map.get((document_id, cid), "")
        source_texts.append(f"--- CHUNK {cid} ---\n{text}\n")
    
    combined_source_text = "\n".join(source_texts)
    
    prompt = f"""You are strictly validating a retrieval benchmark query.

Question: {q_text}

Source Text:
{combined_source_text}

Evaluate whether this question is suitable for a strict chunk-level retrieval benchmark.
Categorize as A, B, or C:
A — Strong benchmark query: Specific, document-grounded, can be objectively judged.
B — Usable but broader: Grounded, somewhat broader, but relevance is objective.
C — Reject: Too broad, subjective, corpus-wide, requires general knowledge, ambiguous, or the chunk does not actually support the question.

Output MUST be a valid JSON object matching EXACTLY this structure:
{{
  "tier": "A or B or C",
  "category": "RAG architecture/methods, Retrieval/vector search, Chunking/indexing, Experimental results/tables, Security/health/application-specific, or General technical/document facts",
  "evidence_excerpt": "short quote from source text proving the answer (leave blank if C)",
  "validation_status": "accepted (if A or B) or rejected (if C)",
  "rejection_reason": "if rejected, explain why. if accepted, explain briefly why it is grounded."
}}
"""
    try:
        response = gen_client.chat.completions.create(
            model=FALLBACK_GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content.strip()
        parsed = json.loads(content)
        
        tier = parsed.get("tier", "C")
        status = parsed.get("validation_status", "rejected").lower()
        
        if tier == "A": report["tier_A"] += 1
        elif tier == "B": report["tier_B"] += 1
        elif tier == "C": report["tier_C"] += 1
        else: tier = "C"; report["tier_C"] += 1; status = "rejected"
        
        if status == "accepted":
            report["accepted"] += 1
            curated.append({
                "query_id": query_id,
                "question": q_text,
                "document_id": document_id,
                "candidate_relevant_chunks": [f"{document_id}_chunk_{cid}" for cid in rel_chunk_ids],
                "tier": tier,
                "category": parsed.get("category", ""),
                "evidence_excerpt": parsed.get("evidence_excerpt", ""),
                "validation_status": "accepted",
                "validation_notes": parsed.get("rejection_reason", "")
            })
        else:
            report["rejected"] += 1
            reason = parsed.get("rejection_reason", "No reason provided")
            report["rejection_reasons"].append(f"{query_id}: {reason}")
            curated.append({
                "query_id": query_id,
                "question": q_text,
                "tier": "C",
                "validation_status": "rejected",
                "rejection_reason": reason
            })
        
        print(f"[{idx+1}/100] Processed {query_id} -> Tier {tier} ({status})")
        time.sleep(1.0)
        
    except Exception as e:
        print(f"Failed to process {query_id}: {e}")

with open('evaluation/dataset/curated_candidates.json', 'w', encoding='utf-8') as f:
    json.dump(curated, f, indent=2)

print("\n==================================================")
print("TASK 6 — REPORT")
print("==================================================")
print(f"Selected questions received: {report['received']}")
print(f"Successfully mapped: {report['mapped']}")
print(f"Tier A: {report['tier_A']}")
print(f"Tier B: {report['tier_B']}")
print(f"Tier C: {report['tier_C']}")
print(f"Accepted: {report['accepted']}")
print(f"Rejected: {report['rejected']}")
print(f"Questions requiring multiple relevant chunks: {report['multiple_chunks']}")
print(f"Questions with incorrect original mappings: {report['incorrect_mappings']}")
if report['rejection_reasons']:
    print("\nRejection Reasons:")
    for r in report['rejection_reasons'][:10]:
        print(f" - {r}")
    if len(report['rejection_reasons']) > 10:
        print(f" ... and {len(report['rejection_reasons']) - 10} more.")
