import sys, os, json
sys.path.insert(0, os.path.abspath('.'))

from pypdf import PdfReader
from app.processing.cleaner import clean_text
from app.processing.chunker import chunk_text

# Load manifest
with open('evaluation/dataset/document_manifest.json', 'r', encoding='utf-8') as f:
    manifest = json.load(f)

all_chunks = []
stats = {
    "total_extracted_characters": 0,
    "total_chunk_count": 0,
    "chunks_per_document": {},
    "failed_documents": [],
    "low_yield_documents": []
}

print(f"Processing {len(manifest)} documents...")

for doc in manifest:
    doc_id = doc["document_id"]
    filepath = os.path.join('evaluation/dataset/docs', doc["filename"])
    
    doc_chunks = []
    chunk_counter = 0
    extracted_chars = 0
    
    try:
        reader = PdfReader(filepath)
        for page_index, page in enumerate(reader.pages):
            raw_text = page.extract_text()
            if not raw_text:
                continue
                
            cleaned_text = clean_text(raw_text)
            if not cleaned_text:
                continue
                
            extracted_chars += len(cleaned_text)
            
            # Exactly mimicking production behavior
            chunks = chunk_text(cleaned_text)
            for chunk in chunks:
                chunk_counter += 1
                doc_chunks.append({
                    "document_id": doc_id,
                    "chunk_id": chunk_counter,
                    "page_number": page_index + 1,
                    "source": doc["filename"],
                    "chunk_text": chunk
                })
        
        all_chunks.extend(doc_chunks)
        stats["total_extracted_characters"] += extracted_chars
        stats["total_chunk_count"] += chunk_counter
        stats["chunks_per_document"][doc_id] = chunk_counter
        
        if chunk_counter < 10:
            stats["low_yield_documents"].append(doc_id)
            
        print(f"Processed {doc_id}: {chunk_counter} chunks")
            
    except Exception as e:
        print(f"Failed to process {doc_id}: {e}")
        stats["failed_documents"].append(doc_id)
        stats["chunks_per_document"][doc_id] = 0

chunk_counts = list(stats["chunks_per_document"].values())
stats["min_chunks"] = min(chunk_counts) if chunk_counts else 0
stats["max_chunks"] = max(chunk_counts) if chunk_counts else 0
stats["mean_chunks"] = sum(chunk_counts) / len(chunk_counts) if chunk_counts else 0

with open('evaluation/dataset/document_chunks.json', 'w', encoding='utf-8') as f:
    json.dump(all_chunks, f, indent=2)
    
with open('evaluation/dataset/chunk_stats.json', 'w', encoding='utf-8') as f:
    json.dump(stats, f, indent=2)

print("\n--- STATS ---")
print(f"Total Extracted Characters: {stats['total_extracted_characters']}")
print(f"Total Chunk Count: {stats['total_chunk_count']}")
print(f"Min Chunks: {stats['min_chunks']}")
print(f"Max Chunks: {stats['max_chunks']}")
print(f"Mean Chunks: {stats['mean_chunks']:.2f}")
print(f"Failed: {stats['failed_documents']}")
