import os, json, sys
from pypdf import PdfReader

sys.path.insert(0, os.path.abspath('.'))
from app.processing.cleaner import clean_text
from app.processing.chunker import chunk_text

SECONDARY_PDFS_DIR = 'evaluation/dataset/secondary_pdfs'
OUTPUT_CHUNKS_FILE = 'evaluation/dataset/secondary_chunks.json'

# We need exactly 7,332 chunks from the secondary corpus to hit 10,000 total.
TARGET_CHUNKS = 7332

def main():
    if not os.path.exists(SECONDARY_PDFS_DIR):
        print(f"Directory not found: {SECONDARY_PDFS_DIR}")
        return
        
    pdf_files = [f for f in os.listdir(SECONDARY_PDFS_DIR) if f.endswith('.pdf')]
    print(f"Found {len(pdf_files)} PDFs in secondary corpus.")
    
    all_chunks = []
    
    for pdf_file in pdf_files:
        if len(all_chunks) >= TARGET_CHUNKS:
            break
            
        doc_id = f"sec_doc_{pdf_file.replace('.pdf', '')}"
        pdf_path = os.path.join(SECONDARY_PDFS_DIR, pdf_file)
        print(f"Processing {pdf_file}...")
        
        try:
            reader = PdfReader(pdf_path)
            chunk_counter = 0
            for page in reader.pages:
                raw_text = page.extract_text()
                if not raw_text: continue
                cleaned_text = clean_text(raw_text)
                if not cleaned_text: continue
                
                chunks = chunk_text(cleaned_text)
                for chunk in chunks:
                    if len(all_chunks) >= TARGET_CHUNKS:
                        break
                    
                    chunk_counter += 1
                    all_chunks.append({
                        "document_id": doc_id,
                        "chunk_id": chunk_counter,
                        "chunk_text": chunk
                    })
                if len(all_chunks) >= TARGET_CHUNKS:
                    break
        except Exception as e:
            print(f"Failed to process {pdf_file}: {e}")
            
    print(f"\nGenerated {len(all_chunks)} secondary chunks.")
    
    if len(all_chunks) < TARGET_CHUNKS:
        print(f"WARNING: Need {TARGET_CHUNKS} chunks, but only got {len(all_chunks)}.")
    
    with open(OUTPUT_CHUNKS_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=2)
        
    print(f"Saved to {OUTPUT_CHUNKS_FILE}")

if __name__ == "__main__":
    main()
