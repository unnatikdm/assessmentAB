import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

from vector_db import VectorDB, get_indexing_status

def test_db():
    pdf_name = "iesc111.pdf"
    pdf_path = os.path.join("data", pdf_name)
    
    print("Checking document store status...")
    status = get_indexing_status("data")
    print("Dataset status:", status)
    
    db = VectorDB(pdf_name)
    if db.index is None:
        print(f"Indexing {pdf_name} now...")
        db.index_pdf(pdf_path)
    else:
        print(f"Index already exists for {pdf_name} with {len(db.metadata)} items.")
        
    print("\n--- Running Test Search Query ---")
    query = "How is sound produced by vibrating objects?"
    print(f"Query: '{query}'")
    
    results = db.search(query_text=query, top_k=3)
    print(f"Search completed. Retrieved {len(results)} results:")
    for i, r in enumerate(results):
        print(f"\nResult {i+1} [Similarity: {r['score']*100:.1f}%] [Page: {r['page']}] [Type: {r['type']}]")
        print(f"Content: {r['content'][:300]}...")
        if r['image_path']:
            print(f"Image Path: {r['image_path']}")

if __name__ == "__main__":
    test_db()
