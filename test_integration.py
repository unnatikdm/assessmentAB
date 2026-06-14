import requests
import json
import base64
import time

BASE_URL = "http://127.0.0.1:8000"

def test_integration():
    print("==================================================")
    print("STARTING END-TO-END INTEGRATION TEST")
    print("==================================================")
    
    print("\n1. Warming up models...")
    res = requests.get(f"{BASE_URL}/api/warmup")
    print("Response status:", res.status_code)
    print("Response JSON:", res.json())
    
    print("\n2. Listing documents...")
    res = requests.get(f"{BASE_URL}/api/list_documents")
    print("Response status:", res.status_code)
    docs = res.json()
    print("Documents listed:")
    for doc in docs:
        if doc["status"] == "Indexed":
            print(f"  - {doc['filename']}: {doc['status']} ({doc['embeddings_count']} embeddings)")
            
    print("\n3. Testing Multimodal Search...")
    search_payload = {
        "pdf_name": "iesc111.pdf",
        "query_text": "What are the characteristics of a sound wave?",
        "top_k": 3
    }
    start = time.time()
    res = requests.post(f"{BASE_URL}/api/search", json=search_payload)
    search_time = time.time() - start
    print("Response status:", res.status_code)
    search_data = res.json()
    print(f"Search completed in {search_time:.2f}s")
    
    results = search_data.get("results", [])
    print(f"Retrieved {len(results)} contexts:")
    retrieved_texts = []
    for idx, r in enumerate(results):
        print(f"  [{idx+1}] Page {r['page']} | Type: {r['type']} | Score: {r['score']*100:.1f}%")
        print(f"      Content: {r['content'][:150]}...")
        retrieved_texts.append(r["content"])
        
    print("\n4. Testing RAG Text Generation...")
    rag_payload = {
        "query_text": "What are the characteristics of a sound wave?",
        "retrieved_contexts": retrieved_texts
    }
    start = time.time()
    res = requests.post(f"{BASE_URL}/api/rag", json=rag_payload)
    rag_time = time.time() - start
    print("Response status:", res.status_code)
    rag_data = res.json()
    print(f"RAG completed in {rag_time:.2f}s")
    print("Generated Answer:\n", rag_data.get("response"))
    
    print("\n5. Testing Summarization...")
    summary_source = (
        "Everyday we hear sounds from various sources like humans, birds, bells, machines, "
        "vehicles, televisions, radios etc. Sound is a form of energy which produces a "
        "sensation of hearing in our ears. There are also other forms of energy like mechanical "
        "energy, light energy, etc. We have talked about mechanical energy in the previous "
        "chapters. You have been taught about conservation of energy, which states that we cannot "
        "create or destroy energy. We can only change it from one form to another."
    )
    summarize_payload = {
        "text": summary_source
    }
    start = time.time()
    res = requests.post(f"{BASE_URL}/api/summarize", json=summarize_payload)
    sum_time = time.time() - start
    print("Response status:", res.status_code)
    sum_data = res.json()
    print(f"Summarization completed in {sum_time:.2f}s")
    print("Generated Summary:\n", sum_data.get("summary"))
    print("Compression Ratio:", sum_data.get("compression_ratio"))
    
    print("\n6. Fetching Performance Metrics...")
    res = requests.get(f"{BASE_URL}/api/metrics")
    print("Response status:", res.status_code)
    metrics = res.json()
    print("Averages:")
    print(json.dumps(metrics.get("averages"), indent=2))
    
    print("==================================================")
    print("TEST COMPLETED SUCCESSFULLY!")
    print("==================================================")

if __name__ == "__main__":
    test_integration()
