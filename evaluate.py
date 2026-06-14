import time
import os
import numpy as np
from PIL import Image
import fitz
from models import get_clip_model, get_blip_model, get_qwen_model
from vector_db import VectorDB
from app import summarize_text_or_image
import sys

sys.stdout.reconfigure(encoding='utf-8')

def compute_ngram_overlap(text1, text2, n=1):
    def get_ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        
    t1_tokens = [tok.lower().strip(",.!?\"'") for tok in text1.split() if tok.strip()]
    t2_tokens = [tok.lower().strip(",.!?\"'") for tok in text2.split() if tok.strip()]
    
    if len(t1_tokens) < n or len(t2_tokens) < n:
        return 0.0
        
    g1 = get_ngrams(t1_tokens, n)
    g2 = get_ngrams(t2_tokens, n)
    
    g1_counts = {}
    for g in g1:
        g1_counts[g] = g1_counts.get(g, 0) + 1
        
    overlap = 0
    for g in g2:
        if g in g1_counts and g1_counts[g] > 0:
            overlap += 1
            g1_counts[g] -= 1
            
    return overlap / len(g1)

def evaluate_system():
    print("==================================================")
    print("STARTING SYSTEM EVALUATION & METHOD JUSTIFICATION")
    print("==================================================")
    
    pdf_name = "iesc111.pdf"
    pdf_path = os.path.join("data", pdf_name)
    
    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} not found.")
        return
        
    db = VectorDB(pdf_name)
    if db.index is None:
        db.index_pdf(pdf_path)
        
    print("\n--- Running Part A: Retrieval Evaluation (Recall & MRR) ---")
    
    text_blocks = [meta for meta in db.metadata if meta["type"] == "text"]
    if len(text_blocks) < 5:
        print("Not enough text blocks in index to evaluate.")
        return
        
    eval_blocks = text_blocks[:min(10, len(text_blocks))]
    retrieval_hits = {1: 0, 3: 0, 5: 0}
    reciprocal_ranks = []
    latencies = []
    scores_correct = []
    
    for idx, item in enumerate(eval_blocks):
        original_text = item["content"]
        original_page = item["page"]
        
        words = original_text.split()
        if len(words) > 6:
            query = " ".join(words[2:min(12, len(words))])
        else:
            query = original_text
            
        print(f"Eval Query {idx+1}: '{query[:50]}...' (Expected Page: {original_page})")
        
        start = time.time()
        results = db.search(query_text=query, top_k=5)
        latencies.append(time.time() - start)
        
        rank = -1
        for rank_idx, res in enumerate(results):
            if res["id"] == item["id"] or (res["page"] == original_page and res["type"] == "text"):
                rank = rank_idx + 1
                scores_correct.append(res["score"])
                break
                
        if rank != -1:
            reciprocal_ranks.append(1.0 / rank)
            for k in [1, 3, 5]:
                if rank <= k:
                    retrieval_hits[k] += 1
        else:
            reciprocal_ranks.append(0.0)
            
    num_queries = len(eval_blocks)
    recall_1 = retrieval_hits[1] / num_queries
    recall_3 = retrieval_hits[3] / num_queries
    recall_5 = retrieval_hits[5] / num_queries
    mrr = np.mean(reciprocal_ranks)
    avg_latency = np.mean(latencies)
    avg_score = np.mean(scores_correct) if scores_correct else 0.0
    
    print("\nPart A - Retrieval Metrics:")
    print(f"  Total Queries Tested: {num_queries}")
    print(f"  Recall@1: {recall_1:.2f} (Direct Hit Rate)")
    print(f"  Recall@3: {recall_3:.2f}")
    print(f"  Recall@5: {recall_5:.2f}")
    print(f"  Mean Reciprocal Rank (MRR): {mrr:.2f}")
    print(f"  Average Query Latency: {avg_latency*1000:.1f} ms")
    print(f"  Average Cosine Similarity for correct hits: {avg_score:.3f}")
    
    print("\n--- Running Part B: Summarization Evaluation (Speed & Quality) ---")
    longest_block = max(text_blocks, key=lambda x: len(x["content"]))
    source_text = longest_block["content"]
    print(f"Source Text Length: {len(source_text)} chars ({len(source_text.split())} words)")
    
    start = time.time()
    summary, total_time = summarize_text_or_image(text=source_text)
    
    summary_len = len(summary.split())
    source_len = len(source_text.split())
    compression_ratio = summary_len / source_len if source_len > 0 else 1.0
    
    overlap_1 = compute_ngram_overlap(source_text, summary, n=1)
    overlap_2 = compute_ngram_overlap(source_text, summary, n=2)
    sentences_count = len(summary.split(". "))
    
    print("Part B - Summarization Metrics:")
    print(f"  Generated Summary:\n  {summary}\n")
    print(f"  Summary Length: {summary_len} words")
    print(f"  Sentences count (approx): {sentences_count}")
    print(f"  Time Taken to Generate: {total_time:.2f} seconds (Limit: < 60.0s)")
    print(f"  Compression Ratio: {compression_ratio:.2%}")
    print(f"  ROUGE-1 Unigram Recall: {overlap_1:.2%}")
    print(f"  ROUGE-2 Bigram Recall: {overlap_2:.2%}")
    
    success = total_time < 60.0 and (2 <= sentences_count <= 6)
    print(f"\nSummary Check Status: {'PASS' if success else 'FAIL'}")
    print("==================================================")

if __name__ == "__main__":
    evaluate_system()
