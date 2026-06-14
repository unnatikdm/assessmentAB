import os
import time
import base64
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List
from PIL import Image
import io

from models import (
    generate_llm_response,
    summarize_text_or_image,
    generate_embedding,
    generate_caption,
    get_clip_model,
    get_blip_model,
    get_qwen_model
)
from vector_db import VectorDB, get_indexing_status

app = FastAPI(title="Multimodal RAG and Summarization System")

os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)
os.makedirs("data", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

indexing_jobs = {}
execution_stats = {
    "search_latencies": [],
    "generation_latencies": [],
    "summary_latencies": [],
    "retrieval_similarities": [],
    "compression_ratios": []
}

class QuerySearchRequest(BaseModel):
    pdf_name: str
    query_text: Optional[str] = None
    query_image_base64: Optional[str] = None
    top_k: Optional[int] = 5

class RagRequest(BaseModel):
    query_text: Optional[str] = None
    query_image_base64: Optional[str] = None
    retrieved_contexts: List[str]

class SummarizeRequest(BaseModel):
    text: Optional[str] = None
    image_base64: Optional[str] = None

def decode_base64_image(base64_str: str) -> Image.Image:
    try:
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]
        img_bytes = base64.b64decode(base64_str)
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {e}")

def bg_index_pdf(filename: str, pdf_path: str):
    db = VectorDB(filename)
    indexing_jobs[filename] = {
        "status": "Indexing",
        "progress": "Starting extractor...",
        "current_page": 0,
        "total_pages": 0
    }
    
    def progress_callback(current, total, phase):
        indexing_jobs[filename] = {
            "status": "Indexing",
            "progress": f"{phase.capitalize()} page {current + 1} of {total}...",
            "current_page": current + 1,
            "total_pages": total
        }
        
    try:
        db.index_pdf(pdf_path, progress_callback=progress_callback)
        indexing_jobs[filename] = {
            "status": "Indexed",
            "progress": "Completed",
            "current_page": indexing_jobs[filename]["total_pages"],
            "total_pages": indexing_jobs[filename]["total_pages"]
        }
    except Exception as e:
        print(f"Error indexing {filename}: {e}")
        indexing_jobs[filename] = {
            "status": "Failed",
            "progress": str(e),
            "current_page": 0,
            "total_pages": 0
        }

@app.get("/")
def get_index():
    return FileResponse("templates/index.html")

@app.get("/api/list_documents")
def api_list_documents():
    status_list = get_indexing_status("data")
    for doc in status_list:
        fname = doc["filename"]
        if fname in indexing_jobs:
            doc["status"] = indexing_jobs[fname]["status"]
            doc["progress"] = indexing_jobs[fname]["progress"]
        else:
            doc["progress"] = "Ready" if doc["status"] == "Indexed" else "Not started"
    return status_list

@app.post("/api/index_document")
def api_index_document(request: dict, background_tasks: BackgroundTasks):
    filename = request.get("filename")
    if not filename:
        raise HTTPException(status_code=400, detail="Filename missing")
        
    pdf_path = os.path.join("data", filename)
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="File not found")
        
    if filename in indexing_jobs and indexing_jobs[filename]["status"] == "Indexing":
        return {"message": "Indexing is already running", "status": "Indexing"}
        
    background_tasks.add_task(bg_index_pdf, filename, pdf_path)
    return {"message": "Indexing started in background", "status": "Indexing"}

@app.post("/api/search")
def api_search(request: QuerySearchRequest):
    start_time = time.time()
    db = VectorDB(request.pdf_name)
    
    if db.index is None:
        raise HTTPException(status_code=400, detail="Document has not been indexed yet.")
        
    query_image = None
    if request.query_image_base64:
        query_image = decode_base64_image(request.query_image_base64)
        
    results = db.search(
        query_text=request.query_text,
        query_image=query_image,
        top_k=request.top_k
    )
    
    elapsed = time.time() - start_time
    execution_stats["search_latencies"].append(elapsed)
    
    for r in results:
        execution_stats["retrieval_similarities"].append(r["score"])
        
    return {
        "results": results,
        "search_time_sec": elapsed
    }

@app.post("/api/rag")
def api_rag(request: RagRequest):
    start_time = time.time()
    context_str = "\n\n".join([f"[Context {i+1} (Page {getattr(c, 'page', '?')})]: {c}" for i, c in enumerate(request.retrieved_contexts)])
    
    query_part = []
    if request.query_text:
        query_part.append(f"Question: {request.query_text}")
    if request.query_image_base64:
        query_img = decode_base64_image(request.query_image_base64)
        caption = generate_caption(query_img)
        query_part.append(f"Uploaded Query Image description: {caption}")
        
    query_str = "\n".join(query_part)
    
    prompt = f"Use the retrieved PDF contexts below to answer the user's question. If the information is not present in the contexts, answer as best as you can using your general knowledge but note that it was not directly found in the document.\n\nRetrieved Contexts:\n{context_str}\n\n{query_str}"
    system_prompt = "You are a highly precise Q&A bot. Answer the question using the retrieved context where possible. Keep the answer clear and concise."
    
    response_text, gen_time = generate_llm_response(prompt, system_prompt)
    
    elapsed = time.time() - start_time
    execution_stats["generation_latencies"].append(elapsed)
    
    return {
        "response": response_text,
        "generation_time_sec": elapsed
    }

@app.post("/api/summarize")
def api_summarize(request: SummarizeRequest):
    start_time = time.time()
    text_input = request.text
    query_image = None
    
    if request.image_base64:
        query_image = decode_base64_image(request.image_base64)
        
    if not text_input and not query_image:
        raise HTTPException(status_code=400, detail="Provide at least text or image to summarize.")
        
    summary_text, total_time = summarize_text_or_image(text=text_input, image=query_image)
    
    compression_ratio = 1.0
    if text_input and text_input.strip():
        source_len = len(text_input.split())
        summary_len = len(summary_text.split())
        if source_len > 0:
            compression_ratio = summary_len / source_len
            
    execution_stats["summary_latencies"].append(total_time)
    execution_stats["compression_ratios"].append(compression_ratio)
    
    return {
        "summary": summary_text,
        "time_taken_sec": total_time,
        "compression_ratio": compression_ratio
    }

@app.get("/api/metrics")
def api_metrics():
    search_lats = execution_stats["search_latencies"]
    gen_lats = execution_stats["generation_latencies"]
    sum_lats = execution_stats["summary_latencies"]
    similarities = execution_stats["retrieval_similarities"]
    compress_ratios = execution_stats["compression_ratios"]
    
    model_details = {
        "embedding_model": "CLIP ViT-B/32 (sentence-transformers, 150M params)",
        "captioning_model": "BLIP Image Captioning Base (Salesforce, 248M params)",
        "llm_model": "Qwen-2.5-0.5B-Instruct (Qwen, 490M params)",
        "hardware_accelerator": "CPU-only (Local Inference)"
    }
    
    return {
        "averages": {
            "search_latency_sec": sum(search_lats) / len(search_lats) if search_lats else 0.0,
            "generation_latency_sec": sum(gen_lats) / len(gen_lats) if gen_lats else 0.0,
            "summary_latency_sec": sum(sum_lats) / len(sum_lats) if sum_lats else 0.0,
            "retrieval_similarity": sum(similarities) / len(similarities) if similarities else 0.0,
            "compression_ratio": sum(compress_ratios) / len(compress_ratios) if compress_ratios else 0.0,
        },
        "details": {
            "search_latencies": search_lats,
            "generation_latencies": gen_lats,
            "summary_latencies": sum_lats,
            "retrieval_similarities": similarities,
            "compression_ratios": compress_ratios,
        },
        "model_details": model_details
    }

@app.get("/api/warmup")
def api_warmup():
    get_clip_model()
    get_blip_model()
    get_qwen_model()
    return {"status": "Models loaded successfully"}
