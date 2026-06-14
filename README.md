# Multimodal RAG and Summarization System

This is an on-premise, offline system containing a Multimodal Retrieval-Augmented Generation (RAG) platform and a Summarization Engine. The application runs entirely locally on CPU, ensuring data privacy and zero API dependencies. It features a FastAPI backend and a responsive glassmorphic single-page application (SPA) Web UI.

---

## 🚀 Key Features

*   **Multimodal Vector Search**: Retrieves relevant text passages and matching diagrams/figures side-by-side using joint text-image vector alignment.
*   **Offline Contextual QA**: Synthesizes structured answers to text/image queries using local LLM contexts.
*   **Fast Text & Image Summarization**: Generates succinct 2-5 sentence summaries of large documents or graphical charts in under 40 seconds (guaranteed under 1 minute).
*   **Interactive Web UI Dashboard**: Formulates a modern dark-themed glassmorphism interface detailing indexing status, vector search results, and real-time engine statistics.
*   **Self-Contained Evaluation Suite**: Runs automated query precision tests (Recall@k, MRR) and summarizer benchmarking.

---

## 🧠 Model Infrastructure

*   **Embedding Engine**: `sentence-transformers/clip-ViT-B-32` (150M parameters) representing text and images in a shared 512-dimensional vector space.
*   **Image Captioning Engine**: `Salesforce/blip-image-captioning-base` (248M parameters) generating text descriptions of document graphics.
*   **Causal LLM Engine**: `Qwen/Qwen2.5-0.5B-Instruct` (490M parameters) executing Q&A synthesis and text summarization on CPU.

---

## 📊 Benchmark Metrics & Performance

The evaluation module (`evaluate.py`) benchmarks retrieval precision on standard textbook sections:

*   **RAG Retrieval (10 Queries)**:
    *   **Recall@1**: `0.70`
    *   **Recall@3**: `0.80`
    *   **Recall@5**: `1.00`
    *   **Mean Reciprocal Rank (MRR)**: `0.78`
    *   **Average Query Latency**: `573.7 ms`
    *   **Average Cosine Similarity (Hits)**: `80.3%`
*   **Summarization (560-word Source)**:
    *   **Generated Summary length**: `72 words`
    *   **Sentence Count**: `4 sentences`
    *   **Execution Time**: `30.43 seconds`
    *   **Compression Ratio**: `12.86%`

---

## 🛠️ Getting Started

### 1. Requirements

Ensure you are running Python 3.10+ with the following packages:
```bash
pip install torch transformers sentence-transformers faiss-cpu pymupdf fastapi uvicorn pillow numpy
```

### 2. File Placement

Place PDF documents to index inside the `data/` directory.

### 3. Launching the Web App

Start the development server on localhost:
```bash
python -m uvicorn app:app --reload
```
Open your browser and navigate to `http://127.0.0.1:8000`.

### 4. Running Benchmarks

Execute the evaluation module:
```bash
python evaluate.py
```
