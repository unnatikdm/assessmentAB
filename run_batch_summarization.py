import os
import sys
import json
import time
import shutil

# Make sure we can import from the project directory
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

# stdout encoding fix (Windows)
sys.stdout.reconfigure(encoding="utf-8")

import fitz  # PyMuPDF
from vector_db import correct_rtl_text
from models import summarize_text_or_image

def extract_text_from_pdf(pdf_path: str) -> tuple[str, int]:
    """Return (full_text, page_count) extracted from a PDF with RTL correction."""
    doc = fitz.open(pdf_path)
    pages = len(doc)
    texts = []
    for page in doc:
        text = page.get_text()
        text = correct_rtl_text(text)
        texts.append(text)
    doc.close()
    return "\n".join(texts), pages

def compute_ngram_overlap(text1: str, text2: str, n: int = 1) -> float:
    """ROUGE-n unigram/bigram precision recall (overlap / len(text1 ngrams))."""
    def get_ngrams(tokens, n):
        return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

    t1 = [tok.lower().strip(",.!?\"'") for tok in text1.split() if tok.strip()]
    t2 = [tok.lower().strip(",.!?\"'") for tok in text2.split() if tok.strip()]

    if len(t1) < n or len(t2) < n:
        return 0.0

    g1 = get_ngrams(t1, n)
    g2 = get_ngrams(t2, n)

    counts = {}
    for g in g1:
        counts[g] = counts.get(g, 0) + 1

    overlap = 0
    for g in g2:
        if g in counts and counts[g] > 0:
            overlap += 1
            counts[g] -= 1

    return overlap / len(g1)

def word_count(text: str) -> int:
    return len(text.split())

def main():
    data_dir = os.path.join(PROJECT_DIR, "data")
    out_dir = os.path.join(PROJECT_DIR, "summaries_output")
    os.makedirs(out_dir, exist_ok=True)

    pdf_files = sorted(
        [f for f in os.listdir(data_dir) if f.lower().endswith(".pdf")]
    )

    if not pdf_files:
        print("No PDF files found in data/. Exiting.")
        return

    print(f"Found {len(pdf_files)} PDFs. Starting summarization...\n")
    print("NOTE: The Qwen model will be loaded on the first call – this may take a minute.\n")

    results = []
    total_start = time.time()

    for idx, fname in enumerate(pdf_files, 1):
        pdf_path = os.path.join(data_dir, fname)
        print(f"[{idx:02d}/{len(pdf_files)}] Processing {fname} ...", end=" ", flush=True)

        # ── 1. Extract text ───────────────────────────────────────────────────
        try:
            full_text, page_count = extract_text_from_pdf(pdf_path)
        except Exception as e:
            print(f"ERROR extracting text: {e}")
            results.append(
                {
                    "filename": fname,
                    "status": "error",
                    "error": str(e),
                }
            )
            continue

        src_words = word_count(full_text)

        # ── 2. Summarize ──────────────────────────────────────────────────────
        try:
            summary, gen_time = summarize_text_or_image(text=full_text)
        except Exception as e:
            print(f"ERROR summarizing: {e}")
            results.append(
                {
                    "filename": fname,
                    "status": "error",
                    "page_count": page_count,
                    "source_word_count": src_words,
                    "error": str(e),
                }
            )
            continue

        # ── 3. Metrics ────────────────────────────────────────────────────────
        sum_words = word_count(summary)
        compression_ratio = round(sum_words / src_words, 4) if src_words > 0 else 1.0
        rouge1 = round(compute_ngram_overlap(full_text, summary, n=1), 4)
        rouge2 = round(compute_ngram_overlap(full_text, summary, n=2), 4)
        sentence_count = len([s for s in summary.split(". ") if s.strip()])

        file_size_kb = round(os.path.getsize(pdf_path) / 1024, 1)

        entry = {
            "filename": fname,
            "status": "ok",
            "file_size_kb": file_size_kb,
            "page_count": page_count,
            "source_word_count": src_words,
            "summary_word_count": sum_words,
            "sentence_count": sentence_count,
            "compression_ratio": compression_ratio,
            "rouge1": rouge1,
            "rouge2": rouge2,
            "generation_time_sec": round(gen_time, 2),
            "summary": summary,
        }
        results.append(entry)
        print(f"done in {gen_time:.1f}s  ({src_words}→{sum_words} words, ROUGE-1={rouge1:.2%})")

    total_elapsed = time.time() - total_start
    ok_results = [r for r in results if r.get("status") == "ok"]

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    def avg(lst):
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    aggregate = {
        "total_pdfs": len(pdf_files),
        "successful": len(ok_results),
        "failed": len(pdf_files) - len(ok_results),
        "total_time_sec": round(total_elapsed, 2),
        "avg_generation_time_sec": avg([r["generation_time_sec"] for r in ok_results]),
        "avg_compression_ratio": avg([r["compression_ratio"] for r in ok_results]),
        "avg_rouge1": avg([r["rouge1"] for r in ok_results]),
        "avg_rouge2": avg([r["rouge2"] for r in ok_results]),
        "avg_source_words": avg([r["source_word_count"] for r in ok_results]),
        "avg_summary_words": avg([r["summary_word_count"] for r in ok_results]),
    }

    # ── Save JSON ─────────────────────────────────────────────────────────────
    json_path = os.path.join(out_dir, "summaries_data.json")
    payload = {"aggregate_metrics": aggregate, "per_pdf": results}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\n✓ JSON saved → {json_path}")

    # ── Save Markdown ─────────────────────────────────────────────────────────
    md_path = os.path.join(out_dir, "summaries_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# PDF Summaries & Metrics Report\n\n")
        f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

        # Aggregate table
        f.write("## Aggregate Metrics\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| Total PDFs processed | {aggregate['total_pdfs']} |\n")
        f.write(f"| Successful | {aggregate['successful']} |\n")
        f.write(f"| Failed | {aggregate['failed']} |\n")
        f.write(f"| Total processing time | {aggregate['total_time_sec']} s |\n")
        f.write(f"| Avg generation time / PDF | {aggregate['avg_generation_time_sec']} s |\n")
        f.write(f"| Avg source word count | {aggregate['avg_source_words']} |\n")
        f.write(f"| Avg summary word count | {aggregate['avg_summary_words']} |\n")
        f.write(f"| Avg compression ratio | {aggregate['avg_compression_ratio']:.2%} |\n")
        f.write(f"| Avg ROUGE-1 | {aggregate['avg_rouge1']:.2%} |\n")
        f.write(f"| Avg ROUGE-2 | {aggregate['avg_rouge2']:.2%} |\n\n")

        # Per-PDF metrics overview table
        f.write("---\n\n")
        f.write("## Per-PDF Metrics Overview\n\n")
        f.write("| # | Filename | Pages | Src Words | Sum Words | Compress | ROUGE-1 | ROUGE-2 | Time (s) |\n")
        f.write("|---|----------|-------|-----------|-----------|----------|---------|---------|----------|\n")
        for i, r in enumerate(results, 1):
            if r.get("status") == "ok":
                f.write(
                    f"| {i} | {r['filename']} | {r['page_count']} | {r['source_word_count']} "
                    f"| {r['summary_word_count']} | {r['compression_ratio']:.2%} "
                    f"| {r['rouge1']:.2%} | {r['rouge2']:.2%} | {r['generation_time_sec']} |\n"
                )
            else:
                f.write(f"| {i} | {r['filename']} | — | — | — | — | — | — | ERROR |\n")

        # Individual summaries
        f.write("\n---\n\n")
        f.write("## Individual PDF Summaries\n\n")
        for i, r in enumerate(results, 1):
            f.write(f"### {i}. `{r['filename']}`\n\n")
            if r.get("status") == "ok":
                f.write(f"- **File size**: {r['file_size_kb']} KB\n")
                f.write(f"- **Pages**: {r['page_count']}\n")
                f.write(f"- **Source words**: {r['source_word_count']}\n")
                f.write(f"- **Summary words**: {r['summary_word_count']} ({r['compression_ratio']:.2%} of source)\n")
                f.write(f"- **ROUGE-1**: {r['rouge1']:.2%}  |  **ROUGE-2**: {r['rouge2']:.2%}\n")
                f.write(f"- **Generation time**: {r['generation_time_sec']} s\n\n")
                f.write(f"**Summary:**\n\n> {r['summary']}\n\n")
            else:
                f.write(f"> ⚠️ Error: {r.get('error', 'Unknown error')}\n\n")
            f.write("---\n\n")

    print(f"✓ Markdown report saved → {md_path}")
    
    # ── Remove old pdf_summaries directory ────────────────────────────────────
    old_dir = os.path.join(PROJECT_DIR, "pdf_summaries")
    if os.path.exists(old_dir):
        try:
            shutil.rmtree(old_dir)
            print("✓ Removed old pdf_summaries directory")
        except Exception as e:
            print(f"⚠️ Error removing old pdf_summaries directory: {e}")

    print(f"\n{'='*60}")
    print("AGGREGATE METRICS SUMMARY")
    print(f"{'='*60}")
    print(f"  PDFs processed     : {aggregate['successful']} / {aggregate['total_pdfs']}")
    print(f"  Total time         : {aggregate['total_time_sec']} s")
    print(f"  Avg time / PDF     : {aggregate['avg_generation_time_sec']} s")
    print(f"  Avg compression    : {aggregate['avg_compression_ratio']:.2%}")
    print(f"  Avg ROUGE-1        : {aggregate['avg_rouge1']:.2%}")
    print(f"  Avg ROUGE-2        : {aggregate['avg_rouge2']:.2%}")
    print(f"{'='*60}")
    print("\nDone! Check summaries_output/ for both outputs.")

if __name__ == "__main__":
    main()
