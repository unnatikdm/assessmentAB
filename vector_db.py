import os
import json
import re
import fitz
import faiss
import numpy as np
from PIL import Image
import io
import shutil
from models import generate_embedding, generate_caption

INDEX_DIR = "vector_indices"
IMAGES_DIR = "static/extracted_images"

os.makedirs(INDEX_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

def is_rtl(text: str) -> bool:
    return any('\u0600' <= c <= '\u06ff' or '\u0750' <= c <= '\u077f' or '\ufb50' <= c <= '\ufdff' or '\ufe70' <= c <= '\ufeff' for c in text)

def correct_rtl_line(line: str) -> str:
    if not is_rtl(line):
        return line
    rev = line[::-1]
    mirror_map = {
        '(': ')', ')': '(',
        '[': ']', ']': '[',
        '{': '}', '}': '{',
        '<': '>', '>': '<',
        '«': '»', '»': '«',
        '“': '”', '”': '“',
        '‘': '’', '’': '‘'
    }
    mirrored = [mirror_map.get(char, char) for char in rev]
    rev_mirrored = "".join(mirrored)
    
    def restore_ltr(match):
        return match.group(0)[::-1]
        
    corrected = re.sub(r'[a-zA-Z0-9]+(?:\s+[a-zA-Z0-9]+)*', restore_ltr, rev_mirrored)
    return corrected

def correct_rtl_text(text: str) -> str:
    if not is_rtl(text):
        return text
    lines = text.split('\n')
    corrected_lines = [correct_rtl_line(line) for line in lines]
    return '\n'.join(corrected_lines)

class VectorDB:
    def __init__(self, doc_id: str):
        self.doc_id = doc_id
        self.index_path = os.path.join(INDEX_DIR, f"{doc_id}.index")
        self.meta_path = os.path.join(INDEX_DIR, f"{doc_id}.json")
        self.index = None
        self.metadata = []
        self.load_if_exists()
        
    def load_if_exists(self) -> bool:
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                return True
            except Exception as e:
                print(f"Error loading index: {e}")
        return False

    def index_pdf(self, pdf_path: str, progress_callback=None):
        doc = fitz.open(pdf_path)
        num_pages = len(doc)
        embeddings_list = []
        metadata_list = []
        dimension = 512
        
        for page_num in range(num_pages):
            if progress_callback:
                progress_callback(page_num, num_pages, "extracting")
                
            page = doc.load_page(page_num)
            page_text = page.get_text()
            page_text = correct_rtl_text(page_text)
            
            paragraphs = [p.strip() for p in page_text.split("\n\n") if p.strip()]
            if not paragraphs and page_text.strip():
                paragraphs = [p.strip() for p in page_text.split("\n") if p.strip()]
                
            chunks = []
            current_chunk = ""
            for para in paragraphs:
                if len(current_chunk) + len(para) < 800:
                    current_chunk += "\n\n" + para if current_chunk else para
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = para
            if current_chunk:
                chunks.append(current_chunk)
                
            if not chunks and page_text.strip():
                chunks.append(page_text.strip())
                
            for chunk_idx, chunk in enumerate(chunks):
                emb = generate_embedding(chunk)
                embeddings_list.append(emb)
                metadata_list.append({
                    "id": f"{self.doc_id}_p{page_num}_t{chunk_idx}",
                    "type": "text",
                    "pdf_name": self.doc_id,
                    "page": page_num + 1,
                    "content": chunk,
                    "image_path": None
                })
                
            image_list = page.get_images(full=True)
            for img_idx, img in enumerate(image_list):
                xref = img[0]
                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    image = Image.open(io.BytesIO(image_bytes))
                    img_filename = f"{self.doc_id}_p{page_num}_img{img_idx}.{image_ext}"
                    img_save_path = os.path.join(IMAGES_DIR, img_filename)
                    image.save(img_save_path)
                    
                    relative_img_path = f"/static/extracted_images/{img_filename}"
                    caption = generate_caption(image)
                    
                    img_emb = generate_embedding(image)
                    embeddings_list.append(img_emb)
                    metadata_list.append({
                        "id": f"{self.doc_id}_p{page_num}_img{img_idx}_vis",
                        "type": "image_visual",
                        "pdf_name": self.doc_id,
                        "page": page_num + 1,
                        "content": f"[Image Description]: {caption}",
                        "image_path": relative_img_path,
                        "caption": caption
                    })
                    
                    caption_emb = generate_embedding(f"A photo of {caption}")
                    embeddings_list.append(caption_emb)
                    metadata_list.append({
                        "id": f"{self.doc_id}_p{page_num}_img{img_idx}_cap",
                        "type": "image_caption",
                        "pdf_name": self.doc_id,
                        "page": page_num + 1,
                        "content": f"[Image Caption]: {caption}",
                        "image_path": relative_img_path,
                        "caption": caption
                    })
                except Exception as ex:
                    print(f"Failed to process image xref={xref} on page {page_num + 1}: {ex}")
                    
        doc.close()
        
        if not embeddings_list:
            self.index = faiss.IndexFlatIP(dimension)
            self.metadata = []
            faiss.write_index(self.index, self.index_path)
            with open(self.meta_path, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f)
            return
            
        embeddings_matrix = np.vstack(embeddings_list).astype('float32')
        faiss.normalize_L2(embeddings_matrix)
        
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings_matrix)
        
        faiss.write_index(index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata_list, f, indent=2, ensure_ascii=False)
            
        self.index = index
        self.metadata = metadata_list

    def search(self, query_text: str = None, query_image: Image.Image = None, top_k: int = 5) -> list[dict]:
        if self.index is None or not self.metadata:
            return []
            
        query_vectors = []
        if query_text and query_text.strip():
            txt_emb = generate_embedding(query_text)
            query_vectors.append(txt_emb)
            
        if query_image is not None:
            img_emb = generate_embedding(query_image)
            query_vectors.append(img_emb)
            
        if not query_vectors:
            return []
            
        if len(query_vectors) == 2:
            combined_vector = (query_vectors[0] + query_vectors[1]) / 2.0
        else:
            combined_vector = query_vectors[0]
            
        combined_vector = combined_vector.reshape(1, -1).astype('float32')
        faiss.normalize_L2(combined_vector)
        
        scores, indices = self.index.search(combined_vector, min(top_k * 2, len(self.metadata)))
        results = []
        seen_images = set()
        
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
                
            meta = self.metadata[idx].copy()
            meta["score"] = float(score)
            
            if meta["image_path"]:
                if meta["image_path"] in seen_images:
                    continue
                seen_images.add(meta["image_path"])
                
            results.append(meta)
            if len(results) >= top_k:
                break
                
        return results

def get_indexing_status(data_dir: str) -> list[dict]:
    pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.pdf')]
    status_list = []
    
    for filename in pdf_files:
        index_path = os.path.join(INDEX_DIR, f"{filename}.index")
        meta_path = os.path.join(INDEX_DIR, f"{filename}.json")
        is_indexed = os.path.exists(index_path) and os.path.exists(meta_path)
        num_items = 0
        if is_indexed:
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta_data = json.load(f)
                    num_items = len(meta_data)
            except:
                pass
                
        status_list.append({
            "filename": filename,
            "status": "Indexed" if is_indexed else "Not Indexed",
            "embeddings_count": num_items
        })
        
    return status_list
