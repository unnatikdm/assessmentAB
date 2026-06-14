import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer
from PIL import Image
import os

_clip_model = None
_blip_processor = None
_blip_model = None
_qwen_tokenizer = None
_qwen_model = None

CLIP_MODEL_NAME = 'sentence-transformers/clip-ViT-B-32'
BLIP_MODEL_NAME = 'Salesforce/blip-image-captioning-base'
QWEN_MODEL_NAME = 'Qwen/Qwen2.5-0.5B-Instruct'

def get_clip_model():
    global _clip_model
    if _clip_model is None:
        start = time.time()
        _clip_model = SentenceTransformer(CLIP_MODEL_NAME, device='cpu')
    return _clip_model

def get_blip_model():
    global _blip_processor, _blip_model
    if _blip_processor is None or _blip_model is None:
        start = time.time()
        _blip_processor = BlipProcessor.from_pretrained(BLIP_MODEL_NAME)
        _blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_NAME).to('cpu')
    return _blip_processor, _blip_model

def get_qwen_model():
    global _qwen_tokenizer, _qwen_model
    if _qwen_tokenizer is None or _qwen_model is None:
        start = time.time()
        _qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_NAME)
        _qwen_model = AutoModelForCausalLM.from_pretrained(QWEN_MODEL_NAME, dtype=torch.float32).to('cpu')
    return _qwen_tokenizer, _qwen_model

def generate_caption(image: Image.Image) -> str:
    processor, model = get_blip_model()
    try:
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=50)
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        print(f"Error generating caption: {e}")
        return "An extracted image from the document."

def generate_embedding(text_or_image):
    model = get_clip_model()
    embeddings = model.encode(text_or_image, convert_to_numpy=True, show_progress_bar=False)
    return embeddings

def generate_llm_response(prompt: str, system_prompt: str = "You are a helpful assistant.", max_new_tokens: int = 256) -> tuple[str, float]:
    tokenizer, model = get_qwen_model()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt")
    
    start_time = time.time()
    with torch.no_grad():
        generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens, temperature=0.7, top_p=0.9)
    generation_time = time.time() - start_time
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response, generation_time

def summarize_text_or_image(text: str = None, image: Image.Image = None, min_lines: int = 2, max_lines: int = 5) -> tuple[str, float]:
    start_time = time.time()
    context_parts = []
    
    if image is not None:
        caption = generate_caption(image)
        context_parts.append(f"[Image Content Description: {caption}]")
        
    if text and text.strip():
        words = text.split()
        if len(words) > 1500:
            text = " ".join(words[:1500]) + "..."
        context_parts.append(text)
        
    context = "\n\n".join(context_parts)
    
    prompt = f"Please summarize the following content into a succinct, clear, and informative summary of exactly {min_lines} to {max_lines} sentences (or lines). Do not include any introductory remarks like 'Here is the summary:' or extra text.\n\nContent:\n{context}"
    
    system_prompt = "You are a professional summarizer. Your output must be exactly 2-5 sentences, highly informational, and containing only the direct summary."
    
    summary, gen_time = generate_llm_response(prompt, system_prompt, max_new_tokens=90)
    total_time = time.time() - start_time
    
    return summary, total_time
