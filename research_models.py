import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from PIL import Image
import os

print("PyTorch version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())

start = time.time()
clip_model = SentenceTransformer('sentence-transformers/clip-ViT-B-32')
print(f"Loaded CLIP in {time.time() - start:.2f}s")

text_emb = clip_model.encode(["A photo of a cat", "A math set formula"])
print("Text embedding shape:", text_emb.shape)

img = Image.new('RGB', (224, 224), color = 'red')
img_emb = clip_model.encode(img)
print("Image embedding shape:", img_emb.shape)

from transformers import BlipProcessor, BlipForConditionalGeneration
start = time.time()
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
print(f"Loaded BLIP in {time.time() - start:.2f}s")

inputs = blip_processor(img, return_tensors="pt")
out = blip_model.generate(**inputs)
caption = blip_processor.decode(out[0], skip_special_tokens=True)
print("BLIP Generated Caption:", caption)

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
start = time.time()
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm_model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
print(f"Loaded Qwen in {time.time() - start:.2f}s")

prompt = "Summarize this in 1 line: Artificial Intelligence is a branch of computer science."
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer([text], return_tensors="pt")

start = time.time()
generated_ids = llm_model.generate(**model_inputs, max_new_tokens=50)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(f"Qwen response: {response}")
print(f"Inference took: {time.time() - start:.2f}s")
