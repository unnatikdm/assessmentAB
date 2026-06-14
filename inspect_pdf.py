import fitz
import sys

sys.stdout.reconfigure(encoding='utf-8')

def inspect_pdf(pdf_path):
    print(f"Inspecting PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    print(f"Number of pages: {len(doc)}")
    
    for page_num in range(min(5, len(doc))):
        page = doc.load_page(page_num)
        text = page.get_text()
        print(f"\n--- Page {page_num + 1} ---")
        print(f"Text length: {len(text)}")
        print(f"Sample text: {text[:200]}...")
        
        image_list = page.get_images(full=True)
        print(f"Number of images on page: {len(image_list)}")
        for img_idx, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            print(f"  Image {img_idx + 1}: xref={xref}, ext={image_ext}, size={len(image_bytes)} bytes")

if __name__ == "__main__":
    for path in ["data/kemh101.pdf", "data/kham101.pdf", "data/iesc111.pdf"]:
        try:
            inspect_pdf(path)
        except Exception as e:
            print(f"Error inspecting {path}: {e}")
