import fitz  # PyMuPDF
import os

def extract_images(pdf_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    for page_index in range(len(doc)):
        images = doc.get_page_images(page_index)
        for img_index, img_info in enumerate(images):
            xref = img_info[0]
            img_dict = doc.extract_image(xref)
            img_bytes = img_dict["image"]
            ext = img_dict["ext"]
            img_name = f"{base_name}_p{page_index+1}_img{img_index}.{ext}"
            img_path = os.path.join(output_dir, img_name)
            with open(img_path, "wb") as img_file:
                img_file.write(img_bytes)
    print(f"Extracted images from {pdf_path}")

if __name__ == "__main__":
    for folder in ['data/slides', 'data/articles']:
        for filename in os.listdir(folder):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(folder, filename)
                extract_images(pdf_path, 'data/images')