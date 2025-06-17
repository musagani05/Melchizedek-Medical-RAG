# src/ingestion/pdf_ocr.py

import os
from pdf2image import convert_from_path
import pytesseract

def ocr_pdf_to_text(pdf_path, output_txt_path):
    print(f"Processing OCR for {pdf_path} ...")
    # Convert halaman PDF jadi gambar (format PIL Image)
    pages = convert_from_path(pdf_path)
    
    full_text = ''
    for page_number, page in enumerate(pages, start=1):
        text = pytesseract.image_to_string(page, lang='eng')  # bisa ganti 'eng' sesuai bahasa
        full_text += f"\n\n--- Page {page_number} ---\n\n"
        full_text += text

    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write(full_text)

    print(f"OCR selesai dan disimpan di {output_txt_path}")

if __name__ == "__main__":
    # Folder PDF scan input (tempat kamu simpan PDF scan)
    input_folder = 'data/scan_pdfs'
    # Folder output hasil OCR text
    output_folder = 'data/pdf_ocr_texts'
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(input_folder, filename)
            output_txt_path = os.path.join(output_folder, filename[:-4] + '.txt')
            ocr_pdf_to_text(pdf_path, output_txt_path)