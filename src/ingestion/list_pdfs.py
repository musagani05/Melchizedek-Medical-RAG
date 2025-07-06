"""
List semua PDF di direktori sumber dan cek ekstensi valid.
"""
import os

PDF_DIR = "data/pdf_texts"

def list_pdfs(pdf_dir):
    files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
    if not files:
        print(f"Tidak ada PDF di {pdf_dir}.")
    else:
        print(f"Ditemukan {len(files)} PDF:")
        for fn in files:
            print("   -", fn)

if __name__ == "__main__":
    list_pdfs(PDF_DIR)