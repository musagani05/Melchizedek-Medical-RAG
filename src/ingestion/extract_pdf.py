"""
Ekstraksi teks PDF → JSON per file, di-segmentasi berdasarkan heading.
"""
import os, re, json
import pdfplumber

INPUT_DIR  = "data/pdf_texts"
OUTPUT_DIR = "data/pdf_texts_json"

# Regex sederhana deteksi heading: baris huruf besar + angka/spasi
HEADING_RE = re.compile(r'^[A-Z][A-Za-z0-9 \-]{2,100}$')

def extract_sections_from_text(text):
    """Pisahkan text ke list (heading, content)."""
    lines = text.split("\n")
    sections = []
    current = {"title": "Introduction", "content": []}
    for line in lines:
        if HEADING_RE.match(line.strip()):
            # jump ke section baru
            sections.append(current)
            current = {"title": line.strip(), "content": []}
        else:
            current["content"].append(line)
    sections.append(current)
    # bersihkan: join content
    for sec in sections:
        sec["content"] = "\n".join(sec["content"]).strip()
    return sections

def process_pdf(pdf_path, out_dir):
    filename = os.path.splitext(os.path.basename(pdf_path))[0]
    print(f"[→] Ekstrak: {filename}")
    with pdfplumber.open(pdf_path) as pdf:
        full_text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    sections = extract_sections_from_text(full_text)
    out_path = os.path.join(out_dir, filename + ".json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "filename": filename,
            "num_pages": len(pdf.pages),
            "sections": sections
        }, f, ensure_ascii=False, indent=2)
    print(f"[✓] Tersimpan → {out_path}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for fn in os.listdir(INPUT_DIR):
        if fn.lower().endswith(".pdf"):
            process_pdf(os.path.join(INPUT_DIR, fn), OUTPUT_DIR)

if __name__ == "__main__":
    main()
