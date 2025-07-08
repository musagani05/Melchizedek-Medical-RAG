import os
import re
import fitz  # PyMuPDF
from PIL import Image
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Parameter untuk chunking
MAX_TOKENS = 1024  # perkiraan token (kata) per chunk
OVERLAP_TOKENS = 128  # tumpang tindih antar chunk


def parse_toc(pdf_path):
    """
    Ambil Table of Contents (TOC) dari PDF.
    Mengembalikan list of tuples: (level, title, page)
    """
    doc = fitz.open(pdf_path)
    toc = doc.get_toc()  # [[level, title, page], ...]
    doc.close()
    return toc


def get_heading_for_page(toc, page_num):
    """
    Tentukan chapter dan section untuk halaman tertentu berdasar TOC.
    """
    chapter = None
    section = None
    for level, title, pg in toc:
        if pg <= page_num:
            if level == 1:
                chapter = title
            elif level == 2:
                section = title
        else:
            break
    return chapter or "–", section or "–"


def extract_structured_text(pdf_path):
    """
    Ekstrak setiap paragraf dari PDF dengan metadata struktur:
      - book: nama file tanpa ekstensi
      - chapter & section dari TOC
      - page: label halaman PDF (sesuai tampilan buku)
      - text: isi paragraf
    Return: list of dict
    """
    book_name = os.path.splitext(os.path.basename(pdf_path))[0]
    toc = parse_toc(pdf_path)
    doc = fitz.open(pdf_path)
    sections = []

    for p in range(len(doc)):
        page = doc.load_page(p)
        raw_label = page.get_label()
        page_label = raw_label if raw_label else str(p + 1)
        chapter, section = get_heading_for_page(toc, p + 1)
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if b["type"] != 0:
                continue
            for line in b["lines"]:
                span = " ".join(s["text"] for s in line["spans"]).strip()
                if span:
                    sections.append({
                        "book":    book_name,
                        "chapter": chapter,
                        "section": section,
                        "page":    page_label,
                        "text":    span
                    })
    doc.close()
    return sections


def chunk_by_structure(units, max_tokens=MAX_TOKENS, overlap_tokens=OVERLAP_TOKENS):
    """
    Boundary-aware chunking: kumpulkan paragraf hingga mendekati max_tokens,
    dengan overlap antar chunk.
    """
    chunks = []
    current_units = []
    current_tokens = 0

    for unit in units:
        tokens = len(unit["text"].split())
        if current_tokens + tokens <= max_tokens:
            current_units.append(unit)
            current_tokens += tokens
        else:
            chunks.append(current_units.copy())
            overlap_units = []
            cum_tokens = 0
            for u in reversed(current_units):
                t = len(u["text"].split())
                if cum_tokens + t > overlap_tokens:
                    break
                overlap_units.insert(0, u)
                cum_tokens += t
            current_units = overlap_units.copy()
            current_tokens = cum_tokens
            current_units.append(unit)
            current_tokens += tokens

    if current_units:
        chunks.append(current_units)
    return chunks


def index_pdf_files(pdf_folder, collection, txt_model):
    """
    Index PDF dengan chunking boundary-aware dan metadata akurat dari TOC.
    """
    for fn in os.listdir(pdf_folder):
        if not fn.lower().endswith(".pdf"):
            continue
        path = os.path.join(pdf_folder, fn)
        units = extract_structured_text(path)
        batches = chunk_by_structure(units)

        for i, batch in enumerate(batches):
            chunk_text = " ".join(u["text"] for u in batch)
            emb = txt_model.encode(chunk_text).tolist()

            # Flatten metadata lists into strings
            chapters = {u["chapter"] for u in batch if u["chapter"] != "–"}
            sections = {u["section"] for u in batch if u["section"] != "–"}
            pages = {u["page"] for u in batch}

            metas = {
                "source":   fn,
                "book":     batch[0]["book"],
                "chapters": ", ".join(chapters) if chapters else "–",
                "sections": ", ".join(sections) if sections else "–",
                "pages":    ", ".join(str(p) for p in pages),
                "type":     "pdf_chunk"
            }
            collection.add(
                ids=[f"{fn}_chunk{i}"],
                documents=[chunk_text],
                embeddings=[emb],
                metadatas=[metas]
            )
        print(f"[INDEX] PDF: {fn} → {len(batches)} chunks")


def index_image_files(image_folder, collection, img_model):
    """
    Index gambar sebagai pseudo-dokumen.
    """
    for fn in os.listdir(image_folder):
        ext = fn.lower().split('.')[-1]
        if ext not in ("png","jpg","jpeg","bmp"):
            continue
        path = os.path.join(image_folder, fn)
        img = Image.open(path).convert('RGB')
        emb = img_model.encode(img).tolist()
        metadata = {"source": fn, "type": "image"}
        collection.add(
            ids=[fn],
            documents=[f"<Image: {fn}>"],
            embeddings=[emb],
            metadatas=[metadata]
        )
        print(f"[INDEX] Image {fn}")


def build_multimodal_index(
    pdf_folder: str = "data/articles",
    image_folder: str = "data/images",
    chroma_path: str = "chroma_db",
    collection_name: str = "rag_medical"
):
    """
    Entry point: bangun index multimodal menggunakan metadata dari TOC
    dan label halaman PDF.
    """
    client = chromadb.PersistentClient(
        path=chroma_path,
        settings=Settings(anonymized_telemetry=False)
    )
    coll = client.get_or_create_collection(collection_name)

    txt_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    img_model = SentenceTransformer('clip-ViT-B-32')

    os.makedirs(chroma_path, exist_ok=True)

    index_pdf_files(pdf_folder, coll, txt_model)
    index_image_files(image_folder, coll, img_model)
