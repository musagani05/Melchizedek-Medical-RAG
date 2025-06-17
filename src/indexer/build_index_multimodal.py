# src/indexer/multimodal_indexer.py
import os
import re
from PIL import Image
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

def preprocess_text(text):
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = ' '.join(text.split())
    return text.strip()

def chunk_text(text, chunk_size=500, overlap=50):
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def index_text_files(folder_path, collection, txt_model):
    for fn in os.listdir(folder_path):
        if not fn.endswith('.txt'):
            continue
        with open(os.path.join(folder_path, fn), encoding='utf-8') as f:
            raw = f.read()
        text   = preprocess_text(raw)
        chunks = chunk_text(text)
        embs   = txt_model.encode(chunks).tolist()
        for i, (c, e) in enumerate(zip(chunks, embs)):
            collection.add(
                ids=[f"{fn}_chunk{i}"],
                documents=[c],
                embeddings=[e],
                metadatas=[{"source": fn, "type": "text"}]
            )
    print("[INDEX] Text indexing done.")

def index_image_files(folder_path, collection, img_model):
    for fn in os.listdir(folder_path):
        if not fn.lower().endswith(('.png','.jpg','jpeg','bmp')):
            continue
        img = Image.open(os.path.join(folder_path, fn)).convert('RGB')
        emb = img_model.encode(img).tolist()
        collection.add(
            ids=[fn],
            documents=[f"<Image: {fn}>"],
            embeddings=[emb],
            metadatas=[{"source": fn, "type": "image"}]
        )
    print("[INDEX] Image indexing done.")

def build_multimodal_index(
    pdf_folder: str = "data/pdf_texts",
    audio_folder: str = "data/audio_texts",
    image_folder: str = "data/images",
    chroma_path: str = "chroma_db",
    collection_name: str = "RAG-MEDICAL"
):
    # 1. Buat client & collection persistent
    client = chromadb.PersistentClient(
        path=chroma_path,
        settings=Settings(anonymized_telemetry=False)
    )
    coll   = client.get_or_create_collection(collection_name)

    # 2. Load model
    txt_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    img_model = SentenceTransformer('clip-ViT-B-32')

    # 3. Index semua modalitas
    os.makedirs(chroma_path, exist_ok=True)
    index_text_files(pdf_folder, coll, txt_model)
    index_text_files(audio_folder, coll, txt_model)
    index_image_files(image_folder, coll, img_model)

    print("[INDEX] Multimodal index build complete.")