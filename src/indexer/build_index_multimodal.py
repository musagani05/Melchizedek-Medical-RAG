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
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def index_text_files(folder_path, collection, txt_model):
    for filename in os.listdir(folder_path):
        if not filename.endswith('.txt'):
            continue
        with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
            raw = f.read()
        text = preprocess_text(raw)
        chunks = chunk_text(text)
        embs = txt_model.encode(chunks).tolist()
        for i, (chunk, emb) in enumerate(zip(chunks, embs)):
            collection.add(
                ids=[f"{filename}_chunk{i}"],
                documents=[chunk],
                embeddings=[emb],
                metadatas=[{"source": filename, "type": "text"}]
            )
        print(f"[INDEX] Text: {filename}")

def index_image_files(folder_path, collection, img_model):
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue
        img = Image.open(os.path.join(folder_path, filename)).convert('RGB')
        emb = img_model.encode(img).tolist()
        collection.add(
            ids=[filename],
            documents=[f"<Image: {filename}>"],
            embeddings=[emb],
            metadatas=[{"source": filename, "type": "image"}]
        )
        print(f"[INDEX] Image: {filename}")

if __name__ == "__main__":
    client = chromadb.PersistentClient(
        path="./chroma_db",
        settings=Settings(anonymized_telemetry=False)
    )
    collection = client.get_or_create_collection("rag_medical")

    txt_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    img_model = SentenceTransformer('clip-ViT-B-32')

    index_text_files('data/pdf_texts', collection, txt_model)
    index_text_files('data/audio_texts', collection, txt_model)
    index_image_files('data/images', collection, img_model)

    print("[INDEX] Multimodal indexing selesai dan data tersimpan otomatis.")