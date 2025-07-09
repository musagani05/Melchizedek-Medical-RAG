#!/usr/bin/env python3
"""
Muat .env via absolute path, load config, bangun FAISS-GPU index menggunakan LangChain
"""
import os
import json
import yaml
from pathlib import Path
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from src.indexer.sapbert_embeddings import SapBERTUMLSEmbeddings

# 1. Muat .env menggunakan absolute path relatif terhadap file ini
dotenv_path = Path(__file__).resolve().parents[2] / "config" / ".env"
load_dotenv(dotenv_path=dotenv_path)

# 2. Muat config.yaml menggunakan absolute path
config_path = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
with open(config_path, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# 3. Baca direktori sumber JSON dan path penyimpanan index dari config
JSON_DIR = cfg.get("pdf_texts_dir_json", "data/pdf_texts_json")
INDEX_DIR = cfg.get("vectorstore", {}).get("path", "faiss_index")

# 4. Inisialisasi embedder SapBERT-UMLS di GPU
model_name = os.getenv("EMBEDDING_MODEL")
embedder = SapBERTUMLSEmbeddings(model_name=model_name, device="cuda")

# 5. Kumpulkan teks dan metadata
texts, metas = [], []
for fn in os.listdir(JSON_DIR):
    if not fn.endswith(".json"):
        continue
    file_path = os.path.join(JSON_DIR, fn)
    with open(file_path, encoding="utf-8") as jf:
        data = json.load(jf)
    doc_id = data.get("filename", fn[:-5])
    for sec in data.get("sections", []):
        texts.append(sec.get("content", ""))
        metas.append({"document": doc_id, "section": sec.get("title", "")})

# 6. Bangun FAISS index (GPU acceleration)
db = FAISS.from_texts(
    texts,
    embedder,
    metadatas=metas,
    faiss_mode="gpu"
)

# 7. Simpan index ke disk
os.makedirs(INDEX_DIR, exist_ok=True)
db.save_local(INDEX_DIR)
print(f"FAISS index tersimpan di {INDEX_DIR}")