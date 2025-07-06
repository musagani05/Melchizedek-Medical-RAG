"""
Load embeddings .npz → tambahkan ke Chroma collection "medical_docs"
"""
import os, json
import numpy as np
import yaml
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings

# 1. Load config/env
load_dotenv(dotenv_path="config/.env")
with open("config/config.yml") as f:
    cfg = yaml.safe_load(f)
DB_PATH = cfg["vectorstore"]["path"]  # e.g. ./chroma_db

# 2. Init Chroma client & collection
client = chromadb.Client(
    Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=DB_PATH
    )
)
coll = client.get_or_create_collection(name="medical_docs")

# 3. Paths
EMB_DIR  = "data/embeddings"
JSON_DIR = "data/pdf_texts_json"

# 4. Add tiap .npz ke collection
for fn in os.listdir(EMB_DIR):
    if not fn.endswith(".npz"):
        continue
    base = fn[:-4]
    data = np.load(os.path.join(EMB_DIR, fn), allow_pickle=True)
    embs = data["embeddings"]
    metas = data["metadata"].tolist()  # list of dict: {document, section}

    # load full text untuk documents field
    doc = json.load(open(os.path.join(JSON_DIR, base + ".json"), encoding="utf-8"))
    texts = [sec["content"] for sec in doc["sections"]]
    ids   = [f"{base}_{i}" for i in range(len(texts))]

    coll.add(
        embeddings=embs.tolist(),
        documents=texts,
        metadatas=metas,
        ids=ids
    )
    print(f"[+] Indexed {base}: {len(texts)} items")

# 5. Persist DB
client.persist()
print("Vectorstore tersimpan di", DB_PATH)