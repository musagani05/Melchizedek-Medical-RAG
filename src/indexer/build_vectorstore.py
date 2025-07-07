#!/usr/bin/env python3
"""
Build and persist FAISS GPU vectorstore from extracted JSON sections,
using LangChain and config/config.yaml for settings.
"""
import os
import json
import yaml
from dotenv import load_dotenv
import numpy as np
import faiss

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# 1. Load environment variables
load_dotenv(dotenv_path=os.path.join("config", ".env"))

# 2. Load config
config_path = os.path.join("config", "config.yaml")
with open(config_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# 3. Retrieve vectorstore settings
INDEX_PATH = cfg["vectorstore"]["path"]

# 4. Initialize embeddings
model_name = os.getenv("EMBEDDING_MODEL")
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={"device":"cuda","trust_remote_code":True}
)

# 5. Load all sections from JSON
JSON_DIR = os.path.join("data", "pdf_texts_json")
texts = []
metadatas = []
ids = []
for fname in sorted(os.listdir(JSON_DIR)):
    if not fname.endswith(".json"):
        continue
    base = fname[:-5]
    with open(os.path.join(JSON_DIR, fname), "r", encoding="utf-8") as jf:
        doc = json.load(jf)
    for idx, sec in enumerate(doc["sections"]):
        texts.append(sec["content"])
        metadatas.append({"document": base, "section": sec["title"]})
        ids.append(f"{base}_{idx}")

# 6. Embed documents
print(f"[→] Embedding {len(texts)} sections with {model_name}...")
vectors = embeddings.embed_documents(texts)

# 7. Build FAISS GPU index
if len(vectors) == 0:
    raise ValueError("No vectors to index. Check JSON extraction step.")
dim = len(vectors[0])
res = faiss.StandardGpuResources()
cfg_gpu = faiss.GpuIndexFlatConfig()
cfg_gpu.device = int(os.getenv("FAISS_GPU_DEVICE", 0))
gpu_index = faiss.GpuIndexFlatIP(res, dim, cfg_gpu)
gpu_index.add(np.array(vectors, dtype=np.float32))
print(f"[✓] Built FAISS GPU index of dimension {dim}")

# 8. Wrap with LangChain FAISS and persist
faiss_store = FAISS.from_documents(
    documents=texts,
    embedding=embeddings,
    metadatas=metadatas,
    ids=ids,
    index=gpu_index
)
os.makedirs(INDEX_PATH, exist_ok=True)
faiss_store.save_local(INDEX_PATH)
print(f"[✓] FAISS GPU vectorstore saved to {INDEX_PATH}")
