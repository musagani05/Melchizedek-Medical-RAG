#!/usr/bin/env python3
"""
Build and persist Chroma vectorstore from embeddings .npz files,
using config/config.yml for settings.
"""
import os
import json
import numpy as np
import yaml
from dotenv import load_dotenv
from chromadb import Client
from chromadb.config import Settings

# Load environment variables
load_dotenv(dotenv_path=os.path.join("config", ".env"))

# Load config from config/config.yml
config_path = os.path.join("config", "config.yml")
with open(config_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# Retrieve vectorstore settings
DB_PATH = cfg["vectorstore"]["path"]
COLLECTION_NAME = cfg["vectorstore"].get("collection_name", "medical_docs")

# Initialize Chroma client with new Settings API
settings = Settings(
    persist_directory=DB_PATH,
    chroma_db_impl="duckdb+parquet"
)
client = Client(settings=settings)

# Get or create the collection
coll = client.get_or_create_collection(name=COLLECTION_NAME)

# Define data directories
EMB_DIR = os.path.join("data", "embeddings")
JSON_DIR = os.path.join("data", "pdf_texts_json")

# Iterate over embeddings files and index
for fn in os.listdir(EMB_DIR):
    if not fn.endswith(".npz"):
        continue
    base = fn[:-4]
    arr = np.load(os.path.join(EMB_DIR, fn), allow_pickle=True)
    embeddings = arr["embeddings"]
    metadata = arr["metadata"].tolist()

    # Load original JSON for document content
    json_file = os.path.join(JSON_DIR, base + ".json")
    with open(json_file, "r", encoding="utf-8") as jf:
        doc = json.load(jf)
    texts = [sec["content"] for sec in doc["sections"]]
    ids = [f"{base}_{i}" for i in range(len(texts))]

    # Add entries to the collection
    coll.add(
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=metadata,
        ids=ids
    )
    print(f"[+] Indexed document {base} ({len(texts)} sections)")

# Persist the database to disk
client.persist()
print(f"Vectorstore persisted at {DB_PATH}")