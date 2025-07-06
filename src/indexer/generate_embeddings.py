"""
Baca JSON per dokumen, embed setiap section via SapBERT, simpan ke .npz,
menggunakan slow tokenizer untuk menghindari error konversi fast tokenizer.
"""
import os
import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, models
from dotenv import load_dotenv

# Load environment variables
dotenv_path = os.path.join("config", ".env")
load_dotenv(dotenv_path=dotenv_path)
MODEL = os.getenv("EMBEDDING_MODEL")  # e.g. cambridgeltl/SapBERT-from-PubMedBERT-fulltext

# Define paths
JSON_DIR = os.path.join("data", "pdf_texts_json")
EMB_DIR = os.path.join("data", "embeddings")
os.makedirs(EMB_DIR, exist_ok=True)

# Initialize SapBERT model with slow tokenizer to bypass fast-tokenizer conversion issues
word_emb_model = models.Transformer(
    MODEL,
    tokenizer_args={"use_fast": False}
)
pooling_model = models.Pooling(
    word_emb_model.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True
)
model = SentenceTransformer(modules=[word_emb_model, pooling_model])

# Iterate over JSON files and generate embeddings
for fname in tqdm(os.listdir(JSON_DIR), desc="Embedding JSON"):
    if not fname.endswith(".json"):
        continue
    base = fname[:-5]
    json_path = os.path.join(JSON_DIR, fname)
    with open(json_path, encoding="utf-8") as f:
        doc = json.load(f)
    # Extract section texts
    texts = [sec["content"] for sec in doc["sections"]]
    # Generate embeddings
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False
    )
    # Save embeddings with metadata
    out_path = os.path.join(EMB_DIR, base + ".npz")
    np.savez_compressed(
        out_path,
        embeddings=embeddings,
        metadata=np.array(
            [{"document": base, "section": sec["title"]} for sec in doc["sections"]],
            dtype=object
        )
    )
    print(f"Saved embeddings -> {out_path}")

print("All embeddings generated and saved in data/embeddings/")
