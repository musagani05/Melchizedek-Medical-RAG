import os, json, yaml
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from src.indexer.sapbert_embeddings import SapBERTUMLSEmbeddings
import faiss

# 1. Load environment & config
load_dotenv(dotenv_path="config/.env")
cfg = yaml.safe_load(open("config/config.yaml"))
JSON_DIR  = cfg["pdf_texts_dir_json"]    # e.g. "../data/pdf_texts_json"
INDEX_DIR = cfg["vectorstore"]["path"]   # e.g. "../faiss_index"

# 2. Muat custom embedder
embedder = SapBERTUMLSEmbeddings(
    model_name=os.getenv("EMBEDDING_MODEL"),
    device="cuda"
)

# 3. Kumpulkan teks & metadata
texts, metas = [], []
for fn in os.listdir(JSON_DIR):
    if not fn.endswith(".json"):
        continue
    doc = json.load(open(os.path.join(JSON_DIR, fn), encoding="utf-8"))
    for sec in doc["sections"]:
        texts.append(sec["content"])
        metas.append({"document": doc["filename"], "section": sec["title"]})

# 4. Bangun FAISS index di CPU
db = FAISS.from_texts(
    texts,
    embedding=embedder,
    metadatas=metas,
    index_factory="Flat"    # pilih Flat (L2) atau IVF… sesuai kebutuhan
)

# 5. Konversi index ke GPU
res = faiss.StandardGpuResources()        # inisialisasi resource GPU
cpu_index = db.index                      # faiss.IndexFlatL2
gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
db.index = gpu_index                      # gantikan dengan GPU index

# 6. Persist index
os.makedirs(INDEX_DIR, exist_ok=True)
db.save_local(INDEX_DIR)
print(f"FAISS-GPU index tersimpan di {INDEX_DIR}")