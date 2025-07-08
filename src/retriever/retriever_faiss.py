"""
Retriever for FAISS GPU index using LangChain and SapBERT embeddings.
"""
import os
import yaml
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document

# 1. Load environment variables
load_dotenv(dotenv_path=os.path.join("config", ".env"))

# 2. Load configuration
config_path = os.path.join("config", "config.yaml")
with open(config_path, "r", encoding="utf-8") as cfg_file:
    cfg = yaml.safe_load(cfg_file)

# 3. Retrieve vectorstore path
INDEX_PATH = cfg["vectorstore"]["path"]  # e.g., data/faiss_index

# 4. Initialize embeddings
model_name = os.getenv("EMBEDDING_MODEL")
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={"device": "cuda", "trust_remote_code": True}
)

# 5. Load FAISS index from disk
db = FAISS.load_local(INDEX_PATH, embeddings)


def retrieve(query: str, k: int = 5) -> list[Document]:
    """
    Perform semantic search and return top-k LangChain Document objects.

    Args:
        query (str): Input query string.
        k (int): Number of top documents to return.

    Returns:
        List of Document(page_content, metadata).
    """
    docs = db.similarity_search(query, k=k)
    return docs


# Smoke test when run as script
def main():
    q = "nyeri dada saat beraktivitas"
    print(f"[→] Query: {q}")
    hits = retrieve(q, k=3)
    for i, doc in enumerate(hits, 1):
        print(f"\n[Result {i}]\nMetadata: {doc.metadata}\nContent snippet: {doc.page_content[:200].replace(chr(10), ' ')}...\n")


if __name__ == "__main__":
    main()