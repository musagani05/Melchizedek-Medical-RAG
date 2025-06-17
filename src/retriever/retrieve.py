import argparse
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# 1. PersistentClient pointing ke folder yang sama
client = chromadb.PersistentClient(
    path="./chroma_db",
    settings=Settings(anonymized_telemetry=False)
)
collection = client.get_or_create_collection("rag_medical")

embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def retrieve_context(query: str, top_k: int = 5):
    q_emb = embed_model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=['documents', 'metadatas']
    )

    hits = []
    for i, doc in enumerate(results['documents'][0]):
        src = results['metadatas'][0][i].get('source', 'unknown')
        hits.append({'chunk': doc, 'source': src})

    # Debug: Pastikan kita benar-benar mendapat potongan
    print(f"[DEBUG] Retrieved {len(hits)} chunks for query “{query}”:")
    for h in hits:
        snippet = h['chunk'][:60].replace("\n", " ")
        print(f"  - {h['source']} → {snippet} …")

    return hits

def main():
    parser = argparse.ArgumentParser(
        description="Retrieve top-k chunks from your Chroma vector store"
    )
    parser.add_argument(
        "--query", "-q", type=str, required=True,
        help="Text query untuk mencari dokumen"
    )
    parser.add_argument(
        "--top_k", "-k", type=int, default=5,
        help="Jumlah hasil teratas yang ingin ditampilkan"
    )
    args = parser.parse_args()

    # Panggil retrieval dan otomatis cetak debug output
    retrieve_context(args.query, args.top_k)

if __name__ == "__main__":
    main()