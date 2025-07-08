import argparse
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np
import subprocess
import re
import os

# Inisialisasi ChromaDB client & collection
def init_collection():
    client = chromadb.PersistentClient(
        path="./chroma_db",
        settings=Settings(anonymized_telemetry=False)
    )
    return client.get_or_create_collection("rag_medical")

collection = init_collection()
embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def retrieve_and_rerank(query: str, coarse_k: int = 20, final_k: int = 5) -> list:
    q_emb = embed_model.encode(query)
    results = collection.query(
        query_embeddings=[q_emb.tolist()],
        n_results=coarse_k,
        include=["documents","metadatas","embeddings"]
    )
    docs, metas, embs = (
        results['documents'][0],
        results['metadatas'][0],
        results['embeddings'][0]
    )
    scores = [(cosine_similarity(q_emb, np.array(e)), i) for i, e in enumerate(embs)]
    scores.sort(key=lambda x: x[0], reverse=True)
    hits = []
    for rank_idx, (score, idx) in enumerate(scores[:final_k], start=1):
        m = metas[idx]
        chapter = m.get('chapters') or m.get('chapter') or '–'
        section = m.get('sections') or m.get('section') or '–'
        hits.append({
            'chunk': docs[idx],
            'metadata': m,
            'book': m.get('book') or m.get('source'),
            'chapters': chapter,
            'sections': section,
            'pages': m.get('pages') or '–',
            'score': score,
            'rank': rank_idx
        })
    return hits


def build_prompt(query: str, hits: list) -> str:
    system = (
        "Anda adalah asisten medis. Jawab dengan parafrase, "
        "sisipkan inline citation [1], [2], … di akhir setiap poin. "
        "Di akhir, tuliskan daftar referensi.\n"
    )
    context_lines = []
    for i, hit in enumerate(hits, start=1):
        parts = [hit['book']]
        if hit['chapters'] != '–': parts.append(hit['chapters'])
        if hit['sections'] != '–': parts.append(hit['sections'])
        parts.append(f"Halaman {hit['pages']}")
        meta_str = ", ".join(parts)
        context_lines.append(f"[{i}] {hit['chunk']} — {meta_str}")
    context = "\n".join(context_lines)
    prompt = (
        f"{system}Pertanyaan: {query}\nKonteks:\n{context}\nJawaban akhir:"
    )
    with open('last_prompt.txt','w',encoding='utf-8') as f:
        f.write(prompt)
    return prompt


def post_process(raw: str, hits: list) -> str:
    text = raw.strip()
    # Tentukan jumlah referensi unik yang akan ditampilkan
    unique_keys = []
    for hit in hits:
        m = hit['metadata']
        key = (hit['book'], hit['chapters'], hit['sections'], hit['pages'])
        if key not in unique_keys:
            unique_keys.append(key)
    n_refs = len(unique_keys)

    # Jika model tidak menyisipkan citation, tambahkan hanya untuk kalimat hingga jumlah referensi
    if not re.search(r"\[\d+\]", text):
        sentences = re.split(r'(?<=[.!?])\s+', text)
        new_sents = []
        for idx, s in enumerate(sentences, start=1):
            s = s.strip()
            if idx <= n_refs and not re.search(r"\[\d+\]$", s):
                new_sents.append(f"{s} [{idx}]")
            else:
                new_sents.append(s)
        text = ' '.join(new_sents)

    # Normalisasi nomor citation sesuai urutan unik
    nums = [int(n) for n in re.findall(r"\[(\d+)\]", text)]
    def repl(m):
        num = int(m.group(1))
        # Map original num to its index in citation order (1..n_refs)
        return f"[{nums.index(num)+1}]"
    text = re.sub(r"\[(\d+)\]", repl, text)

    # Bangun daftar referensi
    refs = []
    for idx, key in enumerate(unique_keys, start=1):
        book, chap, sec, page = key
        parts = [book]
        if chap != '–': parts.append(f"Bab: {chap}")
        if sec != '–': parts.append(f"Subbab: {sec}")
        parts.append(f"Halaman {page}")
        refs.append(f"{idx}. {', '.join(parts)}")

    return f"{text}\n\nReferensi:\n" + "\n".join(refs)


def generate_answer(query: str, top_k: int = 5) -> str:
    hits = retrieve_and_rerank(query, coarse_k=top_k*4, final_k=top_k)
    if not hits:
        return "Tidak ditemukan konteks."
    prompt = build_prompt(query, hits)
    proc = subprocess.run(
        ['ollama', 'run', 'registry.ollama.ai/library/deepseek-r1:7b'],
        input=prompt,
        text=True,
        encoding='utf-8',
        errors='replace',
        capture_output=True
    )
    if proc.returncode != 0:
        return f"Model error: {proc.stderr.strip()}"
    return post_process(proc.stdout, hits)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--query',required=True)
    parser.add_argument('-k','--top_k',type=int,default=5)
    args = parser.parse_args()
    print(generate_answer(args.query, top_k=args.top_k))

if __name__=='__main__':
    main()
