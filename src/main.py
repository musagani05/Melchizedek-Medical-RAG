# main.py
from retriever.retrieve import retrieve_context
from llm.ollama_client import generate_answer

def chat_loop():
    print("=== RAG Medical Assistant ===")
    while True:
        q = input("\nPertanyaan (ketik 'exit' untuk keluar): ")
        if q.lower() in ("exit", "quit"):
            break

        # 1. Ambil konteks top-5
        hits = retrieve_context(q, top_k=5)
        # 2. Panggil LLM untuk jawab
        answer = generate_answer(q, hits)
        print("\n--- Jawaban ---")
        print(answer)

if __name__ == "__main__":
    chat_loop()