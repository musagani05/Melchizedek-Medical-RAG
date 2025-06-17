import os
import gradio as gr

# 1) Import fungsi builder
from src.indexer.build_index_multimodal import build_multimodal_index
from src.retriever.retrieve import retrieve_context

# 2) Jika folder ./chroma_db belum ada atau kosong, build index dulu
if not os.path.isdir("./chroma_db") or not os.listdir("./chroma_db"):
    print("[BOOT] chroma_db missing/empty → building index …")
    build_multimodal_index()

# 3) Handler Gradio
def answer_fn(question: str, top_k: int):
    hits = retrieve_context(question, top_k=top_k)
    if not hits:
        return "⚠️ Maaf, tidak ditemukan konteks untuk pertanyaan ini."

    formatted = []
    for i, hit in enumerate(hits, 1):
        chunk = hit["chunk"]
        src   = hit.get("source", "–")
        formatted.append(f"### Hasil {i}\n**Sumber:** {src}\n\n{chunk}")
    return "\n\n---\n\n".join(formatted)

# 4) Bangun interface
iface = gr.Interface(
    fn=answer_fn,
    inputs=[
        gr.Textbox(lines=2, label="Question", placeholder="Tanya apa saja…"),
        gr.Slider(minimum=1, maximum=10, step=1, value=3, label="Top K")
    ],
    outputs=gr.Markdown(label="Results"),
    title="🩺 RAG-MEDICAL QA",
    description="Tanya apa saja tentang materi medis dari koleksi lokal Anda."
)

if __name__ == "__main__":
    iface.launch()