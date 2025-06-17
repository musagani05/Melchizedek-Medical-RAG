import os
import sys

# ambil path ke folder root (satu tingkat di atas webapp/)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import gradio as gr
from src.retriever.retrieve import retrieve_context

def answer_fn(question: str, top_k: int):
    # panggil fungsi retrieve yang sudah ada
    results = retrieve_context(question, top_k=top_k)
    docs = results["documents"][0]
    metas= results["metadatas"][0]
    # format output
    output = []
    for i, (doc, meta) in enumerate(zip(docs, metas), 1):
        src = meta.get("source", "–")
        output.append(f"### Hasil {i}  \n**Sumber:** {src}  \n{doc}")
    return "\n\n---\n\n".join(output)

iface = gr.Interface(
    fn=answer_fn,
    inputs=[
        gr.components.Textbox(lines=2, placeholder="Tanya sesuatu tentang medis...", label="Question"),
        gr.components.Slider(minimum=1, maximum=10, step=1, value=3, label="Top K")
    ],
    outputs=gr.components.Markdown(label="Results"),
    title="🩺 RAG-MEDICAL QA",
    description="Tanya apa saja tentang materi medis dari koleksi lokal Anda. Powered by ChromaDB & Gradio."
)

if __name__ == "__main__":
    iface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True    # nanti di HF Spaces, parameter ini diabaikan
    )