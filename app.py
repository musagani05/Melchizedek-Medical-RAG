import gradio as gr
from src.retriever.retrieve import retrieve_context

def answer_fn(question: str, top_k: int):
    # 1) Panggil retriever, terima list of hits
    hits = retrieve_context(question, top_k=top_k)

    # 2) Tangani kasus kosong
    if not hits:
        return "⚠️ Maaf, tidak ditemukan konteks untuk pertanyaan ini."

    # 3) Format output
    formatted = []
    for i, hit in enumerate(hits, start=1):
        chunk = hit["chunk"]
        src   = hit.get("source", "–")
        formatted.append(
            f"### Hasil {i}\n**Sumber:** {src}\n\n{chunk}"
        )

    # 4) Gabung dengan pemisah
    return "\n\n---\n\n".join(formatted)

# Bangun interface Gradio
iface = gr.Interface(
    fn=answer_fn,
    inputs=[
        gr.Textbox(lines=2, placeholder="Tanya sesuatu tentang medis...", label="Question"),
        gr.Slider(minimum=1, maximum=10, step=1, value=3, label="Top K")
    ],
    outputs=gr.Markdown(label="Results"),
    title="🩺 RAG-MEDICAL QA",
    description="Tanya apa saja tentang materi medis dari koleksi lokal Anda."
)

if __name__ == "__main__":
    iface.launch()