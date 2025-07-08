import os
import gradio as gr
import sys
# Pastikan folder src ada dalam path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

# Panggil generate_answer dari modul retrieve
from archive.retriever import generate_answer

# On‑the‑fly index build jika folder kosong
def ensure_index():
    if not os.path.isdir("chroma_db") or not os.listdir("chroma_db"):
        from archive.multimodal_indexer import build_multimodal_index
        build_multimodal_index(
            pdf_folder="data/articles",
            audio_folder="data/audio_texts",
            image_folder="data/images",
            chroma_path="chroma_db",
            collection_name="rag_medical"
        )

ensure_index()

# Parameter tetap Top K untuk retrieval + generation
TOP_K = 5

# Definisi UI Gradio
demo = gr.Blocks()
with demo:
    with gr.Row():
        gr.Markdown("## 🩺 Melchizedek Chat Bot")
        flag_btn = gr.Button("🚩", elem_id="flag-btn")

    chatbot = gr.Chatbot(elem_id="chatbot-panel")

    with gr.Row():
        msg = gr.Textbox(
            placeholder="Type a message...",
            show_label=False,
            lines=1,
            elem_id="message-box"
        )
        send_btn = gr.Button("Send", elem_id="send-btn")

    def respond(message, history):
        if not message:
            return "", history
        try:
            bot_msg = generate_answer(message, top_k=TOP_K)
        except Exception as e:
            bot_msg = f"Error: {e}"
        history = history + [(message, bot_msg)]
        return "", history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    send_btn.click(respond, [msg, chatbot], [msg, chatbot])

    def flag_conversation(history):
        with open("flags.log", "a", encoding="utf-8") as f:
            f.write("FLAGGED CONVERSATION 🚩\n")
            for user_msg, bot_msg in history:
                f.write(f"Q: {user_msg}\nA: {bot_msg}\n\n")
        return None
    flag_btn.click(flag_conversation, inputs=[chatbot], outputs=[])

if __name__ == "__main__":
    demo.launch(inbrowser=False)
