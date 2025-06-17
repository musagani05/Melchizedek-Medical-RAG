import subprocess

MODEL_NAME = "registry.ollama.ai/library/deepseek-r1:7b"

def generate_answer(question: str, contexts: list):
    ctx_text = "\n\n".join(f"[{h['source']}] {h['chunk']}" for h in contexts)
    prompt = (
        "You are a helpful medical assistant. Use ONLY the following excerpts to answer the question.\n\n"
        f"{ctx_text}\n\n"
        f"Question: {question}\n"
        "Answer and cite source filenames where relevant."
    )

    cmd = ["ollama", "run", MODEL_NAME]
    print(f"[DEBUG] Running command: {cmd}")

    # ← paksa decode pakai utf-8, ganti karakter yang error
    result = subprocess.run(
        cmd,
        input=prompt,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace'
    )

    # guard None dan strip
    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()

    print("[DEBUG ollama] returncode:", result.returncode)
    print("[DEBUG ollama] stderr:", stderr)
    print("[DEBUG ollama] stdout:", stdout[:200].replace("\n"," "))

    return stdout