import os
import subprocess
from dotenv import load_dotenv

# 1) Load .env
env_path = os.path.join(os.path.dirname(__file__), "../../config/.env")
load_dotenv(dotenv_path=env_path)

# 2) Ambil model dari env
MODEL = os.getenv("OLLAMA_MODEL")
if not MODEL:
    raise ValueError("OLLAMA_MODEL belum diset di .env")

# 3) Prompt uji coba
PROMPT = "Apa itu sciatica?"

# 4) Bangun perintah CLI ollama (GPU akan auto-terdeteksi)
cmd = [
    "ollama", "run", MODEL,
    PROMPT
]

# 5) Jalankan dan tangani output/error
proc = subprocess.run(cmd, capture_output=True, text=True)
if proc.returncode != 0:
    print("❌ Error saat memanggil ollama:\n", proc.stderr)
else:
    print("✅ Response dari ollama:\n", proc.stdout)