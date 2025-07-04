"""
Validasi otomatis ekstraksi: ringkasan & sampel section.
"""
import os, json
import pandas as pd

JSON_DIR = "data/pdf_texts_json"
SAMPLE_PER_FILE = 2

def summarize_file(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    n_sec = len(data["sections"])
    samples = data["sections"][:SAMPLE_PER_FILE]
    return {
        "file": data["filename"],
        "pages": data["num_pages"],
        "n_sections": n_sec,
        "sample_titles": [sec["title"] for sec in samples]
    }

def main():
    summaries = []
    for fn in os.listdir(JSON_DIR):
        if fn.endswith(".json"):
            summaries.append(summarize_file(os.path.join(JSON_DIR, fn)))
    df = pd.DataFrame(summaries)
    print("\n=== Ekstraksi Summary ===")
    print(df.to_markdown(index=False))
    # Tampilkan contoh section
    print("\n=== Contoh Konten (1st file) ===")
    first = summaries[0]["file"]
    data = json.load(open(os.path.join(JSON_DIR, first + ".json"), encoding="utf-8"))
    for sec in data["sections"][:SAMPLE_PER_FILE]:
        print(f"\n-- {sec['title']} --\n{sec['content'][:200]}...\n")

if __name__ == "__main__":
    main()