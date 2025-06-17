import pdfplumber
import os

def extract_text(pdf_path):
    text = ''
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text_page = page.extract_text()
            if text_page:
                text += text_page + '\n'
    return text

if __name__ == '__main__':
    # Membuat folder untuk hasil ekstraksi teks jika belum ada
    os.makedirs('data/pdf_texts', exist_ok=True)

    # Proses ekstraksi untuk semua file PDF di slides dan articles
    for folder in ['data/slides', 'data/articles']:
        for filename in os.listdir(folder):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(folder, filename)
                print(f'Extracting text from {pdf_path}')
                text = extract_text(pdf_path)
                # Simpan hasil ekstraksi sebagai file .txt di pdf_texts
                output_path = os.path.join('data/pdf_texts', filename[:-4] + '.txt')
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                print(f'Saved extracted text to {output_path}')