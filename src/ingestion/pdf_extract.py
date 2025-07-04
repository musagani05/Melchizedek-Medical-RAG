import pdfplumber
import os
import re
import spacy
from PyPDF2 import PdfFileReader

# Memuat model spaCy untuk tokenisasi
nlp = spacy.load("en_core_web_sm")

def extract_text(pdf_path):
    text = ''
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text_page = page.extract_text()
            if text_page:
                text += text_page + '\n'
    return text

def extract_metadata(pdf_path):
    """
    Ekstraksi metadata (seperti judul, pengarang, dan halaman) dari PDF.
    """
    with open(pdf_path, 'rb') as file:
        reader = PdfFileReader(file)
        metadata = reader.getDocumentInfo()
        title = metadata.title if metadata.title else "Unknown Title"
        author = metadata.author if metadata.author else "Unknown Author"
    return title, author

def clean_text(text):
    """
    Pembersihan teks: Menghapus newline, spasi berlebih, dan karakter non-alfabet.
    """
    text = re.sub(r'\n', ' ', text)  # Mengganti newline dengan spasi
    text = re.sub(r'\s+', ' ', text)  # Menghapus spasi berlebih
    text = re.sub(r'[^\w\s]', '', text)  # Menghapus tanda baca
    return text

def chunk_text(text, max_tokens=1024):
    """
    Tokenisasi dan chunking teks menjadi unit yang lebih kecil.
    """
    doc = nlp(text)
    tokens = [token.text for token in doc]
    chunks = []
    chunk = []
    token_count = 0

    for token in tokens:
        token_count += len(token)
        if token_count <= max_tokens:
            chunk.append(token)
        else:
            chunks.append(' '.join(chunk))
            chunk = [token]
            token_count = len(token)
    if chunk:
        chunks.append(' '.join(chunk))

    return chunks

if __name__ == '__main__':
    # Menetapkan path absolut untuk folder articles dan pdf_texts
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))  # Navigate to root project
    articles_folder = os.path.join(base_path, 'data', 'articles')
    pdf_texts_folder = os.path.join(base_path, 'data', 'pdf_texts')

    # Membuat folder untuk hasil ekstraksi teks jika belum ada
    os.makedirs(pdf_texts_folder, exist_ok=True)

    # Proses ekstraksi, pembersihan, tokenisasi, chunking untuk semua file PDF di folder articles
    for filename in os.listdir(articles_folder):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(articles_folder, filename)
            print(f'Extracting text from {pdf_path}')
            text = extract_text(pdf_path)
            # Ekstraksi metadata
            title, author = extract_metadata(pdf_path)
            # Pembersihan teks
            clean_data = clean_text(text)
            # Tokenisasi dan chunking
            text_chunks = chunk_text(clean_data, max_tokens=1024)
            
            # Simpan hasil chunking dan metadata sebagai file .txt di pdf_texts
            output_path = os.path.join(pdf_texts_folder, filename[:-4] + '_chunked.txt')
            with open(output_path, 'w', encoding='utf-8') as f:
                for chunk in text_chunks:
                    # Menulis chunk dan metadata
                    f.write(f"Title: {title}\nAuthor: {author}\n")
                    f.write(f"Chunk: {chunk}\n")
                    f.write("\n" + "-"*50 + "\n")
            print(f'Saved extracted, chunked text with metadata to {output_path}')