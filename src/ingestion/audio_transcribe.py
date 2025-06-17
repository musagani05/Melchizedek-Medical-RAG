import whisper
import os

def transcribe_audio(audio_path, model_name='base'):
    model = whisper.load_model(model_name)
    try:
        result = model.transcribe(audio_path)
        if result is None or 'text' not in result:
            print(f"Warning: no text transcribed from {audio_path}")
            return ""
        return result['text']
    except Exception as e:
        print(f"Error transcribing {audio_path}: {e}")
        return ""

if __name__ == "__main__":
    # Membuat folder hasil transkripsi teks audio
    os.makedirs('data/audio_texts', exist_ok=True)

    # Proses transkripsi semua file audio di folder audio
    for filename in os.listdir('data/audio'):
        if filename.lower().endswith(('.mp3', '.wav', '.m4a')):
            audio_path = os.path.join('data/audio', filename)
            print(f'Transcribing audio {audio_path}')
            text = transcribe_audio(audio_path)
            output_path = os.path.join('data/audio_texts', filename.rsplit('.', 1)[0] + '.txt')
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f'Saved transcription to {output_path}')