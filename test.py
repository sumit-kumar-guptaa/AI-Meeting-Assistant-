from dotenv import load_dotenv
from utils.audio_processor import process_input
from core.transcriber import transcribe_all

load_dotenv()

source = "https://www.youtube.com/watch?v=vb2gfz_isU8"
language = "hinglish"

chunks = process_input(source)
transcript = transcribe_all(chunks, language=language)

print("\n=========TRANSCRIPTION===========\n")
print(transcript)