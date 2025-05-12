# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("audio-classification", model="firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3")

# Load model directly
from transformers import AutoProcessor, AutoModelForAudioClassification

processor = AutoProcessor.from_pretrained("firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3")
model = AutoModelForAudioClassification.from_pretrained("firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3")