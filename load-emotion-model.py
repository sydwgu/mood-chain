# make sure to run 'pip install tensorflow torch transformers'
# before running the following in the terminal: 'python3 load-emotion-model.py'
# (this is assuming that the user already has pip and python3 installed on their machine)

# Load model directly from HuggingFace
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

extractor = AutoFeatureExtractor.from_pretrained("firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3")
model = AutoModelForAudioClassification.from_pretrained("firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3")
