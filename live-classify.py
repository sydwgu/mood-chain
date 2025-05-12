# if needed, run the following in terminal to install required packages: pip install sounddevice scipy torchaudio torch transformers

import sounddevice as sd
import numpy as np
import time
import torch
import torchaudio
from scipy.io.wavfile import write
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

# global variable to track previous speaker's predicited emotion
last_emotion = None

# open source emotion processing model from HuggingFace
model_name = "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3"
extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModelForAudioClassification.from_pretrained(model_name)

# configure recording settings
THRESHOLD = 0.02         # adjust noise threshold based on environment
SILENCE_DURATION = 5.0   # seconds
SAMPLE_RATE = 16000      # hz
CHANNELS = 1
OUTPUT_DIR = "./recordings"
DURATION_LIMIT = 60      # max recording length

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

def rms(data):
    return np.sqrt(np.mean(np.square(data)))

# continuously record and save live audio clips
def record_until_silence():
    print("Listening for speech...")
    recording = []
    started = False
    silence_start = None
    start_time = time.time()

    def callback(indata, frames, time_info, status):
        nonlocal recording, started, silence_start, start_time

        volume = rms(indata)
        if volume > THRESHOLD:
            if not started:
                print("Speech detected, recording...")
                started = True
            silence_start = None
        elif started:
            if silence_start is None:
                silence_start = time.time()
            elif time.time() - silence_start > SILENCE_DURATION:
                print("Silence detected, stopping recording.")
                raise sd.CallbackStop()

        if started:
            recording.append(indata.copy())

        if time.time() - start_time > DURATION_LIMIT:
            print("Max duration reached, stopping.")
            raise sd.CallbackStop()

    with sd.InputStream(callback=callback, samplerate=SAMPLE_RATE, channels=CHANNELS):
        try:
            sd.sleep(int(DURATION_LIMIT * 1000))
        except sd.CallbackStop:
            pass

    audio = np.concatenate(recording, axis=0)
    return audio

# save recorded audio filie to machine
def save_audio(audio, filename):
    write(filename, SAMPLE_RATE, (audio * 32767).astype(np.int16))

# run HuggingFace emotion classification model
def classify_emotion(filepath, extractor, model):
    waveform, sr = torchaudio.load(filepath)
    if sr != extractor.sampling_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=extractor.sampling_rate)
        waveform = resampler(waveform)

    inputs = extractor(waveform, sampling_rate=extractor.sampling_rate, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits

    pred_id = torch.argmax(logits, dim=-1).item()
    return model.config.id2label[pred_id]


# live processing loop
counter = 1
while True:
    audio = record_until_silence()
    filename = os.path.join(OUTPUT_DIR, f"recording_{counter}.wav")
    save_audio(audio, filename)
    print(f"Saved: {filename}")

    last_emotion = classify_emotion(filename, extractor, model)
    print(f"Predicted emotion: {last_emotion}")

    counter += 1
