# If needed, run:
#   pip install sounddevice scipy torchaudio torch transformers soundfile

import sounddevice as sd
import numpy as np
import time
import torch
import torchaudio
from scipy.io.wavfile import write
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import os


THRESHOLD = 0.0005           # Lower threshold to ensure normal speech is detected
SILENCE_DURATION = 5.0       # Stop recording if we have this many seconds of silence after talking
SAMPLE_RATE = 16000
CHANNELS = 1                 # Use 1 channel (mono)
OUTPUT_DIR = "./recordings"
DURATION_LIMIT = 60          # Hard stop after 60 seconds

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the Hugging Face Whisper-based audio classification model
model_name = "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3"
extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModelForAudioClassification.from_pretrained(model_name)

# Track the last predicted emotion (if you want to use it somewhere else)
last_emotion = None

def rms(data: np.ndarray) -> float:
    """Compute the RMS volume of the incoming audio chunk."""
    return np.sqrt(np.mean(np.square(data)))

def record_until_silence() -> np.ndarray:
    """
    Continuously records from the mic until:
      (a) We detect THRESHOLD crossing (start), then
      (b) We detect SILENCE_DURATION of silence (stop), or
      (c) We hit DURATION_LIMIT.

    Returns the recorded audio as a NumPy float32 array.
    """
    print("\nListening for speech...")
    recording_chunks = []
    started = False
    silence_start = None
    start_time = time.time()

    def audio_callback(indata, frames, time_info, status):
        nonlocal recording_chunks, started, silence_start, start_time

        if status:
            print(f"[DEBUG] Sounddevice status: {status}")

        # Compute volume of current chunk
        volume = rms(indata)
        print(f"[DEBUG] Callback volume={volume:.6f}, threshold={THRESHOLD}, indata.shape={indata.shape}")

        # Check if above threshold => speaking
        if volume > THRESHOLD:
            if not started:
                print("Speech detected. Recording...")
                started = True
            silence_start = None
        else:
            # We're below threshold
            if started:
                # If we've started before, check for silence
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start > SILENCE_DURATION:
                    print("Silence detected. Stopping recording.")
                    raise sd.CallbackStop()

        # If we have started, append the current chunk
        if started:
            recording_chunks.append(indata.copy())

        # Safety check: stop if we exceed max duration
        if (time.time() - start_time) > DURATION_LIMIT:
            print("Max duration reached. Stopping recording.")
            raise sd.CallbackStop()

    # Open an input stream
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        callback=audio_callback
    ):
        try:
            # Let the callback run until it raises CallbackStop
            sd.sleep(int(DURATION_LIMIT * 1000))
        except sd.CallbackStop:
            pass

    # If we never recorded anything
    if not recording_chunks:
        print("[DEBUG] No audio data captured.")
        return np.array([], dtype=np.float32)

    # Concatenate all chunks
    audio_data = np.concatenate(recording_chunks, axis=0)
    print(f"[DEBUG] Final audio shape from record_until_silence: {audio_data.shape}")
    return audio_data

def save_audio(audio: np.ndarray, filename: str):
    """Save float32 audio in [-1,1] range to a 16-bit WAV."""
    if audio.size == 0:
        print("[DEBUG] Audio is empty, skipping save.")
        return

    # Convert float32 samples to int16
    scaled = (audio * 32767).astype(np.int16)
    write(filename, SAMPLE_RATE, scaled)
    print(f"Saved: {filename}")

def classify_emotion(filepath, extractor, model):
    """
    Classify emotion using a Hugging Face Whisper-based model.
    Returns None if the audio is too short or has invalid shape.
    """

    waveform, sr = torchaudio.load(filepath)
    print(f"[DEBUG] Loaded '{filepath}'. Initial shape={waveform.shape}, sr={sr}")

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        print(f"[DEBUG] Downmixed shape={waveform.shape}")

    if waveform.shape[-1] == 0:
        print("[DEBUG] Waveform has 0 samples. Skipping classification.")
        return None

    if waveform.shape[-1] < 1000:
        print(f"[DEBUG] Waveform too short ({waveform.shape[-1]} samples). Skipping.")
        return None

    if sr != extractor.sampling_rate:
        resampler = torchaudio.transforms.Resample(sr, extractor.sampling_rate)
        waveform = resampler(waveform)
        print(f"[DEBUG] After resampling, shape={waveform.shape}")

    print(f"[DEBUG] Final waveform shape before extractor: {waveform.shape}")
    if waveform.shape[0] != 1:
        print("[DEBUG] Waveform is not mono after downmix (shape[0] != 1). Skipping.")
        return None
    if waveform.shape[-1] == 0:
        print("[DEBUG] Waveform time dimension is 0. Skipping.")
        return None

    try:
        inputs = extractor(
            waveform,
            sampling_rate=extractor.sampling_rate,
            return_tensors="pt"
        )
    except ValueError as e:
        # Catch any final shape/axes errors from the extractor
        print(f"[DEBUG] Extractor error: {e}")
        return None

    with torch.no_grad():
        logits = model(**inputs).logits

    pred_id = torch.argmax(logits, dim=-1).item()
    emotion = model.config.id2label[pred_id]
    return emotion

######################################################
# MAIN LOOP
######################################################
if __name__ == "__main__":
    counter = 1
    while True:
        audio = record_until_silence()
        print(f"[DEBUG] record_until_silence() returned shape={audio.shape}")

        if audio.size == 0:
            print("No audio recorded. Continue listening...\n")
            continue

        filename = os.path.join(OUTPUT_DIR, f"recording_{counter}.wav")
        save_audio(audio, filename)
        predicted_emotion = classify_emotion(filename, extractor, model)

        if predicted_emotion is not None:
            print("Predicted emotion:", predicted_emotion)
        else:
            print("No emotion predicted.")

        counter += 1