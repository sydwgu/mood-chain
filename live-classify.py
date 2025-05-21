# If needed, run:
#   pip install sounddevice scipy torchaudio torch transformers soundfile librosa

import sounddevice as sd
import numpy as np
import time
import torch
import torchaudio
from scipy.io.wavfile import write
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import os
import sys
import librosa
import json

THRESHOLD = 0.09
SILENCE_DURATION = 2.0
SAMPLE_RATE = 16000
CHANNELS = 1
OUTPUT_DIR = "./recordings"
DURATION_LIMIT = 30

os.makedirs(OUTPUT_DIR, exist_ok=True)


# Load the Hugging Face Whisper-based audio classification model
model_name = "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3"
extractor = AutoFeatureExtractor.from_pretrained(model_name)  # do_normalize=True is default in that model config
model = AutoModelForAudioClassification.from_pretrained(model_name)

last_emotion = None

def rms(data: np.ndarray) -> float:
    """Compute the RMS volume of the incoming audio chunk."""
    return np.sqrt(np.mean(np.square(data)))

def record_until_silence() -> np.ndarray:
    """
    Continuously records from the mic until:
      (a) volume crosses THRESHOLD (start),
      (b) we get SILENCE_DURATION of silence (stop),
      (c) or we hit DURATION_LIMIT.
    Returns the recorded audio as a float32 NumPy array.
    """
    print("\nListening for speech...")
    recording_chunks = []
    started = False
    silence_start = None
    start_time = time.time()

    def audio_callback(indata, frames, time_info, status):
        nonlocal started, silence_start
        volume = rms(indata)
        print(f"volume={volume:.4f}, threshold={THRESHOLD:.4f}, started={started}")

        if volume > THRESHOLD:
            if not started:
                started = True
                print("Speech detected => started recording.")
            silence_start = None
        else:
            if started:
                if silence_start is None:
                    silence_start = time.time()
                    print("Silence started now.")
                else:
                    elapsed_silence = time.time() - silence_start
                    print(f"Silence for {elapsed_silence:.2f}s (need {SILENCE_DURATION}s).")
                    if elapsed_silence >= SILENCE_DURATION:
                        print("Enough silence => stopping.")
                        raise sd.CallbackStop()

        if started:
            recording_chunks.append(indata.copy())

        # Overall duration limit
        if (time.time() - start_time) > DURATION_LIMIT:
            print("Max duration reached => stopping.")
            raise sd.CallbackStop()

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=audio_callback):
        try:
            sd.sleep(int(DURATION_LIMIT * 1000))
        except sd.CallbackStop:
            pass

    if not recording_chunks:
        print("[DEBUG] No audio data captured.")
        return np.array([], dtype=np.float32)

    audio_data = np.concatenate(recording_chunks, axis=0)
    print(f"[DEBUG] Final audio shape: {audio_data.shape}")
    return audio_data

def save_audio(audio: np.ndarray, filename: str):
    """Save float32 audio in [-1,1] to a 16-bit WAV."""
    if audio.size == 0:
        print("[DEBUG] Audio is empty, skipping save.")
        return
    scaled = (audio * 32767).astype(np.int16)
    write(filename, SAMPLE_RATE, scaled)
    print(f"Saved: {filename}")


def string_to_rgb(indexed_list):
    """
    Takes a list of string values and converts each to an assigned RGB value.
    Returns a list of RGB tuples.
    """
    # Define your custom mapping
    color_map = {
        "sad": (0, 0, 255),          # Blue
        "happy": (255, 255, 0),      # Yellow
        "angry": (255, 0, 0),        # Red
        "neutral": (255, 255, 255),  # White
        "disgust": (0, 128, 0),      # Green
        "fearful": (128, 0, 128),    # Purple
        "surprised": (255, 165, 0),  # Orange
        "calm": (64, 224, 208)       # Turquoise
    }

    rgb_values = []
    for item in indexed_list:
        rgb = color_map.get(item.lower(), (0, 0, 0))  # Default to black if not found
        rgb_values.append(rgb)

    return rgb_values


def classify_emotion(filepath, extractor, model, max_duration=30.0):
    """
    Classify emotion using the same approach as your 'working' snippet:
    1) Load with librosa at extractor.sampling_rate
    2) Pad or truncate to max_duration
    3) Use extractor(...) with truncation=True
    4) Forward pass through the model
    """

    # 1. Load with librosa at the model's sampling rate
    audio_array, sr = librosa.load(filepath, sr=extractor.sampling_rate)

    # 2. Pad or truncate to max_duration
    max_length = int(extractor.sampling_rate * max_duration)
    if len(audio_array) > max_length:
        audio_array = audio_array[:max_length]
    else:
        # pad at the end with zeros
        audio_array = np.pad(audio_array, (0, max_length - len(audio_array)))

    # 3. Prepare inputs for the model
    #    (The 'working' code used feature_extractor(..., max_length=..., truncation=True))
    inputs = extractor(
        audio_array,
        sampling_rate=extractor.sampling_rate,
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )

    # 4. Move to CPU or GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 5. Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    pred_id = torch.argmax(logits, dim=-1).item()
    emotion = model.config.id2label[pred_id]
    return emotion


# define json that live updates as emotions are detected
file_path = os.path.join(os.path.dirname(__file__), "emotion_colors.json")

######################################################
# MAIN LOOP
######################################################
if __name__ == "__main__":
    emotions = []
    counter = 1

    try:
        while True:
            audio = record_until_silence()
            if audio.size == 0:
                print("No audio recorded. Continue listening...\n")
                continue

            filename = os.path.join(OUTPUT_DIR, f"recording_{counter}.wav")
            save_audio(audio, filename)

            predicted_emotion = classify_emotion(filename, extractor, model)
            if predicted_emotion is not None:
                print("Predicted emotion:", predicted_emotion)
                emotions.append(predicted_emotion)

                # convert emotion to associated rgb value
                emotion_rgb = string_to_rgb(predicted_emotion)

                # print out to updating json
                color = {
                    "r": emotion_rgb[0],
                    "g": emotion_rgb[1],
                    "b": emotion_rgb[2]
                }

                with open(file_path, "w") as f:
                    json.dump(color, f)
                
            else:
                print("No emotion predicted.")

            counter += 1

    except KeyboardInterrupt:
        print("\nExiting gracefully...")
    finally:
        print("All detected emotions collected during this session:")
        print(emotions)
        sys.exit(0)