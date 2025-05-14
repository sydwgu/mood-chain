from shared_state import shared_data
import time

while True:
    print(f"[emotion_consumer] Detected: {shared_data['last_emotion']}")
    time.sleep(2)
