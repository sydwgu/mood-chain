from shared_state import shared_data
import time

# Simulate live emotion updates
emotions = ['angry', 'happy', 'sad', 'neutral']
i = 0

while True:
    # Simulate getting a new emotion label
    shared_data['last_emotion'] = emotions[i % len(emotions)]
    print(f"[emotion_detector] Updated to: {shared_data['last_emotion']}")
    i += 1
    time.sleep(5)
