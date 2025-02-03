import cv2
import numpy as np
import tensorflow as tf

# Constants
IMAGE_SIZE = (224, 224)
CLASS_LABELS = ['downdog', 'goddess', 'plank', 'tree', 'warrior2']

# Load the trained model
model = tf.keras.models.load_model('models/yoga_pose_model.keras')

def preprocess_frame(frame):
    """Preprocess webcam frame to match training data"""
    resized = cv2.resize(frame, IMAGE_SIZE)
    normalized = resized.astype(np.float32) / 255.0
    return np.expand_dims(normalized, axis=0)

# Initialize webcam
cap = cv2.VideoCapture(0)
cv2.namedWindow('Yoga Pose Detection', cv2.WINDOW_NORMAL)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess and predict
    processed_frame = preprocess_frame(frame)
    predictions = model.predict(processed_frame, verbose=0)
    predicted_class = np.argmax(predictions[0])
    pose_name = CLASS_LABELS[predicted_class]
    confidence = np.max(predictions[0]) * 100

    # Display result
    cv2.putText(frame, f"{pose_name} ({confidence:.1f}%)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow('Yoga Pose Detection', frame)

    # Check for exit conditions
    key = cv2.waitKey(1)
    if key == ord('q') or cv2.getWindowProperty('Yoga Pose Detection', cv2.WND_PROP_VISIBLE) < 1:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
# Add a small delay to ensure windows close properly
cv2.waitKey(1)