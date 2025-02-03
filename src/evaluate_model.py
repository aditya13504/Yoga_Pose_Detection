import tensorflow as tf
from tensorflow.keras import metrics
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# Constants
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 5  # Adjust to your dataset (5 classes)

# Load the saved model
model = tf.keras.models.load_model('models/yoga_pose_model.keras')

def decode_image(image, label):
    """Same as in train_model.py"""
    try:
        image = tf.image.resize(image, IMAGE_SIZE)
        image = tf.cast(image, tf.float32) / 255.0
        return image, label
    except:
        return tf.zeros(IMAGE_SIZE + (3,)), label

def filter_corrupted_images(image, label):
    return tf.reduce_sum(tf.abs(image)) > 0.0

# Load test dataset
test_ds = tf.keras.utils.image_dataset_from_directory(
    r"dataset\TEST", #enter the path of TRAIN dataset here
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
).map(decode_image)\
 .filter(filter_corrupted_images)\
 .prefetch(buffer_size=tf.data.AUTOTUNE)

# Evaluate accuracy
loss, accuracy = model.evaluate(test_ds)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

# Generate predictions
y_true = np.concatenate([y for _, y in test_ds], axis=0)
y_pred = model.predict(test_ds)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_true, axis=1)

# Confusion matrix and classification report
print("\nConfusion Matrix:")
print(confusion_matrix(y_true_classes, y_pred_classes))

print("\nClassification Report:")
print(classification_report(y_true_classes, y_pred_classes, 
                            target_names=['downdog', 'goddess', 'plank', 'tree', 'warrior2']))