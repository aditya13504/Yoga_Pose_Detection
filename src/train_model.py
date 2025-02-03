import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint
import os

# Constants
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 5  # downdog, goddess, plank, tree, warrior2

def decode_image(image, label):
    """Decode and preprocess image with error handling"""
    try:
        image = tf.image.resize(image, IMAGE_SIZE)
        image = tf.cast(image, tf.float32) / 255.0
        return image, label
    except:
        return tf.zeros(IMAGE_SIZE + (3,)), label

def filter_corrupted_images(image, label):
    return tf.reduce_sum(tf.abs(image)) > 0.0

# Load datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    r"dataset\TRAIN", #enter the path of TRAIN dataset here
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
).map(decode_image).filter(filter_corrupted_images).prefetch(tf.data.AUTOTUNE)

val_ds = tf.keras.utils.image_dataset_from_directory(
    r"dataset\TEST",   #enter the path of TEST dataset here
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
).map(decode_image).filter(filter_corrupted_images).prefetch(tf.data.AUTOTUNE)


def create_model():
    model = models.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    return model

model = create_model()

checkpoint = ModelCheckpoint(
    'models/best_model.keras', 
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[checkpoint]
)

model.save('models/yoga_pose_model.keras')
print("Model saved to models/yoga_pose_model.keras")