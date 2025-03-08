import os
import numpy as np
import random
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

# Define dataset paths
input_path = 'D:\\dataset\\asl_alphabet_train\\asl_alphabet_train'

# Image size and batch size
IMG_SIZE = (224, 224)  # Resize images to standard size
BATCH_SIZE = 32

# Create ImageDataGenerator with real-time augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values
    rotation_range=30,  # Randomly rotate images by ±30 degrees
    width_shift_range=0.1,  # Shift images horizontally
    height_shift_range=0.1,  # Shift images vertically
    horizontal_flip=True,  # Flip images horizontally
    brightness_range=[0.8, 1.2],  # Random brightness adjustments
    zoom_range=0.2,  # Random zoom in/out
    validation_split=0.1  # Reserve 10% of images for validation
)

# Load dataset dynamically using ImageDataGenerator
train_generator = train_datagen.flow_from_directory(
    input_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    input_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Get the number of classes dynamically
num_classes = train_generator.num_classes
print(f"Number of classes detected: {num_classes}")

# Define CNN Model (Enhanced Version)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')  # Output layer with dynamic class count
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model using ImageDataGenerator
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=15
)

# Evaluate model on validation set
test_loss, test_acc = model.evaluate(val_generator)
print(f"Validation Accuracy: {test_acc:.4f}")
