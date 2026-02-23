# AI-powred-ultrasonic-scanner-for-pregenent-women
print("CNN script started")

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# FINAL DATASET PATH (DO NOT CHANGE)
TRAIN_DIR = r"C:\CSP\project\Ultrasound_Dataset\Data\Data\train"
TEST_DIR  = r"C:\CSP\project\Ultrasound_Dataset\Data\Data\test"

print("TRAIN_DIR:", TRAIN_DIR)
print("TEST_DIR :", TEST_DIR)


# IMAGE SETTINGS

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 16
EPOCHS = 5   # increased from 5

# DATA AUGMENTATION (IMPROVEMENT)
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)

train_data = train_datagen.flow_from_directory(
    directory=TRAIN_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode="grayscale",
    class_mode="categorical",
    batch_size=BATCH_SIZE
)

test_data = test_datagen.flow_from_directory(
    directory=TEST_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode="grayscale",
    class_mode="categorical",
    batch_size=BATCH_SIZE
)

print("Classes detected:", train_data.class_indices)


# CNN MODEL (IMPROVED)
model = Sequential([
    Conv2D(32, (3, 3), activation="relu",
           input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),            # prevents overfitting
    Dense(train_data.num_classes, activation="softmax")
])


# COMPILE MODEL
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# EARLY STOPPING (SMART TRAINING)
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)


# TRAINING
history = model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=test_data,
    callbacks=[early_stop]
)

# EVALUATION
loss, acc = model.evaluate(test_data)
print("Final Test Accuracy:", acc)


# PLOT TRAINING & VALIDATION GRAPHS

# Accuracy Graph
plt.figure()
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()
plt.show()

# Loss Graph
plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.show()
