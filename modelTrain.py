import os
import random
import string
import numpy as np
import cv2
from captcha.image import ImageCaptcha
from tensorflow.keras import layers, models
import tensorflow as tf

# Create dataset folder
data_dir = "captcha_dataset"
os.makedirs(data_dir, exist_ok=True)

# Generate random CAPTCHA text
def generate_text(length=5):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))

# Generate CAPTCHA images
def generate_captcha_dataset(num_images=5000, img_size=(200, 50)):
    image_gen = ImageCaptcha(width=img_size[0], height=img_size[1])
    
    for i in range(num_images):
        text = generate_text()
        image_path = os.path.join(data_dir, f"{text}.png")
        image = image_gen.generate_image(text)
        image.save(image_path)
    print(f"Generated {num_images} CAPTCHA images in {data_dir}")

generate_captcha_dataset()

# Load and preprocess images
def load_data(data_dir, img_size=(200, 50)):
    X, Y = [], []
    
    for file in os.listdir(data_dir):
        if file.endswith(".png"):
            img = cv2.imread(os.path.join(data_dir, file), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, img_size)
            img = img / 255.0  # Normalize
            X.append(img.reshape(img_size[1], img_size[0], 1))
            Y.append(list(file.split(".")[0]))  # Extract text label as list of characters
    return np.array(X), np.array(Y, dtype=object)

X_train, Y_train = load_data(data_dir)

# Define input shape and character classes
input_shape = (50, 200, 1)
num_classes = 36  # 26 letters + 10 digits

# Build CNN + LSTM Model
def build_model():
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Reshape(target_shape=(-1, 128))(x)
    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.LSTM(64, return_sequences=True)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

model = build_model()

# Train the model
model.fit(X_train, np.array([ord(c) for c in Y_train.flatten()]), epochs=10, batch_size=32, validation_split=0.2)

# Save model
model.save("captcha_model.h5")
print("Model training complete and saved as captcha_model.h5")
