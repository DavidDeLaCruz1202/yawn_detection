"""
Author: David De La Cruz
Date Created: 5/1/2025
Filename: yawn_detector.py
Description: A file that allows the camera to be used, and for a yawn to be detected.
"""

import keras
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
#import cv2
import random
from PIL import Image

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
#from keras.models import load_model


# Store images as arrays in lists
yawning_images = []
not_yawning_images = []
reshape_dimensions = (50, 50)  # Width, height

print("\n>>> Building yawning array...")
for image in os.listdir(r"dataset/yawning"):
    # Read image and convert to grayscale
    opened_img = Image.open(f"dataset/yawning/{image}").convert("L")

    # Resize so all images can be same size, and to reduce number of layer connections in model
    opened_img = opened_img.resize(reshape_dimensions)

    # Convert image to a NumPy array
    img_array = np.array(opened_img)

    # Append to our list
    yawning_images.append(img_array)
print("+++ Complete! +++")


print("\n>>> Building not yawning array...")
for image in os.listdir(r"dataset/not_yawning"):
    # Read image and convert to grayscale
    opened_img = Image.open(f"dataset/not_yawning/{image}").convert("L")

    # Resize so all images can be same size, and to reduce number of layer connections in model
    opened_img = opened_img.resize(reshape_dimensions)

    # Convert image to a NumPy array
    img_array = np.array(opened_img)

    # Append to our list
    not_yawning_images.append(img_array)
print("+++ Complete! +++")


# -------------------------------
# Show statistics about image sizes
analyze = False
if analyze:
    print("\nAnalyzing image sizes of datasets...")
    x_s = []
    y_s = []
    for image in yawning_images:
        image_x, image_y = image.shape
        x_s.append(image_x)
        y_s.append(image_y)

    for image in not_yawning_images:
        image_x, image_y = image.shape
        x_s.append(image_x)
        y_s.append(image_y)

    x_pd_arr = pd.DataFrame(x_s)
    y_pd_arr = pd.DataFrame(y_s)

    print(x_pd_arr.describe())
    print(y_pd_arr.describe())
# -------------------------------

print("\n>>> Building x and y datasets...")
# Building x and y datasets
num_not_yawning = len(not_yawning_images)
num_yawning = len(yawning_images)
labels = []

# 0 = not yawning
for i in range(num_not_yawning):
    labels.append(0)

# 1 = yawning
for i in range(num_yawning):
    labels.append(1)

x = not_yawning_images + yawning_images
y = labels
print("+++ Complete! +++")


print("\n>>> Splitting data into training and testing datasets...")
# Randomly splitting data into training and testing datasets
random.seed(1202)

# This line defines the indices that we're going to be using for our test data
# Specifically, it gets a random sample of 20% of the available indices and converts it into a set (For optimized member checking)
x_test_indices = set(random.sample(range(len(x)), len(x) // 5))

x_train, x_test, y_train, y_test = [], [], [], []

for i in range(len(x)):
    if i in x_test_indices:
        x_test.append(x[i])
        y_test.append(y[i])
    else:  
        x_train.append(x[i])
        y_train.append(y[i])
print("+++ Complete! +++")


print("\n>>> Converting datasets to np arrays and reshaping...")
# Converting lists to np arrays
x_train, x_test, y_train, y_test = np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)

# Flatten images to one dimensional vectors
# Note: All images are the same size due to resizing in earlier parts, so we can do this with either x_train or x_test
pixel_count = x_train.shape[1] * x_train.shape[2]

x_train = x_train.reshape(x_train.shape[0], pixel_count).astype('float32')
x_test = x_test.reshape(x_test.shape[0], pixel_count).astype('float32')
print("+++ Complete! +++")

# Normalize the numbers so that instead of ranging from 0 - 255, it ranges from 0 - 1
print("\n>>> Normalizing x data and converting y to categories...")
x_train /= 255
x_test /= 255

# Use one-hot encoding to convert labels to categories
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
class_count = y_train.shape[1]
print("+++ Complete! +++")

print("\n>>> Building model...")
# Define classification model function
def classification_model():
    # Create model
    model = Sequential()

    # Add model layers
    model.add(Dense(pixel_count, activation='relu', input_shape=(pixel_count,)))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(class_count, activation='softmax'))
    
    # Compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Create instance of model
class_model = classification_model()
print("+++ Complete! +++")

print("\n>>> Training model...")
# Train the model
class_model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, verbose=2)
print("+++ Complete! +++")

print("\n>>> Evaluating model...")
# evaluate the model
scores = class_model.evaluate(x_test, y_test, verbose=0)
print("+++ Complete! +++")

print(f'\nAccuracy: {scores[1]}% \n Error: {1 - scores[1]}')

#class_model.save('yawn_classification_model.h5')
#pretrained_model = load_model('yawn_classification_model.h5')