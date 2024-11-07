#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import VGG16

# Define the paths to the images and labels
image_folder = r'C:\assignment 1\archive\Doctor’s Handwritten Prescription BD dataset\Training\training_words'
label_file = r"C:\assignment 1\archive\Doctor’s Handwritten Prescription BD dataset\Training\training_labels.csv"

# Load labels
labels_df = pd.read_csv(label_file)
print("Column names in the CSV file:", labels_df.columns)

# Initialize lists for images and labels
images = []
labels = []

# Load images and their corresponding labels
for index, row in labels_df.iterrows():
    img_path = os.path.join(image_folder, row['IMAGE'])
    img = load_img(img_path, target_size=(224, 224))  # Resize to VGG16 input size
    img = img_to_array(img)
    images.append(img)
    labels.append(row['MEDICINE_NAME'])

# Convert lists to numpy arrays
images = np.array(images, dtype="float") / 255.0  # Normalize pixel values
labels = np.array(labels)

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
num_classes = len(np.unique(labels_encoded))  # Dynamically determine number of classes
labels_categorical = to_categorical(labels_encoded, num_classes=num_classes)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(images, labels_categorical, test_size=0.2, random_state=42)

# Load the VGG16 model without the top layer (fully connected layers)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model
for layer in base_model.layers:
    layer.trainable = False

# Create a new model on top of the VGG16 base
x = base_model.output
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=outputs)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=23, batch_size=32)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Save the model
model.save('vgg16_model.keras')


# In[6]:


import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
import tensorflow as tf  # Ensure TensorFlow is imported

def load_and_prepare_test_data(test_image_folder, test_label_file, target_size=(224, 224)):
    # Load test labels
    test_labels_df = pd.read_csv(test_label_file)

    # Column names in the CSV file
    filename_col = 'IMAGE'
    label_col = 'MEDICINE_NAME'

    # Initialize lists for test images and labels
    test_images = []
    test_labels = []

    # Load test images and their corresponding labels
    for index, row in test_labels_df.iterrows():
        img_path = os.path.join(test_image_folder, row[filename_col])
        img = load_img(img_path, target_size=target_size)  # Resize to a fixed size
        img = img_to_array(img)
        test_images.append(img)
        test_labels.append(row[label_col])

    # Convert lists to numpy arrays
    test_images = np.array(test_images, dtype="float") / 255.0  # Normalize pixel values
    test_labels = np.array(test_labels)

    # Encode test labels
    label_encoder = LabelEncoder()
    test_labels_encoded = label_encoder.fit_transform(test_labels)
    test_labels_categorical = tf.keras.utils.to_categorical(test_labels_encoded, num_classes=len(label_encoder.classes_))

    return test_images, test_labels_categorical, label_encoder

def evaluate_model(model_path, test_images, test_labels_categorical, label_encoder):
    # Load the trained model
    model = load_model(model_path)

    # Evaluate the model on the test data
    loss, accuracy = model.evaluate(test_images, test_labels_categorical)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

    # Predict the classes
    predictions = model.predict(test_images)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(test_labels_categorical, axis=1)

    # Generate classification report
    from sklearn.metrics import classification_report, confusion_matrix
    print(classification_report(true_classes, predicted_classes, target_names=label_encoder.classes_))

    # Generate confusion matrix
    conf_matrix = confusion_matrix(true_classes, predicted_classes)
    print("Confusion Matrix:")
    print(conf_matrix)

# Paths to your test data
test_image_folder = r'C:\assignment 1\archive\Doctor’s Handwritten Prescription BD dataset\Testing\testing_words'
test_label_file = r"C:\assignment 1\archive\Doctor’s Handwritten Prescription BD dataset\Testing\testing_labels.csv"
model_path = 'vgg16_model.keras'

# Load and prepare test data
test_images, test_labels_categorical, label_encoder = load_and_prepare_test_data(test_image_folder, test_label_file)

# Evaluate the model on the new test data
evaluate_model(model_path, test_images, test_labels_categorical, label_encoder)


# In[8]:


import os
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model

# Define the paths to the test images and labels
test_image_folder = r'C:\assignment 1\archive\Doctor’s Handwritten Prescription BD dataset\Testing\testing_words'
test_label_file = r"C:\assignment 1\archive\Doctor’s Handwritten Prescription BD dataset\Testing\testing_labels.csv"

# Load test labels
test_labels_df = pd.read_csv(test_label_file)

# Column names in the CSV file
filename_col = 'IMAGE'
label_col = 'MEDICINE_NAME'

# Debug: print the first few rows to understand the structure
print("First few rows of the test dataframe:")
print(test_labels_df.head())

# Initialize lists for test images and labels
test_images = []
test_labels = []

# Load test images and their corresponding labels
for index, row in test_labels_df.iterrows():
    img_path = os.path.join(test_image_folder, row[filename_col])
    img = load_img(img_path, target_size=(224, 224))  # Resize to a fixed size
    img = img_to_array(img)
    test_images.append(img)
    test_labels.append(row[label_col])

# Convert lists to numpy arrays
test_images = np.array(test_images, dtype="float") / 255.0  # Normalize pixel values
test_labels = np.array(test_labels)

# Encode test labels
label_encoder = LabelEncoder()
test_labels_encoded = label_encoder.fit_transform(test_labels)
test_labels_categorical = tf.keras.utils.to_categorical(test_labels_encoded, num_classes=78)  # 78 classes

# Load the trained model
model = load_model('vgg16_model.keras')

# Evaluate the model on the test data
loss, accuracy = model.evaluate(test_images, test_labels_categorical)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Optionally: Predict and show a confusion matrix
from sklearn.metrics import classification_report, confusion_matrix

# Predict the classes
predictions = model.predict(test_images)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(test_labels_categorical, axis=1)

# Generate classification report
print(classification_report(true_classes, predicted_classes, target_names=label_encoder.classes_))

# Generate confusion matrix
conf_matrix = confusion_matrix(true_classes, predicted_classes)
print("Confusion Matrix:")
print(conf_matrix)


# In[ ]:




