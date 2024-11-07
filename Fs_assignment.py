

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


image_folder = r'C:\assignment 1\archive\Doctor’s Handwritten Prescription BD dataset\Training\training_words'
label_file = r"C:\assignment 1\archive\Doctor’s Handwritten Prescription BD dataset\Training\training_labels.csv"

labels_df = pd.read_csv(label_file)
print("Column names in the CSV file:", labels_df.columns)

images = []
labels = []

for index, row in labels_df.iterrows():
    img_path = os.path.join(image_folder, row['IMAGE'])
    img = load_img(img_path, target_size=(224, 224))  # Resize to VGG16 input size
    img = img_to_array(img)
    images.append(img)
    labels.append(row['MEDICINE_NAME'])

images = np.array(images, dtype="float") / 255.0  # Normalize pixel values
labels = np.array(labels)


label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
num_classes = len(np.unique(labels_encoded))  # Dynamically determine number of classes
labels_categorical = to_categorical(labels_encoded, num_classes=num_classes)


X_train, X_test, y_train, y_test = train_test_split(images, labels_categorical, test_size=0.2, random_state=42)


base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


for layer in base_model.layers:
    layer.trainable = False


x = base_model.output
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=outputs)


model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=23, batch_size=32)

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

model.save('vgg16_model.keras')




import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
import tensorflow as tf 

def load_and_prepare_test_data(test_image_folder, test_label_file, target_size=(224, 224)):
   
    test_labels_df = pd.read_csv(test_label_file)

    filename_col = 'IMAGE'
    label_col = 'MEDICINE_NAME'

    
    test_images = []
    test_labels = []

    
    for index, row in test_labels_df.iterrows():
        img_path = os.path.join(test_image_folder, row[filename_col])
        img = load_img(img_path, target_size=target_size)  
        img = img_to_array(img)
        test_images.append(img)
        test_labels.append(row[label_col])

   
    test_images = np.array(test_images, dtype="float") / 255.0  # Normalize pixel values
    test_labels = np.array(test_labels)

  
    label_encoder = LabelEncoder()
    test_labels_encoded = label_encoder.fit_transform(test_labels)
    test_labels_categorical = tf.keras.utils.to_categorical(test_labels_encoded, num_classes=len(label_encoder.classes_))

    return test_images, test_labels_categorical, label_encoder

def evaluate_model(model_path, test_images, test_labels_categorical, label_encoder):
    
    model = load_model(model_path)

    
    loss, accuracy = model.evaluate(test_images, test_labels_categorical)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

 
    predictions = model.predict(test_images)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(test_labels_categorical, axis=1)

  
    from sklearn.metrics import classification_report, confusion_matrix
    print(classification_report(true_classes, predicted_classes, target_names=label_encoder.classes_))

    conf_matrix = confusion_matrix(true_classes, predicted_classes)
    print("Confusion Matrix:")
    print(conf_matrix)


test_image_folder = r'C:\assignment 1\archive\Doctor’s Handwritten Prescription BD dataset\Testing\testing_words'
test_label_file = r"C:\assignment 1\archive\Doctor’s Handwritten Prescription BD dataset\Testing\testing_labels.csv"
model_path = 'vgg16_model.keras'


test_images, test_labels_categorical, label_encoder = load_and_prepare_test_data(test_image_folder, test_label_file)

evaluate_model(model_path, test_images, test_labels_categorical, label_encoder)





import os
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model


test_image_folder = r'C:\assignment 1\archive\Doctor’s Handwritten Prescription BD dataset\Testing\testing_words'
test_label_file = r"C:\assignment 1\archive\Doctor’s Handwritten Prescription BD dataset\Testing\testing_labels.csv"


test_labels_df = pd.read_csv(test_label_file)


filename_col = 'IMAGE'
label_col = 'MEDICINE_NAME'

print("First few rows of the test dataframe:")
print(test_labels_df.head())


test_images = []
test_labels = []


for index, row in test_labels_df.iterrows():
    img_path = os.path.join(test_image_folder, row[filename_col])
    img = load_img(img_path, target_size=(224, 224))  
    img = img_to_array(img)
    test_images.append(img)
    test_labels.append(row[label_col])


test_images = np.array(test_images, dtype="float") / 255.0  
test_labels = np.array(test_labels)


label_encoder = LabelEncoder()
test_labels_encoded = label_encoder.fit_transform(test_labels)
test_labels_categorical = tf.keras.utils.to_categorical(test_labels_encoded, num_classes=78)  # 78 classes


model = load_model('vgg16_model.keras')


loss, accuracy = model.evaluate(test_images, test_labels_categorical)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

from sklearn.metrics import classification_report, confusion_matrix


predictions = model.predict(test_images)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(test_labels_categorical, axis=1)


print(classification_report(true_classes, predicted_classes, target_names=label_encoder.classes_))

conf_matrix = confusion_matrix(true_classes, predicted_classes)
print("Confusion Matrix:")
print(conf_matrix)







