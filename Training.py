import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import splitfolders

#splitfolders.ratio('Datos', output="Datos/shuffled", seed=1337, ratio=(.8, 0.1,0.1))
# Load the datasetw     
train_dir = 'Datos/shuffled/train'
test_dir = 'Datos/shuffled/test'

# Define the image dimensions
img_height, img_width = 250, 250

# Define the data generators
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=64,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=64,
    class_mode='binary'
)

# Define the model architecture
model = keras.Sequential([
    keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1, activation='sigmoid') 
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_generator, epochs=10, validation_data=test_generator)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_generator)
print('Test accuracy:', test_acc)

# Use the model to make predictions
predictions = model.predict(test_generator)

# Convert the predictions to labels
predicted_labels = (predictions > 0.5).astype(int)

# Evaluate the model using accuracy score and confusion matrix
test_acc = accuracy_score(test_generator.classes, predicted_labels)
print('Accuracy:', test_acc)
conf_matrix = confusion_matrix(test_generator.classes, predicted_labels)
print(conf_matrix)
